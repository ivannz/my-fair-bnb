import torch

from torch import Tensor, nn
from torch.nn import Module, functional as F

from math import sqrt

from einops import rearrange
from torch_scatter import scatter_softmax, scatter_sum, scatter_mean
from torch_scatter import scatter_log_softmax, scatter_logsumexp, scatter_max

from .data import BatchObservation
from .brancher import BaseNeuralBranchRuleMixin


# the layers proper are after the mixin definition
class NeuralClassifierBranchruleMixin(BaseNeuralBranchRuleMixin):
    """A neural branchrule based on a multiclass classifier"""

    def compute(
        self, input: BatchObservation, target: Tensor = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # optionally compute the loss
        raw = self(input)
        if target is None:
            return raw, {}  # raw.new_full((), float("nan"))

        # get the log-probas and compute the loss terms
        logp_vars = scatter_log_softmax(
            raw,
            input.inx_vars,
            0,
            dim_size=len(input.ptr_vars) - 1,
        )

        # get the log-likelihood of the target variables
        # XXX `ptr_vars` corrects the variable indices in the target batch
        j = input.ptr_vars[:-1] + target.to(logp_vars.device)
        loglik_target = logp_vars[j]

        # get the log likelihood of the forbidden variable set
        mask = torch.ones_like(logp_vars, dtype=bool)
        mask[input.actset] = False
        logp_forbidden = scatter_logsumexp(
            logp_vars[mask],
            input.inx_vars[mask],  # XXX group by batch assignment
            dim=0,
            # It is possible that the forbidden mask to be empty in certain
            #  nodes of certain instances
            # XXX ptr_* is one longer than the batch size
            dim_size=len(input.ptr_vars) - 1,
        )

        # XXX no need to compute either the act-set sizes
        #  or the vars sizes with `input.ptr_vars.diff()`
        # n_forbidden_size = scatter_sum(mask.float(), input.inx_vars, 0)
        loglik_actset = logp_forbidden  # .div(n_forbidden_size)

        # compute the discrete entropy $- \sum_j \pi_j \log \pi_j$
        # XXX For input x and target y, `F.kl_div(x, y, "none", log_target=True)`
        #  returns $e^y (y - x)$, correctly handling infinite `-ve` logits.
        #  `log_target` relates to `y` being logits (True) or probas (False).
        # XXX `.new_zeros(())` creates a scalar zero (yes, an EMPTY tuple)
        # XXX from nle-toolbox
        zero = logp_vars.new_zeros(())
        p_log_p = F.kl_div(zero, logp_vars, reduction="none", log_target=True)
        entropy = scatter_sum(
            p_log_p,
            input.inx_vars,
            0,
            dim_size=len(input.ptr_vars) - 1,
        ).neg()

        return raw, dict(
            neg_target=loglik_target.neg(),
            neg_actset=loglik_actset.neg(),
            entropy=entropy,
        )

    def predict(self, input: BatchObservation) -> Tensor:
        with torch.inference_mode():
            scores = self(input)

        # mask forbidden variables
        mask = torch.ones_like(scores, dtype=bool)
        mask[input.actset] = False
        scores.masked_fill_(mask, float("-inf"))

        # pick the maximally probable action in each batch item
        _, j = scatter_max(
            scores,
            input.inx_vars,
            0,
            # fills empty groups' arg with a meaningless index beyond `scores`
            dim_size=len(input.ptr_vars) - 1,
        )
        if mask[j].any():
            raise RuntimeError("Empty actset!")  # sanity check

        # subtract the base index of each batch item
        j -= input.ptr_vars[:-1]
        return j


def mlp(activation: type = nn.ReLU, /, *n_dims: int) -> nn.Sequential:
    """multi-layer perceptrons are magic"""
    layers = []
    for d0, d1 in zip(n_dims, n_dims[1:]):
        layers.append(nn.Linear(d0, d1))
        layers.append(activation())

    return nn.Sequential(*layers[:-1])


class BipartiteGConv(Module):
    """Bipartite grpah convolution arch compatible with Gasse et al. 2019"""

    def __init__(self, n_embed: int = 64, b_edges: bool = True) -> None:
        super().__init__()
        self.input = nn.Linear(n_embed, n_embed)
        # self.input = mlp(nn.LeakyReLU, n_embed, 4 * n_embed, n_embed)
        if b_edges:
            self.edges = nn.Linear(1, n_embed, bias=False)
            # self.edges = mlp(nn.LeakyReLU, 1, 4 * n_embed, n_embed)
        else:
            self.register_module("edges", None)

        self.other = nn.Linear(n_embed, n_embed, bias=False)
        # self.other = mlp(nn.LeakyReLU, n_embed, 4 * n_embed, n_embed)

        self.final = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(n_embed, n_embed),
        )

        self.output = mlp(nn.LeakyReLU, 2 * n_embed, n_embed)

    def forward(
        self,
        input: Tensor,
        other: Tensor,
        coupling: tuple[Tensor, Tensor],
        weights: Tensor = None,
    ) -> Tensor:
        """Bipartite graph convolution from `other` into `input`"""
        # `coupling` specifies which `other` sends message to which `input`
        #  XXX (u, v) represents a directed edge `u to v` and means
        #  that `input[u]` attends to `other[v]` and sources data from it.
        rj, lj = coupling  # XXX rj (in input) <<-- lj (in other)

        # transform the tensors in parts
        rhs = self.input(input)
        lhs = self.other(other)

        # compute the messages (materialized) and optionaly incldue edges
        message = rhs[rj] + lhs[lj]
        if self.edges is not None:
            message = message + self.edges(weights)
        msg = self.final(message)

        # issue the messages to the target (new inputs)
        # XXX `scatter_sum(x, j)` computes `y_b = \sum_{a: j_a = b} x_a`
        # XXX `y[j[a]] += x[a]`
        out = scatter_sum(msg, rj, dim=0, dim_size=len(rhs))
        return self.output(torch.cat([out, input], dim=-1))


class BipartiteMHXA(Module):
    """Multiheaded cross-attention for bipartite graph"""

    def __init__(
        self, n_embed: int = 32, n_heads: int = 1, b_edges: bool = False
    ) -> None:
        super().__init__()
        self.n_heads = n_heads

        # query projection for `input`
        self.p_q = nn.Linear(n_embed, n_embed, bias=False)

        # key-value projection for `other`
        self.p_kv = nn.Linear(n_embed, 2 * n_embed, bias=False)

        # the final head output mixer
        self.out = nn.Linear(n_embed, n_embed)

        # edge data is used to multiplicatively modulate cross-attention scores
        if b_edges:
            raise NotImplementedError
            # self.edges = mlp(nn.LeakyReLU, 1, 4 * n_embed, n_heads)

        else:
            self.register_module("edges", None)

    def forward(
        self,
        input: Tensor,
        other: Tensor,
        coupling: tuple[Tensor, Tensor],
        weights: Tensor = None,
    ) -> Tensor:
        """Cross attention form `input` (query) to `other` (keys and values)"""
        if weights is not None:
            raise NotImplementedError

        # `coupling` specifies which `input` attends to which `other`
        #  XXX (u, v) represents a directed edge `u to v` and means
        #  that `input[u]` attends to `other[v]` and sources data from it.
        t, s = coupling  # XXX t (in input) <<-- s (in other)

        # get the qkv vectors, properly reshaped for multi-headed attention
        q = rearrange(self.p_q(input), "N (h f) -> N h () f", h=self.n_heads)
        k, v = self.p_kv(other).chunk(2, -1)
        k = rearrange(k, "N (h f) -> N h f ()", h=self.n_heads)
        v = rearrange(v, "N (h f) -> N h f", h=self.n_heads)

        # q is N x H x 1 x F, k is N x H x F x 1
        # XXX indexing by `t` and `s` materializes potentially large tensors!
        score = torch.matmul(q[t], k[s]).div(sqrt(q.shape[-1])).squeeze(-1)

        # softmax over `a` in `(q_b k_{i_a})_{a \colon j_a = b}`
        # XXX `scatter_softmax(x, j)` computes softmax over `x` grouped by j
        #    `y[j==b] = softmax(x[j==b])` for all b in j's range
        # XXX `q[b].matmul(k[i[j==b]])`
        alpha = scatter_softmax(score, t, dim=0, dim_size=len(q))

        # `scatter_sum(x, j)` computes `y_b = \sum_{a: j_a = b} x_a`
        # XXX `y[j[a]] += x[a]`
        out = scatter_sum(alpha * v[s], t, dim=0, dim_size=len(q))
        return self.out(rearrange(out, "... h f -> ... (h f)"))


class BipartiteBlock(Module):
    def __init__(
        self,
        n_embed: int = 32,
        n_heads: int = 1,
        p_drop: float = 0.2,
        b_norm_first: bool = False,
        b_edges: bool = False,
    ) -> None:
        super().__init__()
        self.b_norm_first = b_norm_first

        self.attn = BipartiteMHXA(n_embed, n_heads, b_edges)
        self.do_1 = nn.Dropout(p_drop)
        self.ln_1 = nn.LayerNorm(n_embed)

        self.pwff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.LeakyReLU(),
            nn.Dropout(p_drop),
            nn.Linear(4 * n_embed, n_embed),
        )
        self.do_2 = nn.Dropout(p_drop)
        self.ln_2 = nn.LayerNorm(n_embed)

    def forward(
        self,
        input: Tensor,
        other: Tensor,
        coupling: tuple[Tensor, Tensor],
        weights: Tensor = None,
    ) -> Tensor:
        x = input
        if self.b_norm_first:
            x = x + self.do_1(self.attn(self.ln_1(x), other, coupling, weights))
            return x + self.do_2(self.pwff(self.ln_2(x)))

        x = self.ln_1(x + self.do_1(self.attn(x, other, coupling, weights)))
        return self.ln_2(x + self.do_2(self.pwff(x)))


class NeuralVariableSelector(Module, NeuralClassifierBranchruleMixin):
    """The cross-attention transformer-based variable scoring model"""

    def __init__(
        self,
        n_dim_vars: int = 19,
        n_dim_cons: int = 5,
        n_embed: int = 32,
        n_heads: int = 1,
        n_blocks: int = 1,
        p_drop: float = 0.2,
        b_norm_first: bool = False,
        s_project_graph: str = "none",
        b_edges: bool = False,
    ) -> None:
        assert s_project_graph in ("pre", "post", "none")

        super().__init__()
        self.encoder = nn.ModuleDict(
            dict(
                vars=mlp(nn.LeakyReLU, n_dim_vars, 4 * n_embed, n_embed),
                cons=mlp(nn.LeakyReLU, n_dim_cons, 4 * n_embed, n_embed),
                edge=None,
                # edge=mlp(nn.LeakyReLU, 1, 4 * n_embed, n_embed)
                # XXX this mlp is incompatible with BipartiteMHXA
            )
        )

        blk = [
            BipartiteBlock(n_embed, n_heads, p_drop, b_norm_first, b_edges)
            for _ in range(n_blocks)
        ]
        self.block_cv = nn.ModuleList(blk)
        blk = [
            BipartiteBlock(n_embed, n_heads, p_drop, b_norm_first, b_edges)
            for _ in range(n_blocks)
        ]
        self.block_vc = nn.ModuleList(blk)

        self.head = mlp(nn.LeakyReLU, 2 * n_embed, 4 * n_embed, 1)
        self.s_project_graph = s_project_graph
        if s_project_graph != "none":
            self.projection = mlp(nn.LeakyReLU, n_embed, 4 * n_embed, n_embed)

        else:
            self.register_module("projection", None)

    def forward(self, input: BatchObservation) -> Tensor:
        jc, jv = input.ctov_ij

        # encode the vars and cons features
        cons = self.encoder.cons(input.cons)
        vars = self.encoder.vars(input.vars)
        edge = None
        if self.encoder.edge is not None:
            edge = self.encoder.edge(input.ctov_v)

        # bipartite vcv ladder
        # (v_{t-1}, c_{t-1}) -->> (v_{t-1}, c_t) -->> (v_t, c_t)
        for m_cv, m_vc in zip(self.block_cv, self.block_vc):
            cons = m_cv(cons, vars, (jc, jv), edge)
            vars = m_vc(vars, cons, (jv, jc), edge)

        # compute the graph-level variable embedding
        if self.projection is not None:
            if self.s_project_graph == "post":
                graph = self.projection(
                    scatter_mean(
                        vars,
                        input.inx_vars,
                        dim=0,
                        dim_size=len(input.ptr_vars) - 1,
                    )
                )

            elif self.s_project_graph == "pre":
                graph = scatter_mean(
                    self.projection(vars),
                    input.inx_vars,
                    dim=0,
                    dim_size=len(input.ptr_vars) - 1,
                )

            else:
                raise NotImplementedError

        # get the raw-logit scores of each variable
        x = torch.cat((vars, graph[input.inx_vars]), -1)
        return self.head(x).squeeze(-1)
