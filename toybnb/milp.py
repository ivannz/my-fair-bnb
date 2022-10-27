import numpy as np
import scipy.sparse as sp
import networkx as nx

from numpy.random import default_rng
from collections import namedtuple

MILP = namedtuple("MILP", "c,A_ub,b_ub,A_eq,b_eq,bounds,n,m")


def generate(n: int, m: int, r: int, *, seed: int = None) -> MILP:
    r"""A random instance of a generic MILP of fixed size and structure
        $$
        \min\Bigl\{
            c^\top x
            \colon
            A x \leq b
            \,, x \in \bigl[ l, u \bigr]
            \,, x \in \mathbb{Z}^m \times \mathbb{R}^{n-m}
        \Bigr\}
            \,, $$
    where $
        A \in \mathbb{R}^{r \times n}
    $, $
        b \in \mathbb{R}^r
    $, $
        l, u \in \mathbb{R}^n
    $ with $l \leq u$ and $1 \leq m \leq n$.
    """

    rng = default_rng(seed)
    while True:
        # bounds N x 2 array form
        lu = rng.normal(size=(n, 2)) * 10
        bounds = np.stack((np.floor(lu.min(-1)), np.ceil(lu.max(-1))), axis=-1)

        # affine upper bound
        A_ub, _ = np.linalg.qr(rng.normal(size=(n, r)))
        b_ub = rng.normal(size=min(n, r))

        # affine subspace constraints
        A_eq, b_eq = np.empty((0, n)), np.empty((0,))

        # cost
        c = rng.normal(size=n)
        yield MILP(c, A_ub.T.copy(), b_ub, A_eq, b_eq, bounds, n, m)


def incidence(G: nx.Graph) -> sp.coo_matrix:
    """Get CSR sparse edge incidence matrix."""
    eu, ev = map(np.array, zip(*G.edges))
    e, one = np.arange(len(eu)), np.ones(len(eu))

    E = sp.coo_matrix((one, (e, eu)), shape=(len(e), len(G)))
    E += sp.coo_matrix((one, (e, ev)), shape=E.shape)

    return E.tocsr()


def generate_mis_ba(n: int = 400, m: int = 4, seed: int = None) -> MILP:
    r"""A Maximum Independent Set problem in a Barabasi-Albert graph of fixed size.

    For a undirected graph $G = (V, E)$ the Maximum Independent Set problem is
        $$
        \begin{aligned}
            & \underset{x\in \{0, 1\}}{\text{maximize}}
              & & \sum_{v \in G} x_v
                  \\
            & \text{subject to}
              & & \forall uv \in E
                  \colon x_u + x_v \leq 1
                  \,.
        \end{aligned}
            $$
    """
    rng = default_rng(seed)
    nilone = np.full((n, 2), (0, 1), float)
    while True:
        # get BA with n nodes and m preferentially attached edges
        G = nx.barabasi_albert_graph(n, m, seed=rng)

        c = np.full(len(G), -1.0)
        A_ub = incidence(G)
        b_ub = np.ones(len(G.edges))
        A_eq = np.empty((0, n))
        b_eq = np.empty((0,))

        yield MILP(c, A_ub, b_ub, A_eq, b_eq, nilone, n, n)
