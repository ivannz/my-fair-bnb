import numpy as np
from numpy import ndarray

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


def is_feasible_box(
    x: ndarray, p: MILP, rtol: float = 1e-5, atol: float = 1e-8
) -> ndarray:
    """Check bounding box constraints."""
    lo, hi = p.bounds.T  # N x 2

    # computing constraint slacks and fp-comparing to zero is slow
    # XXX this should be cythonized. Also we can ignore rtols, since
    #  we can subtract and compare to zero
    return np.logical_and(
        # np.isclose(np.maximum(x - hi, 0.0), 0.0, rtol, atol),
        lo - x <= atol,  # lo - abs(lo) * rtol - atol <= x,
        # np.isclose(np.maximum(lo - x, 0.0), 0.0, rtol, atol),
        x - hi <= atol,  # x <= hi + abs(hi) * rtol + atol,
    )


def is_feasible_ub(
    x: ndarray, p: MILP, rtol: float = 1e-5, atol: float = 1e-8
) -> ndarray:
    """Check the affine upper bound inequality constraints."""
    if p.A_ub is None or p.b_ub is None:
        return np.array([], bool)

    # A_ub x \leq b_ub
    # np.isclose(np.maximum(p.A_ub.dot(x), p.b_ub), p.b_ub, rtol, atol)
    # p.A_ub.dot(x) <= (p.b_ub + abs(p.b_ub) * rtol + atol)
    return p.A_ub.dot(x) - p.b_ub <= atol


def is_feasible_eq(
    x: ndarray, p: MILP, rtol: float = 1e-5, atol: float = 1e-8
) -> ndarray:
    """Check the linear equality constraints."""
    if p.A_eq is None or p.b_eq is None:
        return np.array([], bool)

    # A_eq x = b_eq
    # np.isclose(p.A_eq.dot(x) - p.b_eq, 0.0, rtol, atol)
    return abs(p.A_eq.dot(x) - p.b_eq) <= atol


def is_feasible_int(
    x: ndarray, p: MILP, rtol: float = 1e-5, atol: float = 1e-8
) -> ndarray:
    """Check integer feasibility."""
    # test integrality of all variables, then force continuous to True
    # np.isclose(x.round(), x, rtol, atol)
    feasible = abs(x.round() - x) <= atol
    feasible[p.m :] = True
    return feasible


def is_feasible(
    x: ndarray, p: MILP, rtol: float = 1e-5, atol: float = 1e-6
) -> tuple[bool]:
    """Check feasibility of the solution."""
    ok_bnd = is_feasible_box(x, p, rtol, atol).all()
    ok_aub = is_feasible_ub(x, p, rtol, atol).all()
    ok_aeq = is_feasible_eq(x, p, rtol, atol).all()
    ok_int = is_feasible_int(x, p, rtol, atol).all()
    return ok_bnd, ok_aub, ok_aeq, ok_int
