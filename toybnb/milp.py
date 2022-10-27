import numpy as np
from numpy.random import default_rng
from collections import namedtuple

MILP = namedtuple("MILP", "c,A_ub,b_ub,A_eq,b_eq,bounds,n,m")


def generate(n: int, m: int, r: int, *, seed: int = None) -> MILP:
    r"""A random instance of a generic MILP
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
