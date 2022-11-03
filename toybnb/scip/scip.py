import numpy as np
from scipy import sparse as sp
from collections import defaultdict

from scipy.optimize import OptimizeResult
from pyscipopt import Model, quicksum, SCIP_STATUS

from .cip import from_cip
from ..milp import MILP


TO_NUMERIC_STATUS = {
    "optimal": SCIP_STATUS.OPTIMAL,
    "infeasible": SCIP_STATUS.INFEASIBLE,
    "unbounded": SCIP_STATUS.UNBOUNDED,
    "inforunbd": SCIP_STATUS.INFORUNBD,
    "timelimit": SCIP_STATUS.TIMELIMIT,
    "userinterrupt": SCIP_STATUS.USERINTERRUPT,
    "nodelimit": SCIP_STATUS.NODELIMIT,
    "totalnodelimit": SCIP_STATUS.TOTALNODELIMIT,
    "stallnodelimit": SCIP_STATUS.STALLNODELIMIT,
    "gaplimit": SCIP_STATUS.GAPLIMIT,
    "memlimit": SCIP_STATUS.MEMLIMIT,
    "sollimit": SCIP_STATUS.SOLLIMIT,
    "bestsollimit": SCIP_STATUS.BESTSOLLIMIT,
    "restartlimit": SCIP_STATUS.RESTARTLIMIT,
}


def to_scip(p: MILP, **params: dict) -> Model:
    """Convert our own MILP to SCIP model."""
    m = Model("MILP")
    m.setParams(params)

    x = {}
    for j in range(p.n):
        # the first `m` variables are integer-valued, the rest are continuous
        x[j] = m.addVar(f"x[{j}]", "I" if j < p.m else "C", *p.bounds[j])

    if sp.issparse(p.A_ub):
        for k in range(p.A_ub.shape[0]):
            csr = p.A_ub.getrow(k)
            Ax_k = quicksum(v * x[j] for j, v in zip(csr.indices, csr.data))
            m.addCons(Ax_k <= p.b_ub[k], f"A_ub[{k}, j] x[j] <= b_ub[{k}]")

    elif p.A_ub is not None:
        A = p.A_ub
        for k in range(A.shape[0]):
            Ax_k = quicksum(A[k, j] * x[j] for j in range(A.shape[1]))
            m.addCons(Ax_k <= p.b_ub[k], f"A_ub[{k}, j] x[j] <= b_ub[{k}]")

    if sp.issparse(p.A_eq):
        for k in range(p.A_eq.shape[0]):
            csr = p.A_eq.getrow(k)
            Ax_k = quicksum(v * x[j] for j, v in zip(csr.indices, csr.data))
            m.addCons(Ax_k == p.b_eq[k], f"A_eq[{k}, j] x[j] == b_eq[{k}]")

    elif p.A_eq is not None:
        A = p.A_eq
        for k in range(A.shape[0]):
            Ax_k = quicksum(A[k, j] * x[j] for j in range(A.shape[1]))
            m.addCons(Ax_k == p.b_eq[k], f"A_eq[{k}, j] x[j] == b_eq[{k}]")

    # our MILP problems are always minimization
    m.setObjective(quicksum(p.c[j] * v for j, v in x.items()), "minimize")

    return m


def get_result(m: Model) -> OptimizeResult:
    """Extract the best solution found by SCIP as an OptimizeResult."""
    sol = m.getBestSol()

    # if, when creating the instance, the variables were named `var[i, j, ...]`,
    #  then we can let python interpreter do all the parsing and assignment
    # XXX we use default dict for BIGGER brain
    scope = defaultdict(dict)
    for var in m.getVars():
        scope["__var__"] = sol[var]
        # XXX using specialized dict scope is safer when executing less trusted code
        exec(f"{var.name} = __var__", None, scope)  # XXX locals() for shorter body

    del scope["__var__"]

    # check the collected vars
    sol = scope.pop("x")
    assert not scope

    # reconstruct the solution. We use a loop just in case SCIP
    #  reports variables in arbitrary order
    x = np.full(len(sol), np.nan, float)
    for k, v in sol.items():
        x[k] = v

    message = m.getStatus()
    status = TO_NUMERIC_STATUS[message]
    return OptimizeResult(
        x=x,
        fun=m.getObjVal(),
        success=status == SCIP_STATUS.OPTIMAL,
        status=status,
        message=f"{message} ({status})",
        nit=m.getNLPIterations(),
    )


def from_scip(m: Model) -> MILP:
    import os
    from tempfile import mkstemp

    """Convert a SCIP MILP model into out own MILP by parsinga CIP file."""
    try:
        fd, cip = mkstemp(suffix=".cip")
        m.writeProblem(cip, False, False)
        return from_cip(cip)

    finally:
        os.close(fd)
        os.unlink(cip)
