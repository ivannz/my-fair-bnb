from collections import defaultdict

import numpy as np
from pyscipopt import SCIP_STAGE, SCIP_STATUS, Model, quicksum
from pyscipopt.scip import PY_SCIP_LPSOLSTAT as SCIP_LPSOLSTAT
from scipy import sparse as sp
from scipy.optimize import OptimizeResult

from ..milp import MILP
from .cip import from_cip

# remap SCIP_LPSOLSTAT to scipy's status codes
# 0 success, 1 iteration/time limit, 2 infeasible, 3 unbounded, 4 other
SCIP_LPSOLSTAT_TO_NUMERIC = {
    SCIP_LPSOLSTAT.NOTSOLVED: 4,
    SCIP_LPSOLSTAT.OPTIMAL: 0,
    SCIP_LPSOLSTAT.ITERLIMIT: 1,
    SCIP_LPSOLSTAT.TIMELIMIT: 1,
    SCIP_LPSOLSTAT.INFEASIBLE: 2,
    SCIP_LPSOLSTAT.UNBOUNDEDRAY: 3,
    SCIP_LPSOLSTAT.OBJLIMIT: 4,
    SCIP_LPSOLSTAT.ERROR: 4,
}


STATUS_TO_NUMERIC = {
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
    status = STATUS_TO_NUMERIC[message]
    return OptimizeResult(
        x=x,
        fun=m.getObjVal(),
        success=status == SCIP_STATUS.OPTIMAL,
        status=status,
        message=f"{message} ({status})",
        nit=m.getNLPIterations(),
    )


def from_scip(
    m: Model, trans: bool = False, genericnames: bool = False, *, dir: str = None
) -> MILP:
    import os
    from tempfile import mkstemp

    """Convert a SCIP MILP model into out own MILP by parsinga CIP file."""
    try:
        fd, cip = mkstemp(suffix=".cip", dir=dir)
        os.close(fd)

        m.writeProblem(cip, trans, genericnames)
        return from_cip(cip)

    finally:
        if dir is None:
            os.unlink(cip)


def from_scip_lp(m: Model, *, safe: bool = True) -> MILP:
    """Read SCIP's current LP problem into MILP."""
    # it appears that pyscipopt can't read lp data if the state is too early
    if safe and m.getStage() < SCIP_STAGE.SOLVING:
        raise RuntimeError(
            "Avoiding segfault from a SCIP model, which is not at SOLVING stage"
        )

    v_int, v_con = [], []
    for col in m.getLPColsData():
        type = col.getVar().vtype().lower()
        if type not in ("binary", "integer", "continuous"):
            raise NotImplementedError(type)

        which = v_int if type in ("binary", "integer") else v_con
        which.append(col.getLPPos())

    # assign correct indices
    lut = {v: j for j, v in enumerate(v_int + v_con)}

    # get the constraint data
    tab_ub, b_ub = defaultdict(float), []
    tab_eq, b_eq = defaultdict(float), []
    for row in m.getLPRowsData():
        lb, ub, c0 = row.getLhs(), row.getRhs(), row.getConstant()
        if not m.isEQ(lb, ub):
            if not m.isInfinity(abs(ub)):
                # c_0 + a^\top x \leq u
                for c, v in zip(row.getCols(), row.getVals()):
                    tab_ub[len(b_ub), c.getLPPos()] += v
                b_ub.append(ub - c0)

            if not m.isInfinity(abs(lb)):
                # l \leq c_0 + a^\top x
                for c, v in zip(row.getCols(), row.getVals()):
                    tab_ub[len(b_ub), c.getLPPos()] -= v
                b_ub.append(c0 - lb)
        else:
            # u \leq c_0 + a^\top x \leq u
            for c, v in zip(row.getCols(), row.getVals()):
                tab_eq[len(b_eq), c.getLPPos()] += v
            b_eq.append(ub - c0)

    b_ub = np.array(b_ub)
    A_ub = sp.lil_array((len(b_ub), m.getNLPCols()), dtype=float)
    for (i, j), v in tab_ub.items():
        A_ub[i, lut[j]] = v

    b_eq = np.array(b_eq)
    A_eq = sp.lil_array((len(b_eq), m.getNLPCols()), dtype=float)
    for (i, j), v in tab_eq.items():
        A_ub[i, lut[j]] = v

    # get the objective data and bounds
    c0 = m.getObjoffset()
    c = np.full(m.getNLPCols(), np.nan)
    bounds = np.full((m.getNLPCols(), 2), (-np.inf, +np.inf))
    for col in m.getLPColsData():
        j = lut[col.getLPPos()]
        c[j] = col.getObjCoeff()
        bounds[j] = col.getLb(), col.getUb()

    return MILP(
        c,
        A_ub.tocsr(),
        b_ub,
        A_eq.tocsr(),
        b_eq,
        bounds,
        len(v_int) + len(v_con),
        len(v_int),
        c0,
    )
