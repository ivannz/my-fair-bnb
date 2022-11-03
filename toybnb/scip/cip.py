import re
import numpy as np
from scipy import sparse as sp

from ..milp import MILP


# regex for parsing pieces of CIP format
# https://docs.python.org/3/library/re.html#simulating-scanf
re_name, re_type = r"<([^>]+)>", r"\[([^\]]+)\]"
re_float = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+|)(?:[eE][-+]?\d+)?)"
re_op, re_sign = r"([<>=]=)", r"([-+]?)"

pat_var = re.compile(
    rf"^{re_type}\s*{re_name}\s*:\s*obj\s*=\s*{re_float}\s*,"
    rf"\s*original\s*bounds\s*=\s*\[{re_float},\s*{re_float}\]$",
    re.I,
)

pat_con_head = re.compile(rf"^{re_type}\s*{re_name}\s*:\s*(.*)$")
pat_con_expr = re.compile(rf"^{re_sign}\s*{re_float}?\s*{re_name}\s*{re_type}\s*(.*)$")
pat_con_last = re.compile(rf"^{re_op}\s*{re_float}\s*;$")


def add_var(line: str, to: dict) -> str:
    """Parse a variable declaration line and add it to the dict."""
    m = pat_var.match(line)
    if m is None:
        return line

    vt, name, coef, lb, ub = m.groups()

    assert name not in to
    to[name] = vt, (float(lb), float(ub)), float(coef)
    return line


def add_con(line: str, to: dict) -> str:
    """Parse a constraint specification line and add it to the dict."""
    tail = line

    m = pat_con_head.match(line)
    if m is None:
        return line

    ct, name, tail = m.groups()
    assert ct.lower() == "linear"

    terms = []
    while tail:
        m = pat_con_expr.match(tail)
        if m is None:
            break

        sign, val, var, vt, tail = m.groups()
        coef = sign + ("1" if not val else val)
        terms.append((var, float(coef)))
    op, rhs = pat_con_last.match(tail).groups()

    assert name not in to
    to[name] = terms, op, float(rhs)
    return line


def parse_cip(cip: str) -> tuple[str, str, dict, dict]:
    """A very simple non-resilient CIP file reader."""
    with open(cip, "rt") as f:
        assert f.readline().strip() == "STATISTICS"
        _, name = re.split(r"\s*:\s*", f.readline().strip())
        f.readline()
        f.readline()

        assert f.readline().strip() == "OBJECTIVE"
        _, sense = re.split(r"\s*:\s*", f.readline().strip().lower())
        assert sense in ("maximize", "minimize")

        assert f.readline().strip() == "VARIABLES"
        m_vars = {}
        while add_var(f.readline().strip(), m_vars) != "CONSTRAINTS":
            pass

        m_cons = {}
        while add_con(f.readline().strip(), m_cons) != "END":
            pass
        assert not f.readline().strip()

    return name, sense[:3], m_vars, m_cons


def from_cip(filename: str) -> MILP:
    name, sense, m_vars, m_cons = parse_cip(filename)

    # split the vars in two groups
    v_int, v_con = [], []
    for v, (vt, _, _) in m_vars.items():
        which = v_int if vt in ("binary", "integer") else v_con
        which.append(v)

    # assign correct indices
    lut = {v: j for j, v in enumerate(v_int + v_con)}
    m, n = len(v_int), len(v_int) + len(v_con)

    # rebuild A_ub x <= b_ub
    b_ub = np.full(len(m_cons), np.nan, float)
    A_ub = sp.lil_array((len(m_cons), n), dtype=float)
    for j, (lhs, op, rhs) in enumerate(m_cons.values()):
        # flip the sign of geq consraints
        s = +1 if op == "<=" else -1

        b_ub[j] = rhs * s
        for v, coef in lhs:
            A_ub[j, lut[v]] = coef * s

    # reconstruct the cost vector and the bounds
    c = np.zeros(len(lut))
    bounds = np.full((len(lut), 2), (-np.inf, +np.inf))
    for v, (_, lu, coef) in m_vars.items():
        c[lut[v]] = coef
        bounds[lut[v]] = lu

    # flip the cost vector if the problem is maximization
    if sense == "max":
        c *= -1

    return MILP(
        c,
        A_ub.tocsr(),
        b_ub,
        np.empty((0, n), float),
        np.empty((0,), float),
        bounds,
        n,
        m,
    )
