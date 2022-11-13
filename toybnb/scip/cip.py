import re
import numpy as np
from scipy import sparse as sp

from collections import defaultdict
from ..milp import MILP


# regex for parsing pieces of CIP format
# https://docs.python.org/3/library/re.html#simulating-scanf
re_type = r"\[([^\]]*)\]"
re_name = r"<([^>]*)>"
re_float = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+|)(?:[eE][-+]?\d+)?"

pat_head = re.compile(rf"^\s*{re_type}\s*{re_name}\s*:\s*(.*)$", re.I)
pat_var_obj = re.compile(rf"^obj=({re_float})\s*,?(.*)", re.I)
pat_bounds = re.compile(
    rf"^\s*(\S*)[^=]*=\s*\[+({re_float}),\s*({re_float})\]+,*(.*)$",
    re.I,
)
pat_term = re.compile(
    rf"^\s*(([-+]?)\s*({re_float})\s*(?:{re_name}(?:{re_type})?)?)\s*(.*)$",
    re.I,
)
pat_rhs = re.compile(
    rf"^\s*([<>=]=)\s*({re_float})\s*;\s*(.*)$",
    re.I,
)


def parse_expr(tail: str) -> tuple[dict, str]:
    """Parse a linea expression with optional constaint term."""
    # terms have a const by default
    terms = defaultdict(float, [(None, 0.0)])

    m = pat_term.match(tail)
    while m:
        # standalone floats are constant terms
        term, sign, val, var, _, tail = m.groups()
        if not term:
            break

        # standalone signs in front of variables have implicit one
        if var is not None:
            val = val if val else "1"
        terms[var] += float(sign + val)
        m = pat_term.match(tail)

    return terms, tail


def parse_var(tail: str) -> tuple[str, tuple, str]:
    """Parse a variable declaration line."""
    # get the type, variable name and its coefficient
    type, var, tail = pat_head.match(tail).groups()
    obj, tail = pat_var_obj.match(tail).groups()
    if type not in ("integer", "binary", "continuous"):
        raise NotImplementedError(type)

    # fetch the bounds of the variable
    lb, ub = float("-inf"), float("+inf")
    m = pat_bounds.match(tail)
    while m:
        bound, b_lb, b_ub, tail = m.groups()
        if bound not in ("original", "local", "global"):
            raise NotImplementedError

        lb = max(lb, float(b_lb))
        ub = min(ub, float(b_ub))

        m = pat_bounds.match(tail)

    return var, (type, float(obj), lb, ub), tail


def parse_con(tail: str) -> tuple[str, tuple, str]:
    """Parse a constraint specification line."""
    # get the type and name of the constraint
    type, con, tail = pat_head.match(tail).groups()
    if type not in ("linear"):
        raise NotImplementedError(type)

    # get the terms and then the right-hand side
    terms, tail = parse_expr(tail)
    assert tail

    # subtract the rhs from the constant term
    op, rhs, tail = pat_rhs.match(tail).groups()
    terms[None] -= float(rhs)  # xxx substract rhs fomr lhs's const

    # our constraints hanve implicit zero right-hand sides
    return con, (type, op, dict(terms)), tail


def parse_fix(tail: str) -> tuple[str, tuple, str]:
    """Parse a fixed variable specification."""
    # read the var part
    name, var, tail = parse_var(tail)

    # handle the transformation
    type, sep, tail = tail.lstrip().partition(":")

    # type can also be `negated`
    if type not in ("fixed", "aggregated") or not sep:
        raise NotImplementedError(type)

    # parse the transformation
    terms, tail = parse_expr(tail)
    assert not tail

    # fixed variables are both a constraint and a variable
    terms[name] -= 1
    return name, (var, ("linear", "==", dict(terms))), tail


def parse_objective(sense: str, offset: float = 0.0, **ignore) -> tuple[str, float]:
    """Extract the objective sense and the offset."""
    assert sense in ("maximize", "minimize")
    return sense[:3], float(offset)


def cip_sections(filename: str) -> tuple[str, str]:
    """Read CIP file line-by-line with section detection."""
    sequence = {
        "STATISTICS": {"OBJECTIVE"},
        "OBJECTIVE": {"VARIABLES", "FIXED", "CONSTRAINTS", "END"},
        "VARIABLES": {"FIXED", "CONSTRAINTS", "END"},
        "FIXED": {"CONSTRAINTS", "END"},
        "CONSTRAINTS": {"END"},
    }

    with open(filename, "rt") as f:
        # get the name and create a new group
        last, line = "", f.readline().strip()
        assert line == "STATISTICS"
        while line != "END":
            group, sections = line, sequence[line]

            # items in the group start with whitespace
            last, line = line, f.readline().strip()
            while line and line not in sections:
                yield group, line
                last, line = line, f.readline().strip()

    return last


def parse_cip(filename: str) -> tuple[str, str, dict, dict]:
    """A very simple non-resilient CIP file reader."""
    m_cons, m_vars, m_other = {}, {}, {}
    for sec, line in cip_sections(filename):
        if sec == "VARIABLES":
            name, var, line = parse_var(line)
            m_vars[name] = var

        elif sec == "CONSTRAINTS":
            name, con, line = parse_con(line)
            m_cons[name] = con

        elif sec == "FIXED":
            name, (var, con), line = parse_fix(line)
            assert name not in m_vars
            m_vars[name] = var

            assert name not in m_cons
            m_cons["f_c_" + name] = con

        else:
            head, tail = re.split(r"\s*:\s*", line)
            m_other.setdefault(sec, {}).update([(head.lower(), tail)])
            continue

        assert not line

    name = m_other["STATISTICS"].get("problem name")
    objective = parse_objective(**m_other["OBJECTIVE"])

    return name, objective, m_vars, m_cons


def from_cip(filename: str) -> MILP:
    # we don't care about the offset
    name, (sense, c0), m_vars, m_cons = parse_cip(filename)

    # split the vars in two groups: integer and continuous
    v_int, v_con = [], []
    for v, (vt, *_) in m_vars.items():
        which = v_int if vt in ("binary", "integer") else v_con
        which.append(v)

    # assign correct indices
    lut = {v: j for j, v in enumerate(v_int + v_con)}
    m, n = len(v_int), len(v_int) + len(v_con)

    # normalize the constraints and split them into eq and ub
    m_cons_eq, m_cons_ub = [], []
    for _, op, terms in m_cons.values():
        assert op in ("<=", "==", ">=")
        # the parser returns constraint specs with implicit rhs zero
        #  and explicit lhs const term, get it and convert into
        #  `lhs | op | rhs` form. Also flip constraints with `>=` op
        sign = +1.0
        if op == ">=":
            sign, op = -1.0, "<="

        lhs = {k: sign * v for k, v in terms.items() if k is not None}
        rhs = -terms[None] * sign

        which = m_cons_eq if op == "==" else m_cons_ub
        which.append((lhs, rhs))

    # rebuild A_ub x <= b_ub
    b_ub = np.full(len(m_cons_ub), np.nan, float)
    A_ub = sp.lil_array((len(m_cons_ub), n), dtype=float)
    for j, (lhs, rhs) in enumerate(m_cons_ub):
        b_ub[j] = rhs
        for v, coef in lhs.items():
            A_ub[j, lut[v]] = coef

    # rebuild A_eq x == b_eq
    b_eq = np.full(len(m_cons_eq), np.nan, float)
    A_eq = sp.lil_array((len(m_cons_eq), n), dtype=float)
    for j, (lhs, rhs) in enumerate(m_cons_eq):
        b_eq[j] = rhs
        for v, coef in lhs.items():
            A_eq[j, lut[v]] = coef

    # reconstruct the cost vector and the bounds
    c = np.zeros(len(lut))
    bounds = np.full((len(lut), 2), (-np.inf, +np.inf))
    for v, (_, obj, lb, ub) in m_vars.items():
        c[lut[v]] = obj
        bounds[lut[v]] = lb, ub

    # flip the cost vector if the problem is maximization
    if sense == "max":
        c *= -1
        c0 = -c0

    return MILP(c, A_ub.tocsr(), b_ub, A_eq.tocsr(), b_eq, bounds, n, m, c0)
