import itertools
from fractions import Fraction
from wick.convenience import get_sym, normal_ordered
from wick.convenience import Idx, Tensor, FOperator, Sigma, Term, Expression

PYTHON_FILE_TAB = "    "

def space_idx_formatter(name, space_list):    
    s = ""
    for i, space in enumerate(space_list):
        if space == "nm":
            pass
        else:
            s += f"{space[0]}"

    if len(s) > 0:
        s = f"{name}." + s
    else:
        s = f"{name}" 

    return s

def einsum_str_formatter(sstr, istr, fstr, tstr):
    """Format the einsum string to the format that can be used in the code.
    If the tstr is longer than 3 terms, optimize will be set to True.
    """
    einsum_str = "\'" + istr[:-1] + "->" + fstr + "\'"

    opt = "optimize = False"
    istr_split = istr.split(",")[:-1]
    if len(istr_split) >= 3:
        opt = "optimize = True"
    elif len(istr_split) == 2:
        if len(istr_split[0]) == 4 and len(istr_split[1]) == 4:
            opt = "optimize = True"

    return f"{float(sstr): 12.6f} * einsum({einsum_str:20s}{tstr}, {opt})"