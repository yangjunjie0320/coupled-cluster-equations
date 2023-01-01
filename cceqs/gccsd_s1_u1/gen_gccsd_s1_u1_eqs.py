from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick

from wick.convenience import one_e, two_e
from wick.convenience import one_p, two_p, ep11
from wick.convenience import P1, E1, E2, EPS1
from wick.convenience import braE1, braE2
from wick.convenience import ketE1, ketE2
from wick.convenience import braP1, braP1E1
from wick.convenience import commute

PYTHON_FILE_TAB = "    "

def space_idx_formatter(name, space_list):    
    s = ""
    for i, space in enumerate(space_list):
        if space == "nm":
            pass
        else:
            s += f"{space[0]}"

    if len(s) > 0:
        if s == "ovvv":
            s = f"-{name}." + "vovv"
        elif s == "vvov":
            s = f"-{name}." + "vvvo"
        elif s == "ovvo":
            s = f"-{name}." + "vovo"
        elif s == "voov":
            s = f"-{name}." + "vovo"
        elif s == "ovoo":
            s = f"-{name}." + "vooo"
        elif s == "ooov":
            s = f"-{name}." + "oovo"
        elif s == "ovov":
            s = f"{name}." + "vovo"
        else:
            s = f"{name}." + s
    else:
        s = f"{name}" 

    return s

def einsum_str_formatter(sstr, fstr, istr, tstr):
    einsum_str = "\'" + istr + "->" + fstr + "\'"
    return f"{float(sstr): 12.6f} * einsum({einsum_str:20s}{tstr})"

def gen_einsum_fxn(final, name_str="get", return_str="res", arg_str_list=None, file_obj=None):
    if arg_str_list is None:
        arg_str_list = ["h1e", "h2e", "h1p", "h2p", "h1e1p", "t1e", "t2e", "t1p", "t1p1e"]

    arg_str = ""
    for iarg, arg in enumerate(arg_str_list):
        arg_str += f"{arg}, " if iarg != len(arg_str_list) - 1 else f"{arg}"

    function_str  = '''\nfrom numpy import einsum\n\n'''
    function_str += f"def {name_str}({arg_str}):\n"
    function_str += PYTHON_FILE_TAB + "res = 0.0\n"

    einsum_str = final._print_einsum(return_str, exprs_with_space=[H],
                                     space_idx_formatter=space_idx_formatter, einsum_str_formatter=einsum_str_formatter)
    einsum_str = einsum_str.split("\n")

    for i, line in enumerate(einsum_str):
        function_str += f"{PYTHON_FILE_TAB}{line}\n"

    function_str += f"{PYTHON_FILE_TAB}return res\n"

    file_obj.write(function_str)
    file_obj.write("\n\n")

H1e = one_e("h1e", ["occ", "vir"], norder=True)
H2e = two_e("h2e", ["occ", "vir"], norder=True)
Hph = one_p("h1p") + two_p("h2p")
Hep = ep11("h1e1p", ["occ", "vir"], ["nm"], norder=True)
H = H1e + H2e + Hph + Hep

T1  = E1("t1e", ["occ"], ["vir"])
T2  = E2("t2e", ["occ"], ["vir"])
S1  = P1("t1p", ["nm"])
U11 = EPS1("t1p1e", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + U11 

HT    = commute(H, T)
HTT   = commute(HT, T)
HTTT  = commute(HTT, T)
HTTTT = commute(HTTT, T)

with open("gccsd_s1_u1_eqs.py", "w") as f:
    print("Generating gccsd_s1_u1_eqs.py ...")
    print("- energy equation")
    S = (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="gccsd_s1_u1_ene", file_obj=f)

    print("- r1e equations")
    bra = braE1("occ", "vir")
    S = bra * (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="gccsd_s1_u1_r1e", file_obj=f)

    print("- r2e equations")
    bra = braE2("occ", "vir", "occ", "vir")
    S = bra * (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="gccsd_s1_u1_r2e", file_obj=f)

    print("- r1p equations")
    bra = braP1("nm")
    S = bra * (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="gccsd_s1_u1_r1p", file_obj=f)

    print("- r1p1e equations")
    bra = braP1E1("nm", "occ", "vir")
    S = bra * (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="gccsd_s1_u1_r1p1e", file_obj=f)