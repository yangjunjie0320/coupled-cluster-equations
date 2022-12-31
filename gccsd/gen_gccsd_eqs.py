from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e
from wick.convenience import E1, E2
from wick.convenience import braE1, braE2
from wick.convenience import ketE1, ketE2
from wick.convenience import commute

PYTHON_FILE_TAB = "    "

def space_idx_formatter(name, space_list):
    s = f"{name}."

    for i, space in enumerate(space_list):
        s += f"{space[0]}"

    return s

def einsum_str_formatter(sstr, fstr, istr, tstr):
    einsum_str = "\'" + istr + "->" + fstr + "\'"
    return f"{float(sstr): 12.6f} * einsum({einsum_str:20s}{tstr})"

def gen_einsum_fxn(final, name_str="get", return_str="res", arg_str_list=None, file_obj=None):
    if arg_str_list is None:
        arg_str_list = ["f1e", "eris", "t1", "t2"]

    arg_str = ""
    for iarg, arg in enumerate(arg_str_list):
        arg_str += f"{arg}, " if iarg != len(arg_str_list) - 1 else f"{arg}"

    function_str  = '''\nfrom pyscf import lib\n'''
    function_str += '''einsum = lib.einsum\n\n'''
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

H1 = one_e("f1e",  ["occ", "vir"], norder=True)
H2 = two_e("eris", ["occ", "vir"], norder=True)
H = H1 + H2

T1 = E1("t1", ["occ"], ["vir"])
T2 = E2("t2", ["occ"], ["vir"])
L1 = E1("l1", ["vir"], ["occ"])
L2 = E2("l2", ["vir"], ["occ"])
T = T1 + T2
L = L1 + L2

HT    = commute(H, T)
HTT   = commute(HT, T)
HTTT  = commute(commute(commute(H2, T1), T1), T1)
HTTTT = commute(HTTT, T)

with open("ccsd_eqs.py", "w") as f:
    S = (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="ccsd_ene", file_obj=f)

    bra = braE1("occ", "vir")
    S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="ccsd_res_s", file_obj=f)

    bra = braE2("occ", "vir", "occ", "vir")
    S = bra*(H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT)
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="ccsd_res_d", file_obj=f)

    ket = ketE1("occ", "vir")
    S = (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT) * ket
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="ccsd_lambda_rhs_s", file_obj=f)

    S1 = L * (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT) * ket
    out1 = apply_wick(S1)
    out1.resolve()
    ex1 = AExpression(Ex=out1)
    ex1 = ex1.get_connected()
    ex1.sort_tensors()

    S2 = L * ket * (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT)
    out2 = apply_wick(S2)
    out2.resolve()
    ex2 = AExpression(Ex=out2)
    ex2 = ex2.get_connected()
    ex2.sort_tensors()
    gen_einsum_fxn(ex1 - ex2, name_str="ccsd_lambda_lhs_s", file_obj=f)

    ket = ketE2("occ", "vir", "occ", "vir")
    S = (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT) * ket
    out = apply_wick(S)
    out.resolve()
    final = AExpression(Ex=out)
    gen_einsum_fxn(final, name_str="ccsd_lambda_rhs_d", file_obj=f)

    S1 = L * (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT) * ket
    out1 = apply_wick(S1)
    out1.resolve()
    ex1 = AExpression(Ex=out1)
    ex1 = ex1.get_connected()
    ex1.sort_tensors()

    S2 = L * ket * (H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT)
    out2 = apply_wick(S2)
    out2.resolve()
    ex2 = AExpression(Ex=out2)
    ex2 = ex2.get_connected()
    ex2.sort_tensors()
    gen_einsum_fxn(ex1 - ex2, name_str="ccsd_lambda_lhs_d", file_obj=f)


    
