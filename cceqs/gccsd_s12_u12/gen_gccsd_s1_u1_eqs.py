from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick

from wick.convenience import one_e, two_e
from wick.convenience import one_p, two_p, ep11
from wick.convenience import P1, E1, E2, EPS1
from wick.convenience import P2, EPS2
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
        arg_str_list = ["h1e", "h2e", "h1p", "h2p", "h1e1p", "t1e", "t2e", "t1p", "t2p", "t1p1e", "t2p1e"]

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
S2  = P2("t1p", ["nm"])
U11 = EPS1("t1p1e", ["nm"], ["occ"], ["vir"])
U21 = EPS2("t2p1e", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + S2 + U11 + U21 

HT    = commute(H, T)
HTT   = commute(HT, T)
HTTT  = commute(HTT, T)
HTTTT = commute(HTTT, T)

def gen_gccsd_eom_ip_eqs():
    from fractions import Fraction
    from wick.convenience import one_e, two_e
    from wick.convenience import E1, E2
    from wick.convenience import Eip1, Eip2
    from wick.convenience import braE1, braE2
    from wick.convenience import braEip1, braEip2

    T1 = E1("t1e", ["occ"], ["vir"])
    T2 = E2("t2e", ["occ"], ["vir"])
    T  = T1 + T2

    R1 = Eip1("r1e", ["occ"])
    R2 = Eip2("r2e", ["occ"], ["vir"])
    R  = R1 + R2

    HT1 = commute(H,   T)
    HT2 = commute(HT1, T)
    HT3 = commute(HT2, T)
    HT4 = commute(HT3, T)

    hbar  = H + HT1 
    hbar += Fraction('1/2')  * HT2
    hbar += Fraction('1/6')  * HT3
    hbar += Fraction('1/24') * HT4

    expr    = H + HT1 + Fraction('1/2') * HT2
    ene_cor = apply_wick(expr)
    ene_cor.resolve()

    def gen_einsum_fxn(final, name_str="get", return_str="res", arg_str_list=None, file_obj=None):
        import cceqs
        from cceqs.utils import space_idx_formatter
        from cceqs.utils import einsum_str_formatter
        from cceqs.utils import PYTHON_FILE_TAB

        if arg_str_list is None:
            arg_str_list = ["h1e", "h2e", "t1e", "t2e", "r1e", "r2e"]

        arg_str = ""
        for iarg, arg in enumerate(arg_str_list):
            arg_str += f"{arg}, " if iarg != len(arg_str_list) - 1 else f"{arg}"

        function_str  = '''\nfrom numpy import einsum\n\n'''
        function_str += f"def {name_str}({arg_str}):\n"
        function_str += PYTHON_FILE_TAB + "%s = 0.0\n" % return_str

        einsum_str = final._print_einsum(
            return_str, exprs_with_space=[H],
            space_idx_formatter=space_idx_formatter, 
            einsum_str_formatter=einsum_str_formatter
            )
        einsum_str = einsum_str.split("\n")

        for i, line in enumerate(einsum_str):
            function_str += f"{PYTHON_FILE_TAB}{line}\n"

        function_str += f"{PYTHON_FILE_TAB}return res\n"

        file_obj.write(function_str)
        file_obj.write("\n\n")

    with open("_gccsd_eom_ip_eqs.py", "w") as f:
        print("Generating _gccsd_eom_ip_eqs.py ...")

        print("- eom_ip_h1e equations")
        bra = braEip1("occ")
        expr = bra * (hbar - ene_cor) * R
        out = apply_wick(expr)
        out.resolve()
        final = AExpression(Ex=out)
        gen_einsum_fxn(final, name_str="gccsd_eom_ip_h1e", file_obj=f)

        print("- eom_ip_h2e equations")
        bra = braEip2("occ", "occ", "vir")
        expr = bra * (hbar - ene_cor) * R
        out = apply_wick(expr)
        out.resolve()
        final = AExpression(Ex=out)
        gen_einsum_fxn(final, name_str="gccsd_eom_ip_h2e", file_obj=f)

