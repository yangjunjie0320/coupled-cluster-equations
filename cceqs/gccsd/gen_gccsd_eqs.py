import wick
from wick.wick import apply_wick
from wick.expression import AExpression
from wick.convenience import commute

from fractions import Fraction
from wick.convenience import one_e, two_e
from wick.convenience import E1, E2
from wick.convenience import braE1, braE2
from wick.convenience import ketE1, ketE2
from wick.convenience import PE1

H1 = one_e("h1e", ["occ", "vir"], norder=True)
H2 = two_e("h2e", ["occ", "vir"], norder=True, anti=True, compress=True)
H  = H1 + H2

def gen_gccsd_amp_eqs():
    T1 = E1("t1e", ["occ"], ["vir"])
    T2 = E2("t2e", ["occ"], ["vir"])
    T  = T1 + T2

    HT1 = commute(H,   T)
    HT2 = commute(HT1, T)
    HT3 = commute(HT2, T)
    HT4 = commute(HT3, T)

    hbar  = H + HT1 
    hbar += Fraction('1/2')  * HT2 
    hbar += Fraction('1/6')  * HT3
    hbar += Fraction('1/24') * HT4

    def gen_einsum_fxn(final, name_str="get", return_str="res", arg_str_list=None, file_obj=None):
        import cceqs
        from cceqs.utils import space_idx_formatter
        from cceqs.utils import einsum_str_formatter
        from cceqs.utils import PYTHON_FILE_TAB

        if arg_str_list is None:
            arg_str_list = ["h1e", "h2e", "t1e", "t2e"]

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

    with open("_gccsd_amp_eqs.py", "w") as f:
        print("Generating _gccsd_amp_eqs.py ...")
        print("- energy equation")
        out = apply_wick(hbar)
        out.resolve()
        final = AExpression(Ex=out)
        gen_einsum_fxn(final, name_str="gccsd_ene", file_obj=f)

        print("- r1e equations")
        bra   = braE1("occ", "vir")
        expr  = bra * hbar
        out = apply_wick(expr)
        out.resolve()
        final = AExpression(Ex=out)
        gen_einsum_fxn(final, name_str="gccsd_r1e", file_obj=f)

        print("- r2e equations")
        bra   = braE2("occ", "vir", "occ", "vir")
        expr  = bra * hbar
        out = apply_wick(expr)
        out.resolve()
        final = AExpression(Ex=out)
        gen_einsum_fxn(final, name_str="gccsd_r2e", file_obj=f)

def gen_gccsd_lam_eqs():
    from fractions import Fraction
    from wick.convenience import one_e, two_e
    from wick.convenience import E1, E2
    from wick.convenience import braE1, braE2

    T1 = E1("t1e", ["occ"], ["vir"])
    T2 = E2("t2e", ["occ"], ["vir"])
    T  = T1 + T2

    L1 = E1("l1e", ["vir"], ["occ"])
    L2 = E2("l2e", ["vir"], ["occ"])
    L  = L1 + L2

    HT1 = commute(H,   T)
    HT2 = commute(HT1, T)
    HT3 = commute(HT2, T)
    HT4 = commute(HT3, T)

    hbar  = H + HT1 
    hbar += Fraction('1/2')  * HT2 
    hbar += Fraction('1/6')  * HT3
    hbar += Fraction('1/24') * HT4

    def gen_einsum_fxn(final, name_str="get", return_str="res", arg_str_list=None, file_obj=None):
        import cceqs
        from cceqs.utils import space_idx_formatter
        from cceqs.utils import einsum_str_formatter
        from cceqs.utils import PYTHON_FILE_TAB

        if arg_str_list is None:
            arg_str_list = ["h1e", "h2e", "t1e", "t2e", "l1e", "l2e"]

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

    with open("_gccsd_lam_eqs.py", "w") as f:
        print("Generating _gccsd_lam_eqs.py ...")

        print("- rhs1e equations")
        ket   = ketE1("occ", "vir")
        expr  = hbar * ket
        out = apply_wick(expr)
        out.resolve()
        final = AExpression(Ex=out)
        gen_einsum_fxn(final, name_str="gccsd_lam_rhs1e", file_obj=f)

        print("- lhs1e equations")
        expr1 = L * expr
        out1  = apply_wick(expr1)
        out1.resolve()
        final1 = AExpression(Ex=out1)
        final1 = final1.get_connected()
        final1.sort_tensors()

        expr2 = L * ket * hbar
        out2  = apply_wick(expr2)
        out2.resolve()
        final2 = AExpression(Ex=out2)
        final2 = final2.get_connected()
        final2.sort_tensors()

        gen_einsum_fxn(final1 - final2, name_str="gccsd_lam_lhs1e", file_obj=f)

        print("- rhs2e equations")
        ket   = ketE2("occ", "vir", "occ", "vir")
        expr  = hbar * ket
        out = apply_wick(expr)
        out.resolve()
        final = AExpression(Ex=out)
        final = final.get_connected()
        final.sort_tensors()
        gen_einsum_fxn(final, name_str="gccsd_lam_rhs2e", file_obj=f)

        print("- lhs2e equations")
        expr1 = L * expr
        out1  = apply_wick(expr1)
        out1.resolve()
        final1 = AExpression(Ex=out1)
        final1 = final1.get_connected()
        final1.sort_tensors()

        p1 = PE1("occ", "vir")
        expr2 = (H + HT1) * p1 * L * ket
        out2  = apply_wick(expr2)
        out2.resolve()
        final2 = AExpression(Ex=out2)
        final2.sort_tensors()

        gen_einsum_fxn(final1 + final2, name_str="gccsd_lam_lhs2e", file_obj=f)

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

def gen_gccsd_eom_ea_eqs():
    from fractions import Fraction
    from wick.convenience import one_e, two_e
    from wick.convenience import E1, E2
    from wick.convenience import Eea1, Eea2
    from wick.convenience import braE1, braE2
    from wick.convenience import braEea1, braEea2

    T1 = E1("t1e", ["occ"], ["vir"])
    T2 = E2("t2e", ["occ"], ["vir"])
    T  = T1 + T2

    R1 = Eea1("r1e", ["vir"])
    R2 = Eea2("r2e", ["occ"], ["vir"])
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

    with open("_gccsd_eom_ea_eqs.py", "w") as f:
        print("Generating _gccsd_eom_ea_eqs.py ...")

        print("- eom_ea_h1e equations")
        bra = braEea1("vir")
        expr = bra * (hbar - ene_cor) * R
        out = apply_wick(expr)
        out.resolve()
        final = AExpression(Ex=out)
        gen_einsum_fxn(final, name_str="gccsd_eom_ea_h1e", file_obj=f)

        print("- eom_ea_h2e equations")
        bra = braEea2("occ", "vir", "vir")
        expr = bra * (hbar - ene_cor) * R
        out = apply_wick(expr)
        out.resolve()
        final = AExpression(Ex=out)
        gen_einsum_fxn(final, name_str="gccsd_eom_ea_h2e", file_obj=f)

if __name__ == "__main__":
    # gen_gccsd_amp_eqs()
    # gen_gccsd_lam_eqs()
    # gen_gccsd_eom_ip_eqs()
    gen_gccsd_eom_ea_eqs()