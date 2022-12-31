import numpy, scipy

from cqcpy import utils
from cqcpy.ov_blocks import two_e_blocks_full

from epcc.hubbard import Hubbard1D
from epcc.ccsd import ccsd
from epcc.ccsd import ccsd_gen

def g_aint_full(eris):
    nb = eris.nb
    na = eris.na
    C = utils.block_diag(eris.ca, eris.cb)
    U = eris.umat()
    Ua = U - U.transpose((0,1,3,2))
    Ua_mo = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',Ua,C,C,C,C)
    temp = [i for i in range(2*eris.L)]
    oidx = temp[:na] + temp[eris.L:eris.L + nb] 
    vidx = temp[na:eris.L] + temp[eris.L + nb:]

    eris_blocks = {
        "vvvv": Ua_mo[numpy.ix_(vidx,vidx,vidx,vidx)],
        "vvvo": Ua_mo[numpy.ix_(vidx,vidx,vidx,oidx)],
        "vvov": Ua_mo[numpy.ix_(vidx,vidx,oidx,vidx)],
        "vovv": Ua_mo[numpy.ix_(vidx,oidx,vidx,vidx)],
        "ovvv": Ua_mo[numpy.ix_(oidx,vidx,vidx,vidx)],
        "vvoo": Ua_mo[numpy.ix_(vidx,vidx,oidx,oidx)],
        "vovo": Ua_mo[numpy.ix_(vidx,oidx,vidx,oidx)],
        "voov": Ua_mo[numpy.ix_(vidx,oidx,oidx,vidx)],
        "ovvo": Ua_mo[numpy.ix_(oidx,vidx,vidx,oidx)],
        "ovov": Ua_mo[numpy.ix_(oidx,vidx,oidx,vidx)],
        "oovv": Ua_mo[numpy.ix_(oidx,oidx,vidx,vidx)],
        "ooov": Ua_mo[numpy.ix_(oidx,oidx,oidx,vidx)],
        "oovo": Ua_mo[numpy.ix_(oidx,oidx,vidx,oidx)],
        "ovoo": Ua_mo[numpy.ix_(oidx,vidx,oidx,oidx)],
        "vooo": Ua_mo[numpy.ix_(vidx,oidx,oidx,oidx)],
        "oooo": Ua_mo[numpy.ix_(oidx,oidx,oidx,oidx)]
    }

    return two_e_blocks_full(**eris_blocks)

def hub_gccsd(nsite, nelec, hub_u):
    # core guess
    nelec_alph = nelec//2
    nelec_beta = nelec//2
    one_s2 = 1.0/numpy.sqrt(2.0)

    coeff_alph = numpy.zeros((nsite, nsite))
    coeff_beta = coeff_alph.copy()

    model = Hubbard1D(nsite, nelec, hub_u, ca=coeff_alph, cb=coeff_beta, bc='p')
    f1e      = model.fock()
    f1e_alph = f1e[:nsite, :nsite]
    f1e_beta = f1e[nsite:, nsite:]

    ene_alph, coeff_alph = scipy.linalg.eigh(f1e_alph)
    ene_beta, coeff_beta = scipy.linalg.eigh(f1e_beta)

    model = Hubbard1D(nsite, nelec, hub_u, ca=coeff_alph, cb=coeff_beta, bc='p')

    options = { 
            "ethresh" : 1e-11,
            "tthresh" : 1e-9,
            "max_iter" : 100,
            "damp" : 0.0
            }

    etot, ecc, t1, t2 = ccsd(model,options)
    return etot, ecc, t1, t2

def vec_size(no, nv):
    ss = no * nv
    sd = nv * (nv - 1) // 2 * no * (no - 1) // 2
    return ss + sd

def amp_to_vec(amp, no=None, nv=None):
    t1, t2 = amp
    assert t1.shape == (nv, no)
    assert t2.shape == (nv, nv, no, no)

    ns  = no * nv
    nd  = nv * (nv - 1) // 2 * no * (no - 1) // 2

    vec = numpy.zeros(ns + nd)
    vec[:ns] = t1.reshape(-1)

    vd = numpy.zeros(nd)
    count = 0
    for a in range(nv):
        for i in range(no):
            for b in range(a+1, nv):
                for j in range(i+1, no):
                    vd[count] = t2[a, b, i, j]
                    count += 1
    vec[ns:] = vd
    return vec

def vec_to_amp(vec, no=None, nv=None):
    ns  = no * nv
    nd  = nv * (nv - 1) // 2 * no * (no - 1) // 2

    t1 = vec[:ns].reshape(nv, no)
    vd = vec[ns:]

    t2 = numpy.zeros((nv, nv, no, no))
    count = 0
    for a in range(nv):
        for i in range(no):
            for b in range(a+1, nv):
                for j in range(i+1, no):
                    t2[a, b, i, j] = vd[count]
                    t2[b, a, i, j] = -vd[count]
                    t2[a, b, j, i] = -vd[count]
                    t2[b, a, j, i] = vd[count]
                    count += 1
    return t1, t2

def hub_gccsd_krylov(nsite, nelec, hub_u, t1=None, t2=None, l1=None, l2=None):
    # core guess
    nelec_alph = nelec//2
    nelec_beta = nelec//2
    one_s2 = 1.0/numpy.sqrt(2.0)

    coeff_alph = numpy.zeros((nsite, nsite))
    coeff_beta = coeff_alph.copy()

    model = Hubbard1D(nsite, nelec, hub_u, ca=coeff_alph, cb=coeff_beta, bc='p')
    f1e      = model.fock()
    f1e_alph = f1e[:nsite, :nsite]
    f1e_beta = f1e[nsite:, nsite:]

    ene_alph, coeff_alph = scipy.linalg.eigh(f1e_alph)
    ene_beta, coeff_beta = scipy.linalg.eigh(f1e_beta)

    model = Hubbard1D(nsite, nelec, hub_u, ca=coeff_alph, cb=coeff_beta, bc='p')

    options = { 
            "ethresh" : 1e-11,
            "tthresh" : 1e-9,
            "max_iter" : 100,
            "damp" : 0.0
            }

    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    f1e    = model.g_fock()
    eo     = f1e.oo.diagonal()
    ev     = f1e.vv.diagonal()
    f1e.oo = f1e.oo
    f1e.vv = f1e.vv
    nocc = eo.size
    nvir = ev.size

    # get ERIs
    eris = g_aint_full(model)

    from ccsd_eqs import ccsd_ene
    from ccsd_eqs import ccsd_res_s
    from ccsd_eqs import ccsd_res_d
    from ccsd_eqs import ccsd_lambda_lhs_s, ccsd_lambda_rhs_s
    from ccsd_eqs import ccsd_lambda_lhs_d, ccsd_lambda_rhs_d

    def res(x):
        t1, t2 = vec_to_amp(x, no=nocc, nv=nvir)
        res_s = ccsd_res_s(f1e, eris, t1, t2)
        res_d = ccsd_res_d(f1e, eris, t1, t2)
        return amp_to_vec((res_s, res_d), no=nocc, nv=nvir)

    from scipy import optimize
    if t1 is None:
        t1 = numpy.zeros((nvir, nocc))
    if t2 is None:
        t2 = numpy.zeros((nvir, nvir, nocc, nocc))
    x0 = amp_to_vec((t1, t2), no=nocc, nv=nvir)

    from scipy import optimize
    sol = optimize.newton_krylov(res, x0, f_tol=1e-10)

    t1, t2 = vec_to_amp(sol, no=nocc, nv=nvir)
    ene_cc = ccsd_ene(f1e, eris, t1, t2)

    def res(x):
        l1t, l2t = vec_to_amp(x, no=nocc, nv=nvir)
        l1 = l1t.T
        l2 = l2t.transpose(2, 3, 0, 1)
        res_s = ccsd_lambda_lhs_s(f1e, eris, t1, t2, l1, l2) - ccsd_lambda_rhs_s(f1e, eris, t1, t2)
        res_d = ccsd_lambda_lhs_d(f1e, eris, t1, t2, l1, l2) - ccsd_lambda_rhs_d(f1e, eris, t1, t2)
        res_st = res_s.T
        res_dt = res_d.transpose(2, 3, 0, 1)
        return amp_to_vec((res_st, res_dt), no=nocc, nv=nvir)

    if l1 is None:
        l1 = numpy.zeros((nocc, nvir))

    if l2 is None:
        l2 = numpy.zeros((nocc, nocc, nvir, nvir))

    x0 = amp_to_vec((l1.T, l2.T), no=nocc, nv=nvir)
    sol = optimize.newton_krylov(res, x0, f_tol=1e-10)
    l1t, l2t = vec_to_amp(sol, no=nocc, nv=nvir)
    l1 = l1t.T
    l2 = l2t.transpose(2, 3, 0, 1)

    return ene_cc, t1, t2

if __name__ == "__main__":
    nsite = 6 
    nelec = 2
    hub_u = 2.0 
    etot_ref, ecc_ref, t1_ref, t2_ref = hub_gccsd(nsite, nelec, hub_u)
    ecc, t1, t2 = hub_gccsd_krylov(nsite, nelec, hub_u, None, None)
    print("Error in energy = %6.4e" % abs(ecc_ref - ecc))
    print("Error in t1     = %6.4e" % numpy.linalg.norm(t1_ref - t1))
    print("Error in t2     = %6.4e" % numpy.linalg.norm(t2_ref - t2))