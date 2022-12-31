import numpy, scipy

from cqcpy import utils
from cqcpy.ov_blocks import two_e_blocks_full

from epcc.hh_model import HHModel
from epcc.epcc import epcc, ccsd_pt2, epccsd_2_s1, epccsd_2_s2

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

def hub_hol_epcc(nsite, nelec, hub_u, hol_g):
    nelec_alph = nelec//2
    nelec_beta = nelec//2

    u = 4
    w = 0.5

    model = HHModel(
        nsite, nsite, nelec, u, w, 
        g=hol_g, bc='p', gij=None
        )
    ehf = model.hf_energy()

    options = {
        "ethresh" : 1e-6,
        "tthresh" : 1e-6,
        "max_iter" : 500,
        "damp" : 0.3
        }

    ecc, e_corr, amps = epcc(model, options, ret=True)
    return ecc, e_corr, amps

def vec_size(no, nv):
    ss = no * nv
    sd = nv * (nv - 1) // 2 * no * (no - 1) // 2
    return ss + sd

def amp_to_vec(amp, no=None, nv=None, np=None):
    t1e, t2e, t1p, t1p1e = amp
    assert t1e.shape   == (nv, no)
    assert t2e.shape   == (nv, nv, no, no)
    assert t1p.shape   == (np, )
    assert t1p1e.shape == (np, nv, no)

    ns  = no * nv
    nd  = nv * (nv - 1) // 2 * no * (no - 1) // 2

    vec = numpy.zeros(ns + nd + np + np * ns)

    vec[:ns] = t1e.reshape(-1)

    v2e = numpy.zeros(nd)
    count = 0
    for a in range(nv):
        for i in range(no):
            for b in range(a+1, nv):
                for j in range(i+1, no):
                    v2e[count] = t2e[a, b, i, j]
                    count += 1
    vec[ns:(ns+nd)] = v2e

    vec[(ns+nd):(ns+nd+np)] = t1p
    vec[(ns+nd+np):] = t1p1e.reshape(-1)

    return vec

def vec_to_amp(vec, no=None, nv=None, np=None):
    ns  = no * nv
    nd  = nv * (nv - 1) // 2 * no * (no - 1) // 2

    assert vec.shape == (ns + nd +np + np * ns,)

    t1e = vec[:ns].reshape(nv, no)

    v2e = vec[ns:(ns+nd)]
    t2e = numpy.zeros((nv, nv, no, no))
    count = 0
    for a in range(nv):
        for i in range(no):
            for b in range(a+1, nv):
                for j in range(i+1, no):
                    t2e[a, b, i, j] =  v2e[count]
                    t2e[b, a, i, j] = -v2e[count]
                    t2e[a, b, j, i] = -v2e[count]
                    t2e[b, a, j, i] =  v2e[count]
                    count += 1

    t1p   = vec[(ns+nd):(ns+nd+np)].reshape(np,)
    t1p1e = vec[(ns+nd+np):].reshape(np, nv, no)

    amp   = (t1e, t2e, t1p, t1p1e)

    return amp

def hub_hol_epcc_krylov(nsite, nelec, hub_u, amps=None):
    # core guess
    nelec_alph = nelec//2
    nelec_beta = nelec//2
    u = 4
    w = 0.5

    model = HHModel(
        nsite, nsite, nelec, u, w, 
        g=hol_g, bc='p', gij=None
        )
    model.na = nelec_alph
    model.nb = nelec_beta
    ehf = model.hf_energy()

    f1e      = model.fock()
    f1e_alph = f1e[:nsite, :nsite]
    f1e_beta = f1e[nsite:, nsite:]

    ene_alph, coeff_alph = scipy.linalg.eigh(f1e_alph)
    ene_beta, coeff_beta = scipy.linalg.eigh(f1e_beta)

    # orbital energies, denominators, and Fock matrix in spin-orbital basis
    f1e    = model.g_fock()
    eo     = f1e.oo.diagonal()
    ev     = f1e.vv.diagonal()
    f1e.oo = f1e.oo
    f1e.vv = f1e.vv
    nocc   = eo.size
    nvir   = ev.size
    h1e    = f1e

    h2p = model.omega()
    h2p = numpy.diag(h2p)
    h1e1p, h1e1p_ = model.gint()
    h1p, h1p_     = model.mfG()
    nph = h1p.shape[0]

    # get ERIs
    h2e = g_aint_full(model)

    from gccsd_s1_u1_eqs import gccsd_s1_u1_ene
    from gccsd_s1_u1_eqs import gccsd_s1_u1_r1e, gccsd_s1_u1_r2e
    from gccsd_s1_u1_eqs import gccsd_s1_u1_r1p, gccsd_s1_u1_r1p1e

    def res(x):
        t1e, t2e, t1p, t1p1e = vec_to_amp(x, no=nocc, nv=nvir, np=nph)
        r1e   = gccsd_s1_u1_r1e(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e)
        r2e   = gccsd_s1_u1_r2e(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e)
        r1p   = gccsd_s1_u1_r1p(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e)
        r1p1e = gccsd_s1_u1_r1p1e(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e)
        res   = (r1e, r2e, r1p, r1p1e)
        return amp_to_vec(res, no=nocc, nv=nvir, np=nph)

    from scipy import optimize

    if amps is None:
        t1e = numpy.zeros((nocc, nvir))
        t2e = numpy.zeros((nocc, nocc, nvir, nvir))
        t1p = numpy.zeros((nph,))
        t1p1e = numpy.zeros((nph, nvir, nocc))

    else:
        t1e, t2e, t1p, t1p1e = amps

    x0 = amp_to_vec((t1e, t2e, t1p, t1p1e), no=nocc, nv=nvir, np=nph)

    from scipy import optimize
    sol = optimize.newton_krylov(res, x0, f_tol=1e-12, verbose=True)

    amp = vec_to_amp(sol, no=nocc, nv=nvir, np=nph)
    t1e, t2e, t1p, t1p1e = amp
    ene_cc = gccsd_s1_u1_ene(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e)

    return ehf + ene_cc, ene_cc, amp

if __name__ == "__main__":
    nsite = 4 
    nelec = 4
    hub_u = 1.0
    hol_g = 0.1
    etot_ref, ecc_ref, amp_ref = hub_hol_epcc(nsite, nelec, hub_u, hol_g)
    etot,     ecc,     amp     = hub_hol_epcc_krylov(nsite, nelec, hub_u, amps=None)

    t1e_ref, t2e_ref, t1p_ref, t1p1e_ref = amp_ref
    t1e, t2e, t1p, t1p1e = amp
    print("Error in energy = %6.4e" % abs(ecc_ref - ecc))
    print("Error in t1     = %6.4e" % numpy.linalg.norm(t1e_ref - t1e))
    print("Error in t2     = %6.4e" % numpy.linalg.norm(t2e_ref - t2e))
    print("Error in t1p    = %6.4e" % numpy.linalg.norm(t1p_ref - t1p))
    print("Error in t1p1e  = %6.4e" % numpy.linalg.norm(t1p1e_ref - t1p1e))