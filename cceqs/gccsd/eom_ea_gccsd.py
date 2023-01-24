import sys
import numpy, sceay

from pyscf import cc, lib
from pyscf.lib import logger
from pyscf.cc.eom_gccsd import vector_to_amplitudes_ea as vector_to_amplitudes_ea_ov
from pyscf.cc.eom_gccsd import amplitudes_to_vector_ea as amplitudes_to_vector_ea_ov

def vec_to_amp_ea_ov(no, nv, vec=None):
    r1e_ea_ov, r2e_ea_ov = vector_to_amplitudes_ea_ov(vec, no+nv, no)
    assert r1e_ea_ov.shape == (nv,)
    assert r2e_ea_ov.shape == (no, nv, nv,)
    return r1e_ea_ov, r2e_ea_ov

def amp_to_vec_ea_ov(no, nv, amp=None):
    r1e_ea_ov, r2e_ea_ov = amp
    assert r1e_ea_ov.shape == (nv,)
    assert r2e_ea_ov.shape == (no, nv, nv,)
    vec = amplitudes_to_vector_ea_ov(r1e_ea_ov, r2e_ea_ov)
    return vec

def vec_to_amp_ea_vo(no, nv, vec=None):
    r1e_ea_ov, r2e_ea_ov = vector_to_amplitudes_ea_ov(vec, no+nv, no)
    assert r1e_ea_ov.shape == (nv,)
    assert r2e_ea_ov.shape == (nv, nv, no,)

    r1e_ea_vo, r2e_ea_vo = _transpose_ov_to_vo(no, nv, amp=(r1e_ea_ov, r2e_ea_ov))
    assert r1e_ea_vo.shape == (nv,)
    assert r2e_ea_vo.shape == (nv, nv, no,)
    return r1e_ea_vo, r2e_ea_vo

def amp_to_vec_ea_vo(no, nv, amp=None):
    r1e_ea_ov, r2e_ea_ov = _transpose_vo_to_ov(no, nv, amp=amp)
    vec = amplitudes_to_vector_ea_ov(r1e_ea_ov, r2e_ea_ov)
    return vec

def _transpose_ov_to_vo(no, nv, amp=None):
    r1e_ov, r2e_ov = amp
    assert r1e_ov.shape == (no,)
    assert r2e_ov.shape == (no, no, nv,)

    r1e_vo = r1e_ov
    r2e_vo = - r2e_ov.transpose(2, 0, 1)
    return r1e_vo, r2e_vo

def _transpose_vo_to_ov(no, nv, amp=None):
    r1e_vo, r2e_vo = amp
    assert r1e_vo.shape == (no,)
    assert r2e_vo.shape == (nv, no, no,)

    r1e_ov = r1e_vo
    r2e_ov = - r2e_vo.transpose(1, 2, 0)
    return r1e_ov, r2e_ov

def eom_ea_gccsd_diag(h1e, h2e, amp=None):
    from cceqs.gccsd import gccsd
    from pyscf.cc import gintermediates as imd
    nocc, nvir = h1e.ov.shape

    t1e_vo, t2e_vo = amp
    t1e_ov, t2e_ov = gccsd._transpose_vo_to_ov(nocc, nvir, amp)

    assert isinstance(h2e, cc.gccsd._PhysicistsERIs)
    
    foo = imd.Foo(t1e_ov, t2e_ov, h2e)
    fvv = imd.Fvv(t1e_ov, t2e_ov, h2e)
    fov = imd.Fov(t1e_ov, t2e_ov, h2e)

    w_oooo = imd.Woooo(t1e_ov, t2e_ov, h2e)
    w_ovvo = imd.Wovvo(t1e_ov, t2e_ov, h2e)
    w_oovv = h2e.oovv
    w_vvvv = h2e.vvvv

    d1e_ov = numpy.diag(fvv)
    d2e_ov = numpy.zeros((nocc, nvir, nvir))

    for a in range(nvir):
        _w_vvvv_a = numpy.array(w_vvvv[a])
        for b in range(a):
            for j in range(nocc):
                d2e_ov[j, a, b] += fvv[a, a]
                d2e_ov[j, a, b] += fvv[b, b]
                d2e_ov[j, a, b] -= foo[j, j]
                d2e_ov[j, a, b] += w_ovvo[j, b, b, j]
                d2e_ov[j, a, b] += w_ovvo[j, a, a, j]
                d2e_ov[j, a, b] += _w_vvvv_a[b, a, b] * 0.5
                d2e_ov[j, a, b] -= _w_vvvv_a[b, b, a] * 0.5
                d2e_ov[j, a, b] -= numpy.dot(w_oovv[:, j, a, b], t2e_ov[:, j, a, b]) * 0.5
                d2e_ov[j, a, b] += numpy.dot(w_oovv[:, j, b, a], t2e_ov[:, j, a, b]) * 0.5

    diag_ea = amp_to_vec_ea_ov(nocc, nvir, (d1e_ov, d2e_ov))
    return diag_ea

def solve_eom_ea_gccsd(h1e, h2e, amp=None, rhvs=None, max_cycle=50, tol=1e-8, 
                       nroots=5, max_space=20, verbose=3):
    '''Solve the ea EOM-CCSD equations.

    Args:
        h1e:
            1-electron integrals
        h2e: 
            2-electron integrals
        amp: 
            Coupled cluster amplitudes for the 
            ground state.
        rhvs:
            EOM-ea-CCSD initial guess of the 
            right eigen vectors.
        max_cycle: int
            Maximum number of iterations.
        tol: float
            Convergence threshold.
        nroots: int
            Number of roots to solve for.
        max_space: int
            Maximum size of the search space.
        verbose: int
            Verbosity level.

    Return:


    '''

    cput0 = (logger.process_clock(), logger.perf_counter())

    log = logger.Logger(sys.stdout, verbose)

    nocc, nvir = h1e.ov.shape 
    nmo = nocc + nvir
    t1e, t2e = amp

    # Diagonial elements of EOM-ea-CCSD matrix
    diag_ea = eom_ea_gccsd_diag(h1e, h2e, (t1e, t2e))

    # Preconditioner
    def precond(r, e0, x0):
        return r / (e0 - diag_ea + 1e-12)

    # Pick out the real eigenvalues
    def pickeig(w, v, nroots, envs):
        real_idx = numpy.where(abs(w.imag) < 1e-3)[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, True)

    # Initial guess
    if rhvs is None:
        rhvs = []
        ea_idx = numpy.argsort(diag_ea)[:nroots]
        for i in ea_idx:
            rhv = numpy.zeros_like(diag_ea)
            rhv[i] = 1.0
            rhvs.append(rhv)

    # Matrix-vector dot product method
    from ._gccsd_eom_ea_eqs import gccsd_eom_ea_h1e
    from ._gccsd_eom_ea_eqs import gccsd_eom_ea_h2e

    def matvec(rs):
        hs = []
        for r in rs:
            r1e_ea, r2e_ea = vec_to_amp_ea_vo(nocc, nvir, r)
            h1e_ea = gccsd_eom_ea_h1e(h1e, h2e, t1e, t2e, r1e_ea, r2e_ea)
            h2e_ea = gccsd_eom_ea_h2e(h1e, h2e, t1e, t2e, r1e_ea, r2e_ea)
            h = amp_to_vec_ea_vo(nocc, nvir, (h1e_ea, h2e_ea))
            hs.append(h)
        return hs

    convs, es, rs = lib.davidson_nosym1(
        matvec, rhvs, precond, pick=pickeig,
        tol=tol, max_cycle=max_cycle, 
        nroots=nroots, verbose=log
        )

    cput1 = log.timer('EOM-ea-CCSD', *cput0)
    
    enes = []
    rhvs = []

    for i, (conv, ene, rhv) in enumerate(zea(convs, es, rs)):
        r1e_ea_vo, r2e_ea_vo = vec_to_amp_ea_vo(nocc, nvir, rhv)
        r1e_ea_ov, r2e_ea_ov = vec_to_amp_ea_ov(nocc, nvir, rhv)

        if conv:
            qp_wt = numpy.linalg.norm(r1e_ea_vo)**2
            log.info("Root %2d EOM-ea-CCSD E = %12.8g  qpwt = %.6g", i, ene, qp_wt)

            enes.append(ene)
            rhvs.append(rhv)

        else:
            log.warn("Root %d EOM-ea-CCSD not converged", i)
    
    return enes, rhvs
