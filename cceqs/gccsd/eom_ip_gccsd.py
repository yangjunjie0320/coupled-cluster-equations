import sys
import numpy, scipy

from pyscf import cc, lib
from pyscf.lib import logger
from pyscf.cc.eom_gccsd import vector_to_amplitudes_ip as vector_to_amplitudes_ip_ov
from pyscf.cc.eom_gccsd import amplitudes_to_vector_ip as amplitudes_to_vector_ip_ov

def vec_to_amp_ip_ov(no, nv, vec=None):
    r1e_ip_ov, r2e_ip_ov = vector_to_amplitudes_ip_ov(vec, no+nv, no)
    assert r1e_ip_ov.shape == (no,)
    assert r2e_ip_ov.shape == (no, no, nv,)
    return r1e_ip_ov, r2e_ip_ov

def amp_to_vec_ip_ov(no, nv, amp=None):
    r1e_ip_ov, r2e_ip_ov = amp
    assert r1e_ip_ov.shape == (no,)
    assert r2e_ip_ov.shape == (no, no, nv,)
    vec = amplitudes_to_vector_ip_ov(r1e_ip_ov, r2e_ip_ov)
    return vec

def vec_to_amp_ip_vo(no, nv, vec=None):
    r1e_ip_ov, r2e_ip_ov = vector_to_amplitudes_ip_ov(vec, no+nv, no)
    assert r1e_ip_ov.shape == (no,)
    assert r2e_ip_ov.shape == (no, no, nv,)

    r1e_ip_vo, r2e_ip_vo = _transpose_ov_to_vo(no, nv, amp=(r1e_ip_ov, r2e_ip_ov))
    assert r1e_ip_vo.shape == (no,)
    assert r2e_ip_vo.shape == (nv, no, no,)
    return r1e_ip_vo, r2e_ip_vo

def amp_to_vec_ip_vo(no, nv, amp=None):
    r1e_ip_ov, r2e_ip_ov = _transpose_vo_to_ov(no, nv, amp=amp)
    vec = amplitudes_to_vector_ip_ov(r1e_ip_ov, r2e_ip_ov)
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

def eom_ip_gccsd_diag(h1e, h2e, amp=None):
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

    d1e_ov = numpy.diag(foo) * (-1.0)
    d2e_ov = numpy.zeros((nocc, nocc, nvir))

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                d2e_ov[i, j, a] += fvv[a, a]
                d2e_ov[i, j, a] -= foo[i, i]
                d2e_ov[i, j, a] -= foo[j, j]
                d2e_ov[i, j, a] += 0.5 * w_oooo[i, j, i, j]
                d2e_ov[i, j, a] -= 0.5 * w_oooo[j, i, i, j]
                d2e_ov[i, j, a] += w_ovvo[i, a, a, i]
                d2e_ov[i, j, a] += w_ovvo[j, a, a, i]
                d2e_ov[i, j, a] += 0.5 * numpy.dot(w_oovv[i, j, :, a], t2e_ov[i, j, a, :])
                d2e_ov[i, j, a] -= 0.5 * numpy.dot(w_oovv[j, i, :, a], t2e_ov[i, j, a, :])

    diag_ip = amp_to_vec_ip_ov(nocc, nvir, (d1e_ov, d2e_ov))
    return diag_ip

def solve_eom_ip_gccsd(h1e, h2e, amp=None, rhvs=None, max_cycle=50, tol=1e-8, 
                       nroots=5, max_space=20, verbose=3):
    '''Solve the IP EOM-CCSD equations.

    Args:
        h1e:
            1-electron integrals
        h2e: 
            2-electron integrals
        amp: 
            Coupled cluster amplitudes for the 
            ground state.
        rhvs:
            EOM-IP-CCSD initial guess of the 
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

    # Diagonial elements of EOM-IP-CCSD matrix
    diag_ip = eom_ip_gccsd_diag(h1e, h2e, (t1e, t2e))

    # Preconditioner
    def precond(r, e0, x0):
        return r / (e0 - diag_ip + 1e-12)

    # Pick out the real eigenvalues
    def pickeig(w, v, nroots, envs):
        real_idx = numpy.where(abs(w.imag) < 1e-3)[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, True)

    # Initial guess
    if rhvs is None:
        rhvs = []
        ip_idx = numpy.argsort(diag_ip)[:nroots]
        for i in ip_idx:
            rhv = numpy.zeros_like(diag_ip)
            rhv[i] = 1.0
            rhvs.append(rhv)

    # Matrix-vector dot product method
    from ._gccsd_eom_ip_eqs import gccsd_eom_ip_h1e
    from ._gccsd_eom_ip_eqs import gccsd_eom_ip_h2e

    def matvec(rs):
        hs = []
        for r in rs:
            r1e_ip, r2e_ip = vec_to_amp_ip_vo(nocc, nvir, r)
            h1e_ip = gccsd_eom_ip_h1e(h1e, h2e, t1e, t2e, r1e_ip, r2e_ip)
            h2e_ip = gccsd_eom_ip_h2e(h1e, h2e, t1e, t2e, r1e_ip, r2e_ip)
            h = amp_to_vec_ip_vo(nocc, nvir, (h1e_ip, h2e_ip))
            hs.append(h)
        return hs

    convs, es, rs = lib.davidson_nosym1(
        matvec, rhvs, precond, pick=pickeig,
        tol=tol, max_cycle=max_cycle, 
        nroots=nroots, verbose=log
        )

    cput1 = log.timer('EOM-IP-CCSD', *cput0)
    
    enes = []
    rhvs = []

    for i, (conv, ene, rhv) in enumerate(zip(convs, es, rs)):
        r1e_ip_vo, r2e_ip_vo = vec_to_amp_ip_vo(nocc, nvir, rhv)
        r1e_ip_ov, r2e_ip_ov = vec_to_amp_ip_ov(nocc, nvir, rhv)

        if conv:
            qp_wt = numpy.linalg.norm(r1e_ip_vo)**2
            log.info("Root %2d EOM-IP-CCSD E = %12.8g  qpwt = %.6g", i, ene, qp_wt)

            enes.append(ene)
            rhvs.append(rhv)

        else:
            log.warn("Root %d EOM-IP-CCSD not converged", i)
    
    return enes, rhvs
