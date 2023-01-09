import sys
import numpy, scipy

from pyscf import cc, lib
from pyscf.lib import logger
from pyscf.cc.eom_gccsd import vector_to_amplitudes_ip as vector_to_amplitudes_ip_ov
from pyscf.cc.eom_gccsd import amplitudes_to_vector_ip as amplitudes_to_vector_ip_ov

def vec_to_amp_ip_ov(no, nv, r=None):
    r1e_ip_ov, r2e_ip_ov = vector_to_amplitudes_ip_ov(r, no+nv, no)
    assert r1e_ip_ov.shape == (no,)
    assert r2e_ip_ov.shape == (no, no, nv,)
    return r1e_ip_ov, r2e_ip_ov

def amp_to_vec_ip_ov(no, nv, r_ip_ov=None):
    r1e_ip_ov, r2e_ip_ov = r_ip_ov
    assert r1e_ip_ov.shape == (no,)
    assert r2e_ip_ov.shape == (no, no, nv,)
    r = amplitudes_to_vector_ip_ov(r1e_ip_ov, r2e_ip_ov)
    return r

def vec_to_amp_ip_vo(no, nv, r=None):
    r1e_ip_ov, r2e_ip_ov = vector_to_amplitudes_ip_ov(r, no+nv, no)
    assert r1e_ip_ov.shape == (no,)
    assert r2e_ip_ov.shape == (no, no, nv,)

    r1e_ip_vo, r2e_ip_vo = _transpose_ov_to_vo(no, nv, r_ip_ov=(r1e_ip_ov, r2e_ip_ov))
    assert r1e_ip_vo.shape == (no,)
    assert r2e_ip_vo.shape == (nv, no, no,)
    return r1e_ip_vo, r2e_ip_vo

def amp_to_vec_ip_vo(no, nv, r_ip_vo=None):
    r1e_ip_ov, r2e_ip_ov = _transpose_vo_to_ov(no, nv, r_ip_vo=r_ip_vo)
    r = amplitudes_to_vector_ip_ov(r1e_ip_ov, r2e_ip_ov)
    return r

def _transpose_ov_to_vo(no, nv, r_ip_ov=None):
    r1e_ov, r2e_ov = r_ip_ov
    assert r1e_ov.shape == (no,)
    assert r2e_ov.shape == (no, no, nv,)

    r1e_vo = r1e_ov
    r2e_vo = r2e_ov.transpose(2, 0, 1)
    return r1e_vo, r2e_vo

def _transpose_vo_to_ov(no, nv, r_ip_vo=None):
    r1e_vo, r2e_vo = r_ip_vo
    assert r1e_vo.shape == (no,)
    assert r2e_vo.shape == (nv, no, no,)

    r1e_ov = r1e_vo
    r2e_ov = r2e_vo.transpose(1, 2, 0)
    return r1e_ov, r2e_ov

def eom_ip_gccsd_diag(h1e, h2e, r_ip_vo=None):
    from cceqs.gccsd import gccsd
    from pyscf.cc import gintermediates as imd
    nocc, nvir = h1e.ov.shape

    t1e_vo, t2e_vo = r_ip_vo
    t1e_ov, t2e_ov = gccsd._transpose_vo_to_ov(nocc, nvir, r_ip_vo)

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

    diag = amp_to_vec_ip_ov(nocc, nvir, (d1e_ov, d2e_ov))
    return diag

def solve_eom_ip_gccsd(h1e, h2e, amp=None, r_ip=None, max_cycle=50, tol=1e-8, 
                       nroots=5, max_space=20, verbose=3):
    '''Solve the IP EOM-CCSD equations.

    Args:
        h1e : 1-electron part of the Hamiltonian
        h2e : 2-electron part of the Hamiltonian
        amp : initial guess of amplitudes
        max_cycle : max number of iterations
        tol : convergence threshold
        verbose : verbosity level

    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(sys.stdout, verbose)

    nocc, nvir = h1e.ov.shape 
    nmo = nocc + nvir

    t1e, t2e = amp

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

    diag_ip = eom_ip_gccsd_diag(h1e, h2e, (t1e, t2e))

    def precond(r, e0, x0):
        return r / (e0 - diag_ip + 1e-12)

    rs = None

    if r_ip is None:
        rs   = []
        r_ip = []

        ip_idx = numpy.argsort(diag_ip)[:nroots]
        for i in ip_idx:
            r = numpy.zeros_like(diag_ip)
            r[i] = 1.0

            r1e_ip, r2e_ip = vec_to_amp_ip_vo(nocc, nvir, r)
            r_ip.append((r1e_ip, r2e_ip))
            rs.append(r)

    else:
        for r1e_ip, r2e_ip in r_ip:
            r = amp_to_vec_ip_vo(nocc, nvir, (r1e_ip, r2e_ip))
            rs.append(r)

    def pickeig(w, v, nroots, envs):
        real_idx = numpy.where(abs(w.imag) < 1e-3)[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, True)

    convs, es, vs = lib.davidson_nosym1(
        matvec, rs, precond, pick=pickeig,
        tol=tol, max_cycle=max_cycle, 
        nroots=nroots, verbose=log
        )

    print(es)
    print('converged:', convs)