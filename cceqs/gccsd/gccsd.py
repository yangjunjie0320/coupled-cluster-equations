import sys
import numpy, scipy

from pyscf import cc
from pyscf.lib import logger
from pyscf.cc.gccsd import vector_to_amplitudes as vector_to_amplitudes_ov
from pyscf.cc.gccsd import amplitudes_to_vector as amplitudes_to_vector_ov

def vec_to_amp_ov(no, nv, v=None):
    t1e_ov, t2e_ov = vector_to_amplitudes_ov(v, no+nv, no)
    assert t1e_ov.shape == (no, nv)
    assert t2e_ov.shape == (no, no, nv, nv)
    return t1e_ov, t2e_ov

def amp_to_vec_ov(no, nv, amp_ov=None):
    t1e_ov, t2e_ov = amp_ov
    assert t1e_ov.shape == (no, nv)
    assert t2e_ov.shape == (no, no, nv, nv)
    v = amplitudes_to_vector_ov(t1e_ov, t2e_ov)
    return v

def vec_to_amp_vo(no, nv, v=None):
    t1e_ov, t2e_ov = vector_to_amplitudes_ov(v, no+nv, no)
    assert t1e_ov.shape == (no, nv)
    assert t2e_ov.shape == (no, no, nv, nv)
    amp_ov = (t1e_ov, t2e_ov)

    t1e_vo, t2e_vo = transpose_ov_to_vo(no, nv, amp_ov)
    assert t1e_vo.shape == (nv, no)
    assert t2e_vo.shape == (nv, nv, no, no)
    return t1e_vo, t2e_vo

def amp_to_vec_vo(no, nv, amp_vo=None):
    t1e_vo, t2e_vo = amp_vo
    assert t1e_vo.shape == (nv, no)
    assert t2e_vo.shape == (nv, nv, no, no)

    t1e_ov, t2e_ov = transpose_vo_to_ov(no, nv, amp_vo)
    assert t1e_ov.shape == (no, nv)
    assert t2e_ov.shape == (no, no, nv, nv)
    v = amplitudes_to_vector_ov(t1e_ov, t2e_ov)
    return v

def transpose_ov_to_vo(no, nv, amp_ov=None):
    t1e_ov, t2e_ov = amp_ov
    assert t1e_ov.shape == (no, nv)
    assert t2e_ov.shape == (no, no, nv, nv)

    t1e_vo = t1e_ov.transpose()
    t2e_vo = t2e_ov.transpose(2, 3, 0, 1)
    return t1e_vo, t2e_vo

def transpose_vo_to_ov(no, nv, amp_vo=None):
    t1e_vo, t2e_vo = amp_vo
    assert t1e_vo.shape == (nv, no)
    assert t2e_vo.shape == (nv, nv, no, no)

    t1e_ov = t1e_vo.transpose()
    t2e_ov = t2e_vo.transpose(2, 3, 0, 1)
    return t1e_ov, t2e_ov

def solve_gccsd(h1e, h2e, amp=None, max_cycle=50, tol=1e-8, verbose=0):
    '''Solve CCSD lambda equations for given amplitudes.

    Args:
        h1e : fock matrix in MO basis, with the following attributes
              ov, vo, oo, and vv blocks.
        h2e : anti-symmetrized 2e integrals in MO basis, with the following
              attributes:
              - vvvv, vvvo, vvov, vvoo
              - vovv, vovo, voov, vooo
              - ovvv, ovvo, ovov, ovoo
              - oovv, oovo, ooov, oooo
        amp : initial guess of the t1e and t2e amplitudes.  If None, the
              amplitudes are initialized to MP2 amplitudes.
        max_cycle : max number of iterations
        tol : convergence threshold
        verbose : verbosity level

    Returns:
        ene_tot: ene_hf + ene_cor
        ene_cor: CCSD correlation energy
        t1e, t2e: t1e_vo and t2e_vo amplitudes
    '''
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(sys.stdout, verbose)

    nocc, nvir = h1e.ov.shape 
    nmo = nocc + nvir

    ene_hf  = h1e.oo.diagonal().sum()
    ene_hf += numpy.einsum("iijj->", h2e.oooo)
    ene_hf -= numpy.einsum("ijij->", h2e.oooo) * 0.5

    log.info("")
    log.info("HF energy             = % 12.8f", ene_hf)

    if amp is None:
        eo  = h1e.oo.diagonal()
        ev  = h1e.vv.diagonal()
        d1  = eo[:,None] - ev[None,:]
        d2  = (eo[:,None,None,None] + eo[None,:,None,None] - ev[None,None,:,None] - ev[None,None,None,:])
        t1e = h1e.vo / d1.transpose((1,0))
        t2e = h2e.vvoo / d2.transpose((2,3,0,1))

        ene_mp2  = numpy.einsum("ai,ia->", t1e, h1e.ov)
        ene_mp2 += numpy.einsum("abij,ijab->", t2e, h2e.oovv) * 0.25
        log.info("MP2 energy            = % 12.8f", ene_mp2)
    else:
        t1e, t2e = amp

    from ._gccsd_amp_eqs import gccsd_ene
    from ._gccsd_amp_eqs import gccsd_r1e
    from ._gccsd_amp_eqs import gccsd_r2e

    def res(x):
        t1e, t2e = vec_to_amp_vo(nocc, nvir, x)

        r1e = gccsd_r1e(h1e, h2e, t1e, t2e)
        r2e = gccsd_r2e(h1e, h2e, t1e, t2e)

        r = amp_to_vec_vo(nocc, nvir, (r1e, r2e))
        return r

    x0 = amp_to_vec_vo(nocc, nvir, (t1e, t2e))
    log.info('Initial CCSD energy   = % 12.8f', gccsd_ene(h1e, h2e, t1e, t2e))
    log.info('Initial CCSD residual = % 12.4e', numpy.linalg.norm(res(x0)))

    from scipy import optimize
    sol = optimize.newton_krylov(
        res, x0, f_tol=tol, 
        maxiter=max_cycle,
        verbose=(verbose > 4)
        )

    t1e, t2e = vec_to_amp_vo(nocc, nvir, sol)
    ene_cor  = gccsd_ene(h1e, h2e, t1e, t2e)

    log.info('Final CCSD energy     = %12.8f', ene_cor)
    log.info('Final CCSD residual   = %12.4e', numpy.linalg.norm(res(sol)))
    log.info('Total CCSD energy     = %12.8f', ene_cor+ene_hf)
    cput1 = log.timer('CCSD', *cput0)

    return ene_cor+ene_hf, ene_cor, (t1e, t2e)

def solve_gccsd_lambda(h1e, h2e, amp=None, lam=None, max_cycle=50, tol=1e-8, verbose=0):
    '''Solve CCSD lambda equations for given amplitudes.

    Args:
        h1e : fock matrix in MO basis, with the following attributes
              ov, vo, oo, and vv blocks.
        h2e : anti-symmetrized 2e integrals in MO basis, with the following
              attributes:
              - vvvv, vvvo, vvov, vvoo
              - vovv, vovo, voov, vooo
              - ovvv, ovvo, ovov, ovoo
              - oovv, oovo, ooov, oooo
        amp : t1e and t2e amplitudes, in vo shape.
        lam : initial guess of the coupled cluster lambda,
              in vo shape, if None, the lambda will be initialized to
              the corresponding t1e and t2e.
        max_cycle : max number of iterations
        tol : convergence threshold
        verbose : verbosity level
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(sys.stdout, verbose)

    nocc, nvir = h1e.ov.shape 
    nmo = nocc + nvir

    assert amp is not None
    t1e, t2e = amp

    if lam is None:
        l1e = t1e.copy()
        l2e = t2e.copy()
        lam = (l1e, l2e)

    l1e, l2e = lam

    from ._gccsd_lam_eqs import gccsd_lam_rhs1e, gccsd_lam_lhs1e
    from ._gccsd_lam_eqs import gccsd_lam_rhs2e, gccsd_lam_lhs2e

    def res(x):
        l1e_vo, l2e_vo = vec_to_amp_vo(nocc, nvir, x)
        l1e_ov, l2e_ov = transpose_vo_to_ov(nocc, nvir, (l1e_vo, l2e_vo))

        r1e_ov  = gccsd_lam_lhs1e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
        r1e_ov += gccsd_lam_rhs1e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)

        r2e_ov  = gccsd_lam_lhs2e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
        r2e_ov += gccsd_lam_rhs2e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)

        r1e_vo, r2e_vo = transpose_ov_to_vo(nocc, nvir, (r1e_ov, r2e_ov))
        r = amp_to_vec_vo(nocc, nvir, (r1e_vo, r2e_vo))
        return r

    x0 = amp_to_vec_vo(nocc, nvir, (l1e, l2e))
    log.info('Initial CCSD lambda equation residual = % 12.4e', numpy.linalg.norm(res(x0)))

    from scipy import optimize
    sol = optimize.newton_krylov(
        res, x0, f_tol=tol, 
        maxiter=max_cycle,
        verbose=(verbose > 4)
        )

    l1e_vo, l2e_vo = vec_to_amp_vo(nocc, nvir, sol)

    return (l1e_vo, l2e_vo)