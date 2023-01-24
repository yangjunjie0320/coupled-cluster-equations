import sys
import numpy, scipy

from pyscf import cc
from pyscf.lib import logger
from pyscf.cc.gccsd import vector_to_amplitudes as vector_to_amplitudes_ov
from pyscf.cc.gccsd import amplitudes_to_vector as amplitudes_to_vector_ov

from cceqs.cc import CoupledClusterAmplitudeSolverMixin
from cceqs.cc import CoupledClusterLambdaSolverMixin

from ._gccsd_amp_eqs import gccsd_ene
from ._gccsd_amp_eqs import gccsd_r1e
from ._gccsd_amp_eqs import gccsd_r2e

from ._gccsd_lam_eqs import gccsd_lam_rhs1e
from ._gccsd_lam_eqs import gccsd_lam_lhs1e
from ._gccsd_lam_eqs import gccsd_lam_rhs2e
from ._gccsd_lam_eqs import gccsd_lam_lhs2e

def _transpose_ov_to_vo(no, nv, amp=None):
    t1e_ov, t2e_ov = amp
    assert t1e_ov.shape == (no, nv)
    assert t2e_ov.shape == (no, no, nv, nv)

    t1e_vo = t1e_ov.transpose()
    t2e_vo = t2e_ov.transpose(2, 3, 0, 1)
    return t1e_vo, t2e_vo

def _transpose_vo_to_ov(no, nv, amp=None):
    t1e_vo, t2e_vo = amp
    assert t1e_vo.shape == (nv, no)
    assert t2e_vo.shape == (nv, nv, no, no)

    t1e_ov = t1e_vo.transpose()
    t2e_ov = t2e_vo.transpose(2, 3, 0, 1)
    return t1e_ov, t2e_ov

class AmplitudeSolver(CoupledClusterAmplitudeSolverMixin):
    def __init__(self, h1e, h2e, verbose=3):
        self.h1e = h1e
        self.h2e = h2e
        self.verbose = verbose

        nocc, nvir = h1e.ov.shape
        self.nocc = nocc
        self.nvir = nvir

    def get_init_amp(self):
        h1e = self.h1e
        h2e = self.h2e
        assert h1e is not None
        assert h2e is not None

        eo  = h1e.oo.diagonal()
        ev  = h1e.vv.diagonal()
        d1  = eo[:,None] - ev[None,:]
        d2  = (eo[:,None,None,None] + eo[None,:,None,None] - ev[None,None,:,None] - ev[None,None,None,:])
        t1e = h1e.vo / d1.transpose((1,0))
        t2e = h2e.vvoo / d2.transpose((2,3,0,1))

        ene_mp2  = numpy.einsum("ai,ia->", t1e, h1e.ov)
        ene_mp2 += numpy.einsum("abij,ijab->", t2e, h2e.oovv) * 0.25

        log = logger.new_logger(self)
        log.info("MP2 energy            = % 12.8f", ene_mp2)

        return (t1e, t2e)

    def get_ene_hf(self):
        h1e = self.h1e
        h2e = self.h2e
        ene_hf  = h1e.oo.diagonal().sum()
        ene_hf += numpy.einsum("iijj->", h2e.oooo)
        ene_hf -= numpy.einsum("ijij->", h2e.oooo) * 0.5
        return ene_hf

    def get_ene_cor(self, amp=None):
        h1e, h2e = self.h1e, self.h2e
        t1e, t2e = amp
        ene_cor  = gccsd_ene(h1e, h2e, t1e, t2e)
        return ene_cor

    def gen_res_func(self):
        log = logger.new_logger(self)
        h1e, h2e = self.h1e, self.h2e
        assert h1e is not None
        assert h2e is not None

        iter_ccsd = 0

        def res_func(vec, verbose=True):
            t1e, t2e = self.vec_to_amp(vec)
            ene_cor = self.get_ene_cor(amp=(t1e, t2e))

            r1e = gccsd_r1e(h1e, h2e, t1e, t2e)
            r2e = gccsd_r2e(h1e, h2e, t1e, t2e)
            res = self.res_to_vec((r1e, r2e))

            nonlocal iter_ccsd

            if verbose:
                log.info('CCSD iter %4d, energy = %12.8f, residual = %12.4e',
                        iter_ccsd, ene_cor, numpy.linalg.norm(res))
                iter_ccsd += 1

            return res

        return res_func

    def amp_to_vec(self, amp):
        nocc, nvir = self.nocc, self.nvir
        t1e_vo, t2e_vo = amp
        assert t1e_vo.shape == (nvir, nocc)
        assert t2e_vo.shape == (nvir, nvir, nocc, nocc,)
        t1e_ov, t2e_ov = _transpose_vo_to_ov(nocc, nvir, amp=(t1e_vo, t2e_vo))
        return amplitudes_to_vector_ov(t1e_ov, t2e_ov)

    def vec_to_amp(self, vec):
        nocc, nvir = self.nocc, self.nvir
        t1e_ov, t2e_ov = vector_to_amplitudes_ov(vec, nocc+nvir, nocc)
        t1e_vo, t2e_vo = _transpose_ov_to_vo(nocc, nvir, amp=(t1e_ov, t2e_ov))
        return t1e_vo, t2e_vo

    def res_to_vec(self, res):
        nocc, nvir = self.nocc, self.nvir
        r1e_vo, r2e_vo = res
        assert r1e_vo.shape == (nvir, nocc)
        assert r2e_vo.shape == (nvir, nvir, nocc, nocc)
        r1e_ov, r2e_ov = _transpose_vo_to_ov(nocc, nvir, amp=(r1e_vo, r2e_vo))
        return amplitudes_to_vector_ov(r1e_ov, r2e_ov)

    def vec_to_res(self, vec):
        nocc, nvir = self.nocc, self.nvir
        r1e_ov, r2e_ov = vector_to_amplitudes_ov(vec, nocc+nvir, nocc)
        r1e_vo, r2e_vo = _transpose_ov_to_vo(nocc, nvir, amp=(r1e_ov, r2e_ov))
        return r1e_vo, r2e_vo

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
        l1e_ov, l2e_ov = _transpose_vo_to_ov(nocc, nvir, (l1e_vo, l2e_vo))

        r1e_ov  = gccsd_lam_lhs1e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
        r1e_ov += gccsd_lam_rhs1e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)

        r2e_ov  = gccsd_lam_lhs2e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
        r2e_ov += gccsd_lam_rhs2e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)

        r1e_vo, r2e_vo = _transpose_ov_to_vo(nocc, nvir, (r1e_ov, r2e_ov))
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

class LambdaSolver(CoupledClusterLambdaSolverMixin):
    def __init__(self, h1e, h2e, amp=None, verbose=3):
        self.h1e = h1e
        self.h2e = h2e
        self.amp = amp
        self.verbose = verbose

        nocc, nvir = h1e.ov.shape
        self.nocc = nocc
        self.nvir = nvir

    def gen_res_func(self):
        log = logger.new_logger(self)
        h1e, h2e = self.h1e, self.h2e
        amp = self.amp

        assert h1e is not None
        assert h2e is not None
        assert amp is not None

        t1e, t2e = amp

        iter_ccsd = 0

        def res_func(vec, verbose=True):
            l1e, l2e = self.vec_to_lam(nocc, nvir, vec)

            r1e_ov  = gccsd_lam_lhs1e(h1e, h2e, t1e, t2e, l1e, l2e)
            r1e_ov += gccsd_lam_rhs1e(h1e, h2e, t1e, t2e, l1e, l2e)

            r2e_ov  = gccsd_lam_lhs2e(h1e, h2e, t1e, t2e, l1e, l2e)
            r2e_ov += gccsd_lam_rhs2e(h1e, h2e, t1e, t2e, l1e, l2e)

            r1e_vo, r2e_vo = _transpose_ov_to_vo(nocc, nvir, (r1e_ov, r2e_ov))
            r = amp_to_vec_vo(nocc, nvir, (r1e_vo, r2e_vo))

            nonlocal iter_ccsd

            if verbose:
                log.info('CCSD iter %4d, energy = %12.8f, residual = %12.4e',
                        iter_ccsd, ene_cor, numpy.linalg.norm(res))
                iter_ccsd += 1

            return res

        return res_func

    def amp_to_vec(self, amp):
        nocc, nvir = self.nocc, self.nvir
        t1e_vo, t2e_vo = amp
        assert t1e_vo.shape == (nvir, nocc)
        assert t2e_vo.shape == (nvir, nvir, nocc, nocc,)
        t1e_ov, t2e_ov = _transpose_vo_to_ov(nocc, nvir, amp=(t1e_vo, t2e_vo))
        return amplitudes_to_vector_ov(t1e_ov, t2e_ov)

    def vec_to_amp(self, vec):
        nocc, nvir = self.nocc, self.nvir
        t1e_ov, t2e_ov = vector_to_amplitudes_ov(vec, nocc+nvir, nocc)
        t1e_vo, t2e_vo = _transpose_ov_to_vo(nocc, nvir, amp=(t1e_ov, t2e_ov))
        return t1e_vo, t2e_vo

    def res_to_vec(self, res):
        nocc, nvir = self.nocc, self.nvir
        r1e_vo, r2e_vo = res
        assert r1e_vo.shape == (nvir, nocc)
        assert r2e_vo.shape == (nvir, nvir, nocc, nocc)
        r1e_ov, r2e_ov = _transpose_vo_to_ov(nocc, nvir, amp=(r1e_vo, r2e_vo))
        return amplitudes_to_vector_ov(r1e_ov, r2e_ov)

    def vec_to_res(self, vec):
        nocc, nvir = self.nocc, self.nvir
        r1e_ov, r2e_ov = vector_to_amplitudes_ov(vec, nocc+nvir, nocc)
        r1e_vo, r2e_vo = _transpose_ov_to_vo(nocc, nvir, amp=(r1e_ov, r2e_ov))
        return r1e_vo, r2e_vo