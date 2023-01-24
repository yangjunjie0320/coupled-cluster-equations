import sys
import numpy, scipy

from pyscf import lib
from pyscf.lib import logger

class CoupledClusterAmplitudeSolverMixin(lib.StreamObject):
    '''SCF base class.   non-relativistic RHF.

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default equals to :class:`Mole.max_memory`
        chkfile : str
            checkpoint file to save MOs, orbital energies etc.  Writing to
            chkfile can be disabled if this attribute is set to None or False.
        conv_tol : float
            converge threshold.  Default is 1e-9
    '''

    max_cycle = 50
    tol = 1e-6
    verbose = 5

    def __init__(self):
        raise NotImplementedError

    def gen_res_func(self):
        raise NotImplementedError

    def get_init_amp(self):
        raise NotImplementedError

    def get_ene_hf(self):
        raise NotImplementedError

    def get_ene_cor(self, amp):
        raise NotImplementedError

    def amp_to_vec(self, amp):
        raise NotImplementedError

    def vec_to_amp(self, vec):
        raise NotImplementedError

    def res_to_vec(self, res):
        raise NotImplementedError

    def vec_to_res(self, vec):
        raise NotImplementedError

    def kernel(self, amp=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log   = logger.new_logger(self)
        
        if amp is None:
            amp = self.get_init_amp()
            log.timer('Time to generate initial amplitudes', *cput0)

        ene_hf   = self.get_ene_hf()
        res_func = self.gen_res_func()
        vec_init = self.amp_to_vec(amp)
        res_init = res_func(vec_init, verbose=False)
        ene_init = self.get_ene_hf() + self.get_ene_cor(amp)

        log.info('Mean-field energy          = % 12.8f', ene_hf)
        log.info('Initial correlation energy = % 12.8f', ene_init - ene_hf)
        log.info('Initial total energy       = % 12.8f', ene_init)
        log.info('Initial residual norm      = % 12.4e\n', numpy.linalg.norm(res_init))

        from scipy import optimize
        vec_sol = optimize.newton_krylov(
            res_func, vec_init, 
            f_tol=self.tol, verbose=0,
            maxiter=self.max_cycle,
            )
        
        amp_sol = self.vec_to_amp(vec_sol)
        ene_sol = ene_hf + self.get_ene_cor(amp_sol)
        log.timer('\nTime to solve coupled-cluster amplitude equations', *cput0)

        log.info('\nFinal correlation energy   = % 12.8f', ene_sol - ene_hf)
        log.info('Final total energy         = % 12.8f', ene_sol)
        log.info('Final residual norm        = % 12.4e', numpy.linalg.norm(res_func(vec_sol, verbose=False)))

        return ene_sol, ene_sol - ene_hf, amp_sol

class CoupledClusterLambdaSolverMixin(object):
    max_cycle = 50
    tol = 1e-8
    verbose = 5

    def __init__(self):
        raise NotImplementedError

    def gen_res_func(self):
        raise NotImplementedError

    def get_init_lam(self):
        raise NotImplementedError

    def lam_to_vec(self, lam):
        raise NotImplementedError

    def vec_to_lam(self, vec):
        raise NotImplementedError

    def res_to_vec(self, res):
        raise NotImplementedError

    def vec_to_res(self, vec):
        raise NotImplementedError

    def kernel(self, amp=None, lam=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log   = logger.Logger(sys.stdout, verbose)
        
        assert amp is not None

        if lam is None:
            lam = amp

        res_func = self.gen_res_func()
        vec_init = self.lam_to_vec(lam)
        res_init = res_func(vec_init)

        log.info('Initial residual norm      = % 12.4e', numpy.linalg.norm(res_init))

        from scipy import optimize
        vec_sol = optimize.newton_krylov(
            res_func, vec_init, 
            f_tol=self.tol, verbose=0,
            maxiter=self.max_cycle,
            )
        
        lam_sol = self.lam_to_vec(vec_sol)
        log.timer('Time to solve coupled-cluster lambda equations', *cput0)
        log.info('Final residual norm        = % 12.4e', numpy.linalg.norm(res_func(vec_sol)))

        return lam_sol