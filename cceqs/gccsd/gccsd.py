import sys
import numpy, scipy

from pyscf import cc
from pyscf.lib import logger
from pyscf.cc.gccsd import vector_to_amplitudes, amplitudes_to_vector

class _Fock:
    pass

class _PhysicistsERIs:
    pass

def vec_to_amp(no, nv, v=None):
    amp = vector_to_amplitudes(v, no+nv, no)
    return transpose(amp)

def amp_to_vec(no, nv, amp=None):
    t1t, t2t = transpose(amp)
    return amplitudes_to_vector(t1t, t2t)

def transpose(amp):
    t1, t2 = amp
    return t1.transpose(), t2.transpose(2, 3, 1, 0)

def solve_gccsd(h1e, h2e, nelecs, amp=None, max_cycle=50, tol=1e-8, verbose=0):
    '''Solve CCSD equations for given Hamiltonian and number of electrons.
    '''
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(sys.stdout, verbose)

    nelec_alph, nelec_beta = nelecs
    nocc, nvir = h1e.ov.shape 
    nmo = nocc + nvir
    assert nocc == nelec_alph + nelec_beta

    if amp is None:
        t1e = numpy.zeros((nvir, nocc))
        t2e = numpy.zeros((nvir, nvir, nocc, nocc))
    else:
        t1e, t2e = amp

    from ._gccsd_amp_eqs import gccsd_ene
    from ._gccsd_amp_eqs import gccsd_r1e
    from ._gccsd_amp_eqs import gccsd_r2e

    def res(x):
        t1e, t2e = vec_to_amp(nocc, nvir, x)
        r1e = gccsd_r1e(h1e, h2e, t1e, t2e)
        r2e = gccsd_r2e(h1e, h2e, t1e, t2e)
        return amp_to_vec(nocc, nvir, (r1e, r2e))

    x0 = amp_to_vec(nocc, nvir, (t1e, t2e))
    log.info("")
    log.info('Initial CCSD energy   = % 12.8f', gccsd_ene(h1e, h2e, t1e, t2e))
    log.info('Initial CCSD residual = % 6.4e', numpy.linalg.norm(res(x0)))

    from scipy import optimize
    sol = optimize.newton_krylov(
        res, x0, f_tol=tol, 
        maxiter=max_cycle,
        verbose=(verbose > 3)
        )

    t1e, t2e = vec_to_amp(nocc, nvir, sol)
    log.info('Final CCSD energy   = %12.8f', gccsd_ene(h1e, h2e, t1e, t2e))
    cput1 = log.timer('CCSD', *cput0)

    return ene_cor, (t1e, t2e)