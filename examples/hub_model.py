import numpy, scipy

from cqcpy import utils
from cqcpy.ov_blocks import two_e_blocks_full

from epcc.hubbard import Hubbard1D
from epcc.ccsd import ccsd
from epcc.ccsd import ccsd_gen

def hub_gccsd_ref(nsite, nelec, hub_u):
    nelec_alph, nelec_beta= nelecs
    nelec = nelec_alph + nelec_beta

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
            "ethresh" : 1e-4,
            "tthresh" : 1e-6,
            "max_iter" : 100,
            "damp" : 0.0
        }

    ene_tot, ene_cor, t1e, t2e = ccsd(model,options)
    return ene_tot, ene_cor, (t1e, t2e)

def hub_gccsd_krylov(nsite, nelecs, hub_u, t1e=None, t2e=None):
    # core guess

    if t1e is None or t2e is None:
        amp = None
    else:
        amp = (t1e, t2e)

    nelec_alph, nelec_beta= nelecs
    nelec = nelec_alph + nelec_beta

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

    from cceqs.utils import g_aint_full
    h1e = model.g_fock()
    h2e = g_aint_full(model)

    import cceqs
    from cceqs.gccsd import gccsd
    ene_tot, ene_cor, (t1e, t2e) = gccsd.solve_gccsd(
        h1e, h2e, nelecs, verbose=5, amp=amp,
        max_cycle=50, tol=1e-6
        )

    return ene_tot, ene_cor, (t1e, t2e)


if __name__ == "__main__":
    nsite  = 8
    nelecs = (4, 4)
    hub_u  = 2.0 

    import sys
    from pyscf.lib import logger

    verbose = logger.DEBUG
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(sys.stdout, verbose)
    etot_ref, ecc_ref, amp_ref = hub_gccsd_ref(nsite, nelecs, hub_u)
    t1_ref, t2_ref = amp_ref
    print("Reference Total energy       = %16.8f" % etot_ref)
    print("Reference Correlation energy = %16.8f" % ecc_ref)
    log.timer('Hubbard model', *cput0)

    cput0 = (logger.process_clock(), logger.perf_counter())
    e_tot, ecc, amp            = hub_gccsd_krylov(nsite, nelecs, hub_u, None, None)
    t1, t2 = amp
    print("Total energy                 = %16.8f" % e_tot)
    print("Correlation energy           = %16.8f" % ecc)
    log.timer('Hubbard model', *cput0)

    print("Error in energy = %6.4e" % (abs(ecc_ref - ecc) + abs(etot_ref - e_tot)))
    print("Error in t1     = %6.4e" % numpy.linalg.norm(t1_ref - t1))
    print("Error in t2     = %6.4e" % numpy.linalg.norm(t2_ref - t2))