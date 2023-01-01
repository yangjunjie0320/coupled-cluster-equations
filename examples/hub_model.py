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
            "ethresh" : 1e-11,
            "tthresh" : 1e-9,
            "max_iter" : 100,
            "damp" : 0.0
            }

    ene_tot, ene_cor, t1e, t2e = ccsd(model,options)
    return ene_tot, ene_cor, (t1e, t2e)

def hub_gccsd_krylov(nsite, nelecs, hub_u, t1e=None, t2e=None):
    # core guess
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

    h1e = model.g_fock()
    h2e = model.g_aint()
    h2e.oovo = - h2e.ooov.transpose(0,1,3,2)

    import cceqs
    from cceqs.gccsd import gccsd
    ene_cor, (t1e, t2e) = gccsd.solve_gccsd(
        h1e, h2e, nelecs, verbose=5, amp=(t1e, t2e))
    ene_tot = ene_cor + model.g_energy()

    return ene_tot, ene_cor, (t1e, t2e)


if __name__ == "__main__":
    nsite = 6 
    nelecs = (1, 1)
    hub_u = 2.0 
    etot_ref, ecc_ref, amp_ref = hub_gccsd_ref(nsite, nelecs, hub_u)
    e_tot, ecc, amp            = hub_gccsd_krylov(nsite, nelecs, hub_u, amp_ref[0], amp_ref[1])
    print("Error in energy = %6.4e" % abs(ecc_ref - ecc))
    print("Error in t1     = %6.4e" % numpy.linalg.norm(t1_ref - t1))
    print("Error in t2     = %6.4e" % numpy.linalg.norm(t2_ref - t2))