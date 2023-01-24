import numpy, scipy, sys
from sys import stdout

import pyscf
from pyscf import gto, scf, ao2mo, cc
from pyscf.tools.dump_mat import dump_rec

from cqcpy import utils
from cqcpy.ov_blocks import two_e_blocks_full

from epcc.hubbard import Hubbard1D
from epcc.ccsd import ccsd
from epcc.ccsd import ccsd_gen

import cceqs
from cceqs.gccsd import gccsd

def get_ghf_h1e_h2e(mf=None):
    assert isinstance(mf, scf.ghf.GHF)
    assert mf.converged

    cc_obj = cc.GCCSD(mf)
    eris   = cc_obj.ao2mo()
    class _FockBlocks(object):
        pass

    class _ERIBlocks(object):
        pass
    
    nocc = eris.nocc
    fock = eris.fock

    fock_blocks = _FockBlocks()
    fock_blocks.oo = fock[:nocc, :nocc]
    fock_blocks.vv = fock[nocc:, nocc:]
    fock_blocks.ov = fock[:nocc, nocc:]
    fock_blocks.vo = fock[nocc:, :nocc]

    eris.vvoo = eris.oovv.transpose(2, 3, 0, 1).copy()
    eris.ovoo = eris.ooov.transpose(2, 3, 0, 1).copy()
    eris.vvov = eris.ovvv.transpose(2, 3, 0, 1).copy()

    return fock_blocks, eris

if __name__ == "__main__":
    mol = gto.Mole()
    mol.atom = """
        O   -0.0000000   -0.0781681   -0.0000000
        H    0.0000000    0.5280145   -0.7830365
        H   -0.0000000    0.5280145    0.7830365
    """
    mol.basis = "sto3g"
    mol.verbose = 0
    mol.build()

    mf = scf.GHF(mol)
    mf.kernel()

    cc_obj = cc.GCCSD(mf)
    cc_obj.kernel()

    nroots = 5
    enes_ref, rhvs_ref = cc_obj.ipccsd(nroots=nroots)

    from cceqs.gccsd.gccsd        import solve_gccsd
    from cceqs.gccsd.eom_ip_gccsd import solve_eom_ip_gccsd
    h1e, h2e = get_ghf_h1e_h2e(mf)
    no, nv   = h1e.ov.shape
    ene_tot, ene_cor, (t1e, t2e) = solve_gccsd(
        h1e, h2e, verbose=0, 
        amp=None, max_cycle=50, tol=1e-8
        )

    enes, rhvs = solve_eom_ip_gccsd(
        h1e, h2e, amp=(t1e, t2e), rhvs=None, 
        max_cycle=50, tol=1e-10, 
        max_space=nroots * 10, 
        nroots=nroots, verbose=4
        )

    from pyscf.tools.dump_mat import dump_rec
    from cceqs.gccsd.eom_ip_gccsd import vec_to_amp_ip_vo
    print("\nIP reference energies:")
    for i in range(nroots):
        r1e_ip_ref, r2e_ip_ref = vec_to_amp_ip_vo(no, nv, rhvs_ref[i])
        r1e_ip, r2e_ip         = vec_to_amp_ip_vo(no, nv, rhvs[i])

        dot  = abs(numpy.dot(rhvs_ref[i], rhvs[i]))
        print("Root %2d ene = %10.8f, err = %6.4e %6.4e" % (
            i, enes[i], abs(enes[i] - enes_ref[i]), abs(1.0 - dot)
            ))
        