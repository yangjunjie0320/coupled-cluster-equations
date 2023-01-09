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

def test_dr_dt(h1e, h2e, t1e, t2e, verbose=3, random_amp=False):
    from cceqs.gccsd._gccsd_amp_eqs import gccsd_ene
    from cceqs.gccsd._gccsd_amp_eqs import gccsd_r1e
    from cceqs.gccsd._gccsd_amp_eqs import gccsd_r2e
    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_lhs1e
    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_lhs2e

    from pyscf.lib import logger

    log = logger.new_logger(sys.stdout, verbose)

    nv, no  = t1e.shape
    x0      = gccsd.amp_to_vec_vo(no, nv, amp_vo=(t1e, t2e))

    def func(x):
        t1e, t2e = gccsd.vec_to_amp_vo(no, nv, x)
        r1e = gccsd_r1e(h1e, h2e, t1e, t2e)
        r2e = gccsd_r2e(h1e, h2e, t1e, t2e)
        r = gccsd.amp_to_vec_vo(no, nv, amp_vo=(r1e, r2e))
        return r

    def grad(x):
        t1e, t2e  = gccsd.vec_to_amp_vo(no, nv, x)

        dr_dt_vec = numpy.zeros((x.size, x.size))

        for r_idx in range(x.size):
            vec_lam = numpy.zeros_like(x)
            vec_lam[r_idx] = 1.0

            l1e_ov, l2e_ov = gccsd.vec_to_amp_ov(no, nv, vec_lam)
            lhs1e = gccsd_lam_lhs1e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
            lhs2e = gccsd_lam_lhs2e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
            dr_dt_ov = (lhs1e, lhs2e)

            vec_drdt = gccsd.amp_to_vec_ov(no, nv, amp_ov=dr_dt_ov)
            dr_dt_vec[r_idx, :] = vec_drdt

        return dr_dt_vec

    if random_amp:
        x0 = numpy.random.random(x0.size)

    for epsilon in [1e-2, 1e-4, 1e-6]:
        log.info("epsilon = %6.4e, check_grad = %6.4e" % (epsilon, scipy.optimize.check_grad(func, grad, x0, epsilon=epsilon, direction="all")))


def profile_gccsd_amp_eqs(h1e, h2e, t1e, t2e):
    import numpy, sys, line_profiler
    from pyscf.lib import logger

    from cceqs.gccsd._gccsd_amp_eqs import gccsd_ene
    from cceqs.gccsd._gccsd_amp_eqs import gccsd_r1e
    from cceqs.gccsd._gccsd_amp_eqs import gccsd_r2e

    def func(h1e, h2e, t1e, t2e):
        gccsd_ene(h1e, h2e, t1e, t2e)
        gccsd_r1e(h1e, h2e, t1e, t2e)
        gccsd_r2e(h1e, h2e, t1e, t2e)

    lp = line_profiler.LineProfiler()
    lp.add_function(gccsd_ene)
    lp.add_function(gccsd_r1e)
    lp.add_function(gccsd_r2e)
    lp_wrapper = lp(func)
    lp_wrapper(h1e, h2e, t1e, t2e)
    lp.print_stats(open("./prof_gccsd_lam_eqs.log", "w"))

def profile_gccsd_lam_eqs(h1e, h2e, t1e, t2e):
    import numpy, sys, line_profiler
    from pyscf.lib import logger

    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_rhs1e, gccsd_lam_lhs1e
    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_rhs2e, gccsd_lam_lhs2e

    nv, no = t1e.shape
    l1e_vo = t1e.copy()
    l2e_vo = t2e.copy()
    l1e_ov, l2e_ov = gccsd.transpose_vo_to_ov(no, nv, (l1e_vo, l2e_vo))

    def func(h1e, h2e, t1e, t2e):
        gccsd_lam_rhs1e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
        gccsd_lam_lhs1e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
        gccsd_lam_rhs2e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
        gccsd_lam_lhs2e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)

    lp = line_profiler.LineProfiler()
    lp.add_function(gccsd_lam_rhs1e)
    lp.add_function(gccsd_lam_lhs1e)
    lp.add_function(gccsd_lam_rhs2e)
    lp.add_function(gccsd_lam_lhs2e)
    lp_wrapper = lp(func)
    lp_wrapper(h1e, h2e, t1e, t2e)
    lp.print_stats(open("./prof_gccsd_amp_eqs.log", "w"))

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

    assert isinstance(mf, scf.ghf.GHF)
    assert mf.converged

    cc_obj   = cc.gccsd.GCCSD(mf)
    eris     = cc_obj.ao2mo()
    ene_cor_ref, t1e_ov_ref, t2e_ov_ref = cc_obj.kernel(eris=eris)
    print("Reference HF    energy = % 12.8f" % mf.energy_elec()[0])
    print("Reference CCSD  energy = % 12.8f" % ene_cor_ref)
    print("Reference total energy = % 12.8f" % (mf.energy_elec()[0] + ene_cor_ref))
    nocc, nvir = t1e_ov_ref.shape

    t1e_ref, t2e_ref = gccsd._transpose_ov_to_vo(nocc, nvir, amp_ov=(t1e_ov_ref, t2e_ov_ref))
    amp_vo_ref = (t1e_ref, t2e_ref)

    l1e_ov_ref, l2e_ov_ref = cc_obj.solve_lambda(eris=eris, t1=t1e_ov_ref, t2=t2e_ov_ref)
    l1e_ref, l2e_ref = gccsd._transpose_ov_to_vo(nocc, nvir, amp_ov=(l1e_ov_ref, l2e_ov_ref))
    lam_vo_ref = (l1e_ref, l2e_ref)

    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_rhs1e, gccsd_lam_lhs1e
    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_rhs2e, gccsd_lam_lhs2e

    h1e, h2e = get_ghf_h1e_h2e(mf)
    profile_gccsd_amp_eqs(h1e, h2e, t1e_ref, t2e_ref)
    profile_gccsd_lam_eqs(h1e, h2e, t1e_ref, t2e_ref)

    ene_tot, ene_cor, (t1e, t2e) = gccsd.solve_gccsd(
        h1e, h2e, verbose=4, 
        amp=None, max_cycle=50, tol=1e-8
        )

    print("error t1e: %6.4e" % numpy.linalg.norm(t1e - t1e_ref))
    print("error t2e: %6.4e" % numpy.linalg.norm(t2e - t2e_ref))
    print("error ene: %6.4e" % abs(ene_cor - ene_cor_ref))

    l1e, l2e = gccsd.solve_gccsd_lambda(
        h1e, h2e, lam=None,
        amp=(t1e_ref, t2e_ref),
        verbose=4, tol=1e-8,
        max_cycle=50,
        )

    print("error l1e: %6.4e" % numpy.linalg.norm(l1e - l1e_ref))
    print("error l2e: %6.4e" % numpy.linalg.norm(l2e - l2e_ref))

    test_dr_dt(h1e, h2e, t1e_ref, t2e_ref, verbose=5, random_amp=False)
    test_dr_dt(h1e, h2e, t1e_ref, t2e_ref, verbose=5, random_amp=True)
