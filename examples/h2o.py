import numpy, scipy, sys

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
    mol = mf.mol
    assert isinstance(mf, scf.ghf.GHF)
    assert mf.converged

    orb       = mf.mo_coeff
    nao, norb = orb.shape

    orbo   = orb[:, mf.mo_occ > 0]
    orbv   = orb[:, mf.mo_occ == 0]
    nocc   = orbo.shape[1]
    nvir   = orbv.shape[1]
    occ_ix = [i for i in range(0,    nocc)]
    vir_ix = [a for a in range(nocc, norb)]

    orb_alph  = orb[:nao//2, :]
    orb_beta  = orb[nao//2:, :]

    class _FockBlocks(object):
        pass

    class _ERIBlocks(object):
        pass

    from functools import reduce
    fock_ao   = mf.get_fock()
    fock      = reduce(numpy.dot, (orb.T, fock_ao, orb))
    eris_aaaa = ao2mo.kernel(mf._eri, orb_alph)
    eris_bbbb = ao2mo.kernel(mf._eri, orb_beta)
    eris_aabb = ao2mo.kernel(mf._eri, (orb_alph, orb_alph, orb_beta, orb_beta))

    eris  = eris_aaaa + eris_bbbb
    eris += eris_aabb + eris_aabb.T

    eris = ao2mo.restore(1, eris, norb)
    eris_anti = eris.transpose(0, 2, 1, 3) - eris.transpose(0, 2, 3, 1)
    
    fock_blocks = _FockBlocks()
    fock_blocks.oo = fock[numpy.ix_(occ_ix, occ_ix)]
    fock_blocks.vv = fock[numpy.ix_(vir_ix, vir_ix)]
    fock_blocks.vo = fock[numpy.ix_(vir_ix, occ_ix)]
    fock_blocks.ov = fock[numpy.ix_(occ_ix, vir_ix)]

    eris_blocks = _ERIBlocks()
    eris_blocks.vvvv = eris_anti[numpy.ix_(vir_ix, vir_ix, vir_ix, vir_ix)]
    eris_blocks.vvvo = eris_anti[numpy.ix_(vir_ix, vir_ix, vir_ix, occ_ix)]
    eris_blocks.vvov = eris_anti[numpy.ix_(vir_ix, vir_ix, occ_ix, vir_ix)]
    eris_blocks.vovv = eris_anti[numpy.ix_(vir_ix, occ_ix, vir_ix, vir_ix)]
    eris_blocks.ovvv = eris_anti[numpy.ix_(occ_ix, vir_ix, vir_ix, vir_ix)]
    eris_blocks.vvoo = eris_anti[numpy.ix_(vir_ix, vir_ix, occ_ix, occ_ix)]
    eris_blocks.vovo = eris_anti[numpy.ix_(vir_ix, occ_ix, vir_ix, occ_ix)]
    eris_blocks.voov = eris_anti[numpy.ix_(vir_ix, occ_ix, occ_ix, vir_ix)]
    eris_blocks.ovvo = eris_anti[numpy.ix_(occ_ix, vir_ix, vir_ix, occ_ix)]
    eris_blocks.ovov = eris_anti[numpy.ix_(occ_ix, vir_ix, occ_ix, vir_ix)]
    eris_blocks.oovv = eris_anti[numpy.ix_(occ_ix, occ_ix, vir_ix, vir_ix)]
    eris_blocks.ooov = eris_anti[numpy.ix_(occ_ix, occ_ix, occ_ix, vir_ix)]
    eris_blocks.oovo = eris_anti[numpy.ix_(occ_ix, occ_ix, vir_ix, occ_ix)]
    eris_blocks.ovoo = eris_anti[numpy.ix_(occ_ix, vir_ix, occ_ix, occ_ix)]
    eris_blocks.vooo = eris_anti[numpy.ix_(vir_ix, occ_ix, occ_ix, occ_ix)]
    eris_blocks.oooo = eris_anti[numpy.ix_(occ_ix, occ_ix, occ_ix, occ_ix)]

    return fock_blocks, eris_blocks

def test_dres_dt(h1e, h2e, t1e, t2e, verbose=3, dt=1e-6):
    from cceqs.gccsd._gccsd_amp_eqs import gccsd_ene
    from cceqs.gccsd._gccsd_amp_eqs import gccsd_r1e
    from cceqs.gccsd._gccsd_amp_eqs import gccsd_r2e
    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_lhs1e
    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_lhs2e

    from pyscf.lib import logger

    log = logger.new_logger(sys.stdout, verbose)

    nv, no = t1e.shape
    vec_amp = gccsd.amp_to_vec_vo(no, nv, amp_vo=(t1e, t2e))
    dr_dt_fd = numpy.zeros((vec_amp.size, vec_amp.size))
    dr_dt_an = numpy.zeros((vec_amp.size, vec_amp.size))

    for amp_idx in range(vec_amp.size):     
        vec_fd  = numpy.zeros_like(vec_amp)
        vec_fd[amp_idx] = 1.0
        for dt in numpy.linspace(-0.1, 0.1, 41):
            t1e_1, t2e_1 = gccsd.vec_to_amp_vo(no, nv, vec_amp + dt * vec_fd)
            r1e_1 = gccsd_r1e(h1e, h2e, t1e_1, t2e_1)
            r2e_1 = gccsd_r2e(h1e, h2e, t1e_1, t2e_1)

            # t1e_2, t2e_2 = gccsd.vec_to_amp_vo(no, nv, vec_amp - dt * vec_fd)
            # r1e_2 = gccsd_r1e(h1e, h2e, t1e_2, t2e_2)
            # r2e_2 = gccsd_r2e(h1e, h2e, t1e_2, t2e_2)

            # vec_r1 = gccsd.amp_to_vec_vo(no, nv, amp_vo=(r1e_1, r2e_1))
            # vec_r2 = gccsd.amp_to_vec_vo(no, nv, amp_vo=(r1e_2, r2e_2))

            vr1e = r1e_1.reshape(-1)
            tmp_str = "% 6.4e, " % dt
            for i in range(5):
                tmp_str += "% 6.4e, " % vr1e[i]
            print(tmp_str[:-2])

            # dr1e_dt_fd = (r1e_1 - r1e_2) / (2 * dt)
            # dr2e_dt_fd = (r2e_1 - r2e_2) / (2 * dt)

            # vec_drdt_fd  = gccsd.amp_to_vec_vo(no, nv, amp_vo=(dr1e_dt_fd, dr2e_dt_fd))
            # dr_dt_fd[amp_idx, :] = vec_drdt_fd
        
        assert 1 == 2

    for res_idx in range(vec_amp.size):
        vec_lam = numpy.zeros_like(vec_amp)
        vec_lam[res_idx] = 1.0

        l1e_ov, l2e_ov = gccsd.vec_to_amp_ov(no, nv, vec_lam)
        lhs1e = gccsd_lam_lhs1e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)
        lhs2e = gccsd_lam_lhs2e(h1e, h2e, t1e, t2e, l1e_ov, l2e_ov)

        dr1e_dt_an = lhs1e
        dr2e_dt_an = lhs2e

        vec_drdt_an = gccsd.amp_to_vec_ov(no, nv, amp_ov=(dr1e_dt_an, dr2e_dt_an))
        dr_dt_an[:, res_idx] = vec_drdt_an

    log.debug("dt = %6.4e, err = %6.4e" % (dt, numpy.linalg.norm(dr_dt_fd - dr_dt_an)))
    assert numpy.linalg.norm(dr_dt_fd - dr_dt_an) < dt

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
        O  0.000000000000     0.000000000000    -0.068516648000
        H  0.000000000000    -0.790689420000     0.543766648000
        H  0.000000000000     0.790689420000     0.543766648000
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

    t1e_ref, t2e_ref = gccsd.transpose_ov_to_vo(nocc, nvir, amp_ov=(t1e_ov_ref, t2e_ov_ref))
    amp_vo_ref = (t1e_ref, t2e_ref)

    l1e_ov_ref, l2e_ov_ref = cc_obj.solve_lambda(eris=eris, t1=t1e_ov_ref, t2=t2e_ov_ref)
    l1e_ref, l2e_ref = gccsd.transpose_ov_to_vo(nocc, nvir, amp_ov=(l1e_ov_ref, l2e_ov_ref))
    lam_vo_ref = (l1e_ref, l2e_ref)

    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_rhs1e, gccsd_lam_lhs1e
    from cceqs.gccsd._gccsd_lam_eqs import gccsd_lam_rhs2e, gccsd_lam_lhs2e

    h1e, h2e = get_ghf_h1e_h2e(mf)
    profile_gccsd_amp_eqs(h1e, h2e, t1e_ref, t2e_ref)
    profile_gccsd_lam_eqs(h1e, h2e, t1e_ref, t2e_ref)

    ene_tot, ene_cor, (t1e, t2e) = gccsd.solve_gccsd(
        h1e, h2e, verbose=4, 
        amp=None, max_cycle=50, tol=1e-6
        )

    print("error t1e: %6.4e" % numpy.linalg.norm(t1e - t1e_ref))
    print("error t2e: %6.4e" % numpy.linalg.norm(t2e - t2e_ref))

    
    l1e, l2e = gccsd.solve_gccsd_lambda(
        h1e, h2e, lam=None,
        amp=(t1e_ref, t2e_ref),
        verbose=4, tol=1e-8,
        max_cycle=50,
        )

    print("error l1e: %6.4e" % numpy.linalg.norm(l1e - l1e_ref))
    print("error l2e: %6.4e" % numpy.linalg.norm(l2e - l2e_ref))

    test_dres_dt(h1e, h2e, t1e_ref, t2e_ref, dt=1e-2, verbose=5)
    test_dres_dt(h1e, h2e, t1e_ref, t2e_ref, dt=1e-4, verbose=5)
    test_dres_dt(h1e, h2e, t1e_ref, t2e_ref, dt=1e-6, verbose=5)