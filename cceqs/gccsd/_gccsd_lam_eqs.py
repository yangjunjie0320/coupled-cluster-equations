
from numpy import einsum

def gccsd_lam_rhs1e(h1e, h2e, t1e, t2e, l1e, l2e):
    res = 0.0
    res +=     1.000000 * einsum('ia->ia'            , h1e.ov, optimize = False)
    res +=    -1.000000 * einsum('ijab,bi->ja'       , h2e.oovv, t1e, optimize = False)
    return res



from numpy import einsum

def gccsd_lam_lhs1e(h1e, h2e, t1e, t2e, l1e, l2e):
    res = 0.0
    res +=    -1.000000 * einsum('ij,ja->ia'         , h1e.oo, l1e, optimize = False)
    res +=     1.000000 * einsum('ba,ib->ia'         , h1e.vv, l1e, optimize = False)
    res +=    -1.000000 * einsum('ibja,jb->ia'       , h2e.ovov, l1e, optimize = False)
    res +=     0.500000 * einsum('ibjk,kjab->ia'     , h2e.ovoo, l2e, optimize = True)
    res +=    -0.500000 * einsum('bcja,jicb->ia'     , h2e.vvov, l2e, optimize = True)
    res +=    -1.000000 * einsum('ja,bj,ib->ia'      , h1e.ov, t1e, l1e, optimize = True)
    res +=    -1.000000 * einsum('ib,bj,ja->ia'      , h1e.ov, t1e, l1e, optimize = True)
    res +=     1.000000 * einsum('jikb,bj,ka->ia'    , h2e.ooov, t1e, l1e, optimize = True)
    res +=    -1.000000 * einsum('jika,bj,kb->ia'    , h2e.ooov, t1e, l1e, optimize = True)
    res +=    -1.000000 * einsum('jbac,cj,ib->ia'    , h2e.ovvv, t1e, l1e, optimize = True)
    res +=     1.000000 * einsum('ibac,cj,jb->ia'    , h2e.ovvv, t1e, l1e, optimize = True)
    res +=    -0.500000 * einsum('ja,bcjk,kicb->ia'  , h1e.ov, t2e, l2e, optimize = True)
    res +=    -0.500000 * einsum('ib,cbjk,kjac->ia'  , h1e.ov, t2e, l2e, optimize = True)
    res +=     0.500000 * einsum('jkab,cbkj,ic->ia'  , h2e.oovv, t2e, l1e, optimize = True)
    res +=     0.500000 * einsum('jibc,cbjk,ka->ia'  , h2e.oovv, t2e, l1e, optimize = True)
    res +=     1.000000 * einsum('jiab,cbjk,kc->ia'  , h2e.oovv, t2e, l1e, optimize = True)
    res +=    -1.000000 * einsum('jbka,cj,kicb->ia'  , h2e.ovov, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('ibjc,ck,jkab->ia'  , h2e.ovov, t1e, l2e, optimize = True)
    res +=     0.500000 * einsum('jikl,bj,lkab->ia'  , h2e.oooo, t1e, l2e, optimize = True)
    res +=     0.500000 * einsum('bcad,dj,jicb->ia'  , h2e.vvvv, t1e, l2e, optimize = True)
    res +=     0.250000 * einsum('jkla,bckj,licb->ia', h2e.ooov, t2e, l2e, optimize = True)
    res +=    -1.000000 * einsum('jikb,cbjl,klac->ia', h2e.ooov, t2e, l2e, optimize = True)
    res +=     0.500000 * einsum('jika,bcjl,klcb->ia', h2e.ooov, t2e, l2e, optimize = True)
    res +=     1.000000 * einsum('jbac,dcjk,kidb->ia', h2e.ovvv, t2e, l2e, optimize = True)
    res +=    -0.250000 * einsum('ibcd,dcjk,kjab->ia', h2e.ovvv, t2e, l2e, optimize = True)
    res +=    -0.500000 * einsum('ibac,dcjk,kjdb->ia', h2e.ovvv, t2e, l2e, optimize = True)
    res +=    -1.000000 * einsum('jkab,cj,bk,ic->ia' , h2e.oovv, t1e, t1e, l1e, optimize = True)
    res +=    -1.000000 * einsum('jibc,ck,bj,ka->ia' , h2e.oovv, t1e, t1e, l1e, optimize = True)
    res +=     1.000000 * einsum('jiab,bk,cj,kc->ia' , h2e.oovv, t1e, t1e, l1e, optimize = True)
    res +=    -0.500000 * einsum('jkla,bj,ck,licb->ia', h2e.ooov, t1e, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('jikb,bl,cj,klac->ia', h2e.ooov, t1e, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('jbac,ck,dj,kidb->ia', h2e.ovvv, t1e, t1e, l2e, optimize = True)
    res +=     0.500000 * einsum('ibcd,dj,ck,jkab->ia', h2e.ovvv, t1e, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('jkab,cj,dbkl,lidc->ia', h2e.oovv, t1e, t2e, l2e, optimize = True)
    res +=    -0.250000 * einsum('jkab,bl,cdkj,lidc->ia', h2e.oovv, t1e, t2e, l2e, optimize = True)
    res +=     0.500000 * einsum('jkab,bj,cdkl,lidc->ia', h2e.oovv, t1e, t2e, l2e, optimize = True)
    res +=    -0.250000 * einsum('jibc,dj,cbkl,lkad->ia', h2e.oovv, t1e, t2e, l2e, optimize = True)
    res +=     1.000000 * einsum('jibc,ck,dbjl,klad->ia', h2e.oovv, t1e, t2e, l2e, optimize = True)
    res +=     0.500000 * einsum('jibc,cj,dbkl,lkad->ia', h2e.oovv, t1e, t2e, l2e, optimize = True)
    res +=    -0.500000 * einsum('jiab,cj,dbkl,lkdc->ia', h2e.oovv, t1e, t2e, l2e, optimize = True)
    res +=    -0.500000 * einsum('jiab,bk,cdjl,kldc->ia', h2e.oovv, t1e, t2e, l2e, optimize = True)
    res +=     0.500000 * einsum('jkab,bl,cj,dk,lidc->ia', h2e.oovv, t1e, t1e, t1e, l2e, optimize = True)
    res +=     0.500000 * einsum('jibc,ck,bl,dj,klad->ia', h2e.oovv, t1e, t1e, t1e, l2e, optimize = True)
    return res



from numpy import einsum

def gccsd_lam_rhs2e(h1e, h2e, t1e, t2e, l1e, l2e):
    res = 0.0
    res +=     1.000000 * einsum('jiba->ijab'        , h2e.oovv, optimize = False)
    return res



from numpy import einsum

def gccsd_lam_lhs2e(h1e, h2e, t1e, t2e, l1e, l2e):
    res = 0.0
    res +=     1.000000 * einsum('ia,jb->ijab'       , l1e, h1e.ov, optimize = False)
    res +=    -1.000000 * einsum('ib,ja->ijab'       , l1e, h1e.ov, optimize = False)
    res +=    -1.000000 * einsum('ja,ib->ijab'       , l1e, h1e.ov, optimize = False)
    res +=     1.000000 * einsum('jb,ia->ijab'       , l1e, h1e.ov, optimize = False)
    res +=     1.000000 * einsum('ik,kjba->ijab'     , h1e.oo, l2e, optimize = False)
    res +=    -1.000000 * einsum('jk,kiba->ijab'     , h1e.oo, l2e, optimize = False)
    res +=     1.000000 * einsum('ca,jibc->ijab'     , h1e.vv, l2e, optimize = False)
    res +=    -1.000000 * einsum('cb,jiac->ijab'     , h1e.vv, l2e, optimize = False)
    res +=    -1.000000 * einsum('icba,jc->ijab'     , h2e.ovvv, l1e, optimize = False)
    res +=    -1.000000 * einsum('jika,kb->ijab'     , h2e.ooov, l1e, optimize = False)
    res +=     1.000000 * einsum('jikb,ka->ijab'     , h2e.ooov, l1e, optimize = False)
    res +=     1.000000 * einsum('jcba,ic->ijab'     , h2e.ovvv, l1e, optimize = False)
    res +=     1.000000 * einsum('icka,kjbc->ijab'   , h2e.ovov, l2e, optimize = True)
    res +=    -1.000000 * einsum('ickb,kjac->ijab'   , h2e.ovov, l2e, optimize = True)
    res +=    -1.000000 * einsum('jcka,kibc->ijab'   , h2e.ovov, l2e, optimize = True)
    res +=     1.000000 * einsum('jckb,kiac->ijab'   , h2e.ovov, l2e, optimize = True)
    res +=    -0.500000 * einsum('jikl,lkba->ijab'   , h2e.oooo, l2e, optimize = True)
    res +=    -0.500000 * einsum('cdba,jidc->ijab'   , h2e.vvvv, l2e, optimize = True)
    res +=    -1.000000 * einsum('ka,ck,jibc->ijab'  , h1e.ov, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('kb,ck,jiac->ijab'  , h1e.ov, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('ic,ck,kjba->ijab'  , h1e.ov, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('jc,ck,kiba->ijab'  , h1e.ov, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kiba,ck,jc->ijab'  , h2e.oovv, t1e, l1e, optimize = True)
    res +=     1.000000 * einsum('kjba,ck,ic->ijab'  , h2e.oovv, t1e, l1e, optimize = True)
    res +=     1.000000 * einsum('jiac,ck,kb->ijab'  , h2e.oovv, t1e, l1e, optimize = True)
    res +=    -1.000000 * einsum('jibc,ck,ka->ijab'  , h2e.oovv, t1e, l1e, optimize = True)
    res +=    -1.000000 * einsum('ck,ia,kjbc->ijab'  , t1e, l1e, h2e.oovv, optimize = True)
    res +=     1.000000 * einsum('ck,ib,kjac->ijab'  , t1e, l1e, h2e.oovv, optimize = True)
    res +=     1.000000 * einsum('ck,ja,kibc->ijab'  , t1e, l1e, h2e.oovv, optimize = True)
    res +=    -1.000000 * einsum('ck,jb,kiac->ijab'  , t1e, l1e, h2e.oovv, optimize = True)
    res +=    -1.000000 * einsum('kilc,ck,ljba->ijab', h2e.ooov, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('kila,ck,ljbc->ijab', h2e.ooov, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kilb,ck,ljac->ijab', h2e.ooov, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('kjlc,ck,liba->ijab', h2e.ooov, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kjla,ck,libc->ijab', h2e.ooov, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('kjlb,ck,liac->ijab', h2e.ooov, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kcad,dk,jibc->ijab', h2e.ovvv, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('kcbd,dk,jiac->ijab', h2e.ovvv, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kcba,dk,jidc->ijab', h2e.ovvv, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('icad,dk,kjbc->ijab', h2e.ovvv, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('icbd,dk,kjac->ijab', h2e.ovvv, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('jikc,cl,klba->ijab', h2e.ooov, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('jcad,dk,kibc->ijab', h2e.ovvv, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('jcbd,dk,kiac->ijab', h2e.ovvv, t1e, l2e, optimize = True)
    res +=     0.500000 * einsum('klac,dclk,jibd->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=    -0.500000 * einsum('klbc,dclk,jiad->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=     0.250000 * einsum('klba,cdlk,jidc->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=    -0.500000 * einsum('kicd,dckl,ljba->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kiac,dckl,ljbd->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=     1.000000 * einsum('kibc,dckl,ljad->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=    -0.500000 * einsum('kiba,cdkl,ljdc->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=     0.500000 * einsum('kjcd,dckl,liba->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=     1.000000 * einsum('kjac,dckl,libd->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kjbc,dckl,liad->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=     0.500000 * einsum('kjba,cdkl,lidc->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=     0.250000 * einsum('jicd,dckl,lkba->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=     0.500000 * einsum('jiac,dckl,lkbd->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=    -0.500000 * einsum('jibc,dckl,lkad->ijab', h2e.oovv, t2e, l2e, optimize = True)
    res +=    -1.000000 * einsum('klac,dk,cl,jibd->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('klbc,dk,cl,jiad->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=    -0.500000 * einsum('klba,ck,dl,jidc->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('kicd,dl,ck,ljba->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kiac,cl,dk,ljbd->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('kibc,cl,dk,ljad->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kjcd,dl,ck,liba->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=     1.000000 * einsum('kjac,cl,dk,libd->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=    -1.000000 * einsum('kjbc,cl,dk,liad->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    res +=    -0.500000 * einsum('jicd,dk,cl,klba->ijab', h2e.oovv, t1e, t1e, l2e, optimize = True)
    return res


