
from pyscf import lib
einsum = lib.einsum

def ccsd_ene(f1e, eris, t1, t2):
    res = 0.0
    res +=     1.000000 * einsum('ia,ai->'           , f1e.ov, t1)
    res +=     0.250000 * einsum('ijab,baji->'       , eris.oovv, t2)
    res +=    -0.500000 * einsum('ijab,bi,aj->'      , eris.oovv, t1, t1)
    return res



from pyscf import lib
einsum = lib.einsum

def ccsd_res_s(f1e, eris, t1, t2):
    res = 0.0
    res +=     1.000000 * einsum('ai->ai'            , f1e.vo)
    res +=    -1.000000 * einsum('ji,aj->ai'         , f1e.oo, t1)
    res +=     1.000000 * einsum('ab,bi->ai'         , f1e.vv, t1)
    res +=    -1.000000 * einsum('jb,abji->ai'       , f1e.ov, t2)
    res +=    -1.000000 * einsum('jaib,bj->ai'       , eris.ovov, t1)
    res +=     0.500000 * einsum('jkib,abkj->ai'     , eris.ooov, t2)
    res +=    -0.500000 * einsum('jabc,cbji->ai'     , eris.ovvv, t2)
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , f1e.ov, t1, t1)
    res +=    -1.000000 * einsum('jkib,aj,bk->ai'    , eris.ooov, t1, t1)
    res +=     1.000000 * einsum('jabc,ci,bj->ai'    , eris.ovvv, t1, t1)
    res +=    -0.500000 * einsum('jkbc,aj,cbki->ai'  , eris.oovv, t1, t2)
    res +=    -0.500000 * einsum('jkbc,ci,abkj->ai'  , eris.oovv, t1, t2)
    res +=     1.000000 * einsum('jkbc,cj,abki->ai'  , eris.oovv, t1, t2)
    res +=     1.000000 * einsum('jkbc,ci,aj,bk->ai' , eris.oovv, t1, t1, t1)
    return res



from pyscf import lib
einsum = lib.einsum

def ccsd_res_d(f1e, eris, t1, t2):
    res = 0.0
    res +=     1.000000 * einsum('baji->abij'        , eris.vvoo)
    res +=     1.000000 * einsum('ki,bakj->abij'     , f1e.oo, t2)
    res +=    -1.000000 * einsum('kj,baki->abij'     , f1e.oo, t2)
    res +=     1.000000 * einsum('ac,bcji->abij'     , f1e.vv, t2)
    res +=    -1.000000 * einsum('bc,acji->abij'     , f1e.vv, t2)
    res +=    -1.000000 * einsum('kaji,bk->abij'     , eris.ovoo, t1)
    res +=     1.000000 * einsum('kbji,ak->abij'     , eris.ovoo, t1)
    res +=    -1.000000 * einsum('baic,cj->abij'     , eris.vvov, t1)
    res +=     1.000000 * einsum('bajc,ci->abij'     , eris.vvov, t1)
    res +=    -0.500000 * einsum('klji,balk->abij'   , eris.oooo, t2)
    res +=     1.000000 * einsum('kaic,bckj->abij'   , eris.ovov, t2)
    res +=    -1.000000 * einsum('kajc,bcki->abij'   , eris.ovov, t2)
    res +=    -1.000000 * einsum('kbic,ackj->abij'   , eris.ovov, t2)
    res +=     1.000000 * einsum('kbjc,acki->abij'   , eris.ovov, t2)
    res +=    -0.500000 * einsum('bacd,dcji->abij'   , eris.vvvv, t2)
    res +=    -1.000000 * einsum('kc,ak,bcji->abij'  , f1e.ov, t1, t2)
    res +=     1.000000 * einsum('kc,bk,acji->abij'  , f1e.ov, t1, t2)
    res +=     1.000000 * einsum('kc,ci,bakj->abij'  , f1e.ov, t1, t2)
    res +=    -1.000000 * einsum('kc,cj,baki->abij'  , f1e.ov, t1, t2)
    res +=     1.000000 * einsum('klji,bk,al->abij'  , eris.oooo, t1, t1)
    res +=     1.000000 * einsum('kaic,cj,bk->abij'  , eris.ovov, t1, t1)
    res +=    -1.000000 * einsum('kajc,ci,bk->abij'  , eris.ovov, t1, t1)
    res +=    -1.000000 * einsum('kbic,cj,ak->abij'  , eris.ovov, t1, t1)
    res +=     1.000000 * einsum('kbjc,ci,ak->abij'  , eris.ovov, t1, t1)
    res +=     1.000000 * einsum('bacd,di,cj->abij'  , eris.vvvv, t1, t1)
    res +=     1.000000 * einsum('klic,ak,bclj->abij', eris.ooov, t1, t2)
    res +=    -1.000000 * einsum('klic,bk,aclj->abij', eris.ooov, t1, t2)
    res +=     0.500000 * einsum('klic,cj,balk->abij', eris.ooov, t1, t2)
    res +=    -1.000000 * einsum('klic,ck,balj->abij', eris.ooov, t1, t2)
    res +=    -1.000000 * einsum('kljc,ak,bcli->abij', eris.ooov, t1, t2)
    res +=     1.000000 * einsum('kljc,bk,acli->abij', eris.ooov, t1, t2)
    res +=    -0.500000 * einsum('kljc,ci,balk->abij', eris.ooov, t1, t2)
    res +=     1.000000 * einsum('kljc,ck,bali->abij', eris.ooov, t1, t2)
    res +=     0.500000 * einsum('kacd,bk,dcji->abij', eris.ovvv, t1, t2)
    res +=    -1.000000 * einsum('kacd,di,bckj->abij', eris.ovvv, t1, t2)
    res +=     1.000000 * einsum('kacd,dj,bcki->abij', eris.ovvv, t1, t2)
    res +=    -1.000000 * einsum('kacd,dk,bcji->abij', eris.ovvv, t1, t2)
    res +=    -0.500000 * einsum('kbcd,ak,dcji->abij', eris.ovvv, t1, t2)
    res +=     1.000000 * einsum('kbcd,di,ackj->abij', eris.ovvv, t1, t2)
    res +=    -1.000000 * einsum('kbcd,dj,acki->abij', eris.ovvv, t1, t2)
    res +=     1.000000 * einsum('kbcd,dk,acji->abij', eris.ovvv, t1, t2)
    res +=     0.500000 * einsum('klcd,adji,bclk->abij', eris.oovv, t2, t2)
    res +=    -1.000000 * einsum('klcd,adki,bclj->abij', eris.oovv, t2, t2)
    res +=    -0.500000 * einsum('klcd,baki,dclj->abij', eris.oovv, t2, t2)
    res +=    -0.500000 * einsum('klcd,bdji,aclk->abij', eris.oovv, t2, t2)
    res +=     1.000000 * einsum('klcd,bdki,aclj->abij', eris.oovv, t2, t2)
    res +=     0.250000 * einsum('klcd,dcji,balk->abij', eris.oovv, t2, t2)
    res +=    -0.500000 * einsum('klcd,dcki,balj->abij', eris.oovv, t2, t2)
    res +=    -1.000000 * einsum('klic,cj,bk,al->abij', eris.ooov, t1, t1, t1)
    res +=     1.000000 * einsum('kljc,ci,bk,al->abij', eris.ooov, t1, t1, t1)
    res +=    -1.000000 * einsum('kacd,di,cj,bk->abij', eris.ovvv, t1, t1, t1)
    res +=     1.000000 * einsum('kbcd,di,cj,ak->abij', eris.ovvv, t1, t1, t1)
    res +=     1.000000 * einsum('klcd,di,cj,bk,al->abij', eris.oovv, t1, t1, t1, t1)
    return res



from pyscf import lib
einsum = lib.einsum

def ccsd_lambda_rhs_s(f1e, eris, t1, t2):
    res = 0.0
    res +=     1.000000 * einsum('ia->ia'            , f1e.ov)
    res +=    -1.000000 * einsum('ijab,bi->ja'       , eris.oovv, t1)
    return res



from pyscf import lib
einsum = lib.einsum

def ccsd_lambda_lhs_s(f1e, eris, t1, t2, l1, l2):
    res = 0.0
    res +=    -1.000000 * einsum('ij,ja->ia'         , f1e.oo, l1)
    res +=     1.000000 * einsum('ba,ib->ia'         , f1e.vv, l1)
    res +=    -1.000000 * einsum('ibja,jb->ia'       , eris.ovov, l1)
    res +=     0.500000 * einsum('ibjk,kjab->ia'     , eris.ovoo, l2)
    res +=    -0.500000 * einsum('bcja,jicb->ia'     , eris.vvov, l2)
    res +=    -1.000000 * einsum('ja,bj,ib->ia'      , f1e.ov, t1, l1)
    res +=    -1.000000 * einsum('ib,bj,ja->ia'      , f1e.ov, t1, l1)
    res +=     1.000000 * einsum('jikb,bj,ka->ia'    , eris.ooov, t1, l1)
    res +=    -1.000000 * einsum('jika,bj,kb->ia'    , eris.ooov, t1, l1)
    res +=    -1.000000 * einsum('jbac,cj,ib->ia'    , eris.ovvv, t1, l1)
    res +=     1.000000 * einsum('ibac,cj,jb->ia'    , eris.ovvv, t1, l1)
    res +=    -0.500000 * einsum('ja,bcjk,kicb->ia'  , f1e.ov, t2, l2)
    res +=    -0.500000 * einsum('ib,cbjk,kjac->ia'  , f1e.ov, t2, l2)
    res +=     0.500000 * einsum('jkab,cbkj,ic->ia'  , eris.oovv, t2, l1)
    res +=     0.500000 * einsum('jibc,cbjk,ka->ia'  , eris.oovv, t2, l1)
    res +=     1.000000 * einsum('jiab,cbjk,kc->ia'  , eris.oovv, t2, l1)
    res +=    -1.000000 * einsum('jbka,cj,kicb->ia'  , eris.ovov, t1, l2)
    res +=    -1.000000 * einsum('ibjc,ck,jkab->ia'  , eris.ovov, t1, l2)
    res +=     0.500000 * einsum('jikl,bj,lkab->ia'  , eris.oooo, t1, l2)
    res +=     0.500000 * einsum('bcad,dj,jicb->ia'  , eris.vvvv, t1, l2)
    res +=     0.250000 * einsum('jkla,bckj,licb->ia', eris.ooov, t2, l2)
    res +=    -1.000000 * einsum('jikb,cbjl,klac->ia', eris.ooov, t2, l2)
    res +=     0.500000 * einsum('jika,bcjl,klcb->ia', eris.ooov, t2, l2)
    res +=     1.000000 * einsum('jbac,dcjk,kidb->ia', eris.ovvv, t2, l2)
    res +=    -0.250000 * einsum('ibcd,dcjk,kjab->ia', eris.ovvv, t2, l2)
    res +=    -0.500000 * einsum('ibac,dcjk,kjdb->ia', eris.ovvv, t2, l2)
    res +=    -1.000000 * einsum('jkab,cj,bk,ic->ia' , eris.oovv, t1, t1, l1)
    res +=    -1.000000 * einsum('jibc,ck,bj,ka->ia' , eris.oovv, t1, t1, l1)
    res +=     1.000000 * einsum('jiab,bk,cj,kc->ia' , eris.oovv, t1, t1, l1)
    res +=    -0.500000 * einsum('jkla,bj,ck,licb->ia', eris.ooov, t1, t1, l2)
    res +=    -1.000000 * einsum('jikb,bl,cj,klac->ia', eris.ooov, t1, t1, l2)
    res +=     1.000000 * einsum('jbac,ck,dj,kidb->ia', eris.ovvv, t1, t1, l2)
    res +=     0.500000 * einsum('ibcd,dj,ck,jkab->ia', eris.ovvv, t1, t1, l2)
    res +=     1.000000 * einsum('jkab,cj,dbkl,lidc->ia', eris.oovv, t1, t2, l2)
    res +=    -0.250000 * einsum('jkab,bl,cdkj,lidc->ia', eris.oovv, t1, t2, l2)
    res +=     0.500000 * einsum('jkab,bj,cdkl,lidc->ia', eris.oovv, t1, t2, l2)
    res +=    -0.250000 * einsum('jibc,dj,cbkl,lkad->ia', eris.oovv, t1, t2, l2)
    res +=     1.000000 * einsum('jibc,ck,dbjl,klad->ia', eris.oovv, t1, t2, l2)
    res +=     0.500000 * einsum('jibc,cj,dbkl,lkad->ia', eris.oovv, t1, t2, l2)
    res +=    -0.500000 * einsum('jiab,cj,dbkl,lkdc->ia', eris.oovv, t1, t2, l2)
    res +=    -0.500000 * einsum('jiab,bk,cdjl,kldc->ia', eris.oovv, t1, t2, l2)
    res +=     0.500000 * einsum('jkab,bl,cj,dk,lidc->ia', eris.oovv, t1, t1, t1, l2)
    res +=     0.500000 * einsum('jibc,ck,bl,dj,klad->ia', eris.oovv, t1, t1, t1, l2)
    return res



from pyscf import lib
einsum = lib.einsum

def ccsd_lambda_rhs_d(f1e, eris, t1, t2):
    res = 0.0
    res +=     1.000000 * einsum('ijab->jiba'        , eris.oovv)
    return res



from pyscf import lib
einsum = lib.einsum

def ccsd_lambda_lhs_d(f1e, eris, t1, t2, l1, l2):
    res = 0.0
    res +=     1.000000 * einsum('ik,kjba->ijab'     , f1e.oo, l2)
    res +=    -1.000000 * einsum('jk,kiba->ijab'     , f1e.oo, l2)
    res +=     1.000000 * einsum('ca,jibc->ijab'     , f1e.vv, l2)
    res +=    -1.000000 * einsum('cb,jiac->ijab'     , f1e.vv, l2)
    res +=    -1.000000 * einsum('icba,jc->ijab'     , eris.ovvv, l1)
    res +=    -1.000000 * einsum('jika,kb->ijab'     , eris.ooov, l1)
    res +=     1.000000 * einsum('jikb,ka->ijab'     , eris.ooov, l1)
    res +=     1.000000 * einsum('jcba,ic->ijab'     , eris.ovvv, l1)
    res +=     1.000000 * einsum('icka,kjbc->ijab'   , eris.ovov, l2)
    res +=    -1.000000 * einsum('ickb,kjac->ijab'   , eris.ovov, l2)
    res +=    -1.000000 * einsum('jcka,kibc->ijab'   , eris.ovov, l2)
    res +=     1.000000 * einsum('jckb,kiac->ijab'   , eris.ovov, l2)
    res +=    -0.500000 * einsum('jikl,lkba->ijab'   , eris.oooo, l2)
    res +=    -0.500000 * einsum('cdba,jidc->ijab'   , eris.vvvv, l2)
    res +=    -1.000000 * einsum('ka,ck,jibc->ijab'  , f1e.ov, t1, l2)
    res +=     1.000000 * einsum('kb,ck,jiac->ijab'  , f1e.ov, t1, l2)
    res +=     1.000000 * einsum('ic,ck,kjba->ijab'  , f1e.ov, t1, l2)
    res +=    -1.000000 * einsum('jc,ck,kiba->ijab'  , f1e.ov, t1, l2)
    res +=    -1.000000 * einsum('kiba,ck,jc->ijab'  , eris.oovv, t1, l1)
    res +=     1.000000 * einsum('kjba,ck,ic->ijab'  , eris.oovv, t1, l1)
    res +=     1.000000 * einsum('jiac,ck,kb->ijab'  , eris.oovv, t1, l1)
    res +=    -1.000000 * einsum('jibc,ck,ka->ijab'  , eris.oovv, t1, l1)
    res +=    -1.000000 * einsum('kilc,ck,ljba->ijab', eris.ooov, t1, l2)
    res +=     1.000000 * einsum('kila,ck,ljbc->ijab', eris.ooov, t1, l2)
    res +=    -1.000000 * einsum('kilb,ck,ljac->ijab', eris.ooov, t1, l2)
    res +=     1.000000 * einsum('kjlc,ck,liba->ijab', eris.ooov, t1, l2)
    res +=    -1.000000 * einsum('kjla,ck,libc->ijab', eris.ooov, t1, l2)
    res +=     1.000000 * einsum('kjlb,ck,liac->ijab', eris.ooov, t1, l2)
    res +=    -1.000000 * einsum('kcad,dk,jibc->ijab', eris.ovvv, t1, l2)
    res +=     1.000000 * einsum('kcbd,dk,jiac->ijab', eris.ovvv, t1, l2)
    res +=    -1.000000 * einsum('kcba,dk,jidc->ijab', eris.ovvv, t1, l2)
    res +=    -1.000000 * einsum('icad,dk,kjbc->ijab', eris.ovvv, t1, l2)
    res +=     1.000000 * einsum('icbd,dk,kjac->ijab', eris.ovvv, t1, l2)
    res +=     1.000000 * einsum('jikc,cl,klba->ijab', eris.ooov, t1, l2)
    res +=     1.000000 * einsum('jcad,dk,kibc->ijab', eris.ovvv, t1, l2)
    res +=    -1.000000 * einsum('jcbd,dk,kiac->ijab', eris.ovvv, t1, l2)
    res +=     0.500000 * einsum('klac,dclk,jibd->ijab', eris.oovv, t2, l2)
    res +=    -0.500000 * einsum('klbc,dclk,jiad->ijab', eris.oovv, t2, l2)
    res +=     0.250000 * einsum('klba,cdlk,jidc->ijab', eris.oovv, t2, l2)
    res +=    -0.500000 * einsum('kicd,dckl,ljba->ijab', eris.oovv, t2, l2)
    res +=    -1.000000 * einsum('kiac,dckl,ljbd->ijab', eris.oovv, t2, l2)
    res +=     1.000000 * einsum('kibc,dckl,ljad->ijab', eris.oovv, t2, l2)
    res +=    -0.500000 * einsum('kiba,cdkl,ljdc->ijab', eris.oovv, t2, l2)
    res +=     0.500000 * einsum('kjcd,dckl,liba->ijab', eris.oovv, t2, l2)
    res +=     1.000000 * einsum('kjac,dckl,libd->ijab', eris.oovv, t2, l2)
    res +=    -1.000000 * einsum('kjbc,dckl,liad->ijab', eris.oovv, t2, l2)
    res +=     0.500000 * einsum('kjba,cdkl,lidc->ijab', eris.oovv, t2, l2)
    res +=     0.250000 * einsum('jicd,dckl,lkba->ijab', eris.oovv, t2, l2)
    res +=     0.500000 * einsum('jiac,dckl,lkbd->ijab', eris.oovv, t2, l2)
    res +=    -0.500000 * einsum('jibc,dckl,lkad->ijab', eris.oovv, t2, l2)
    res +=    -1.000000 * einsum('klac,dk,cl,jibd->ijab', eris.oovv, t1, t1, l2)
    res +=     1.000000 * einsum('klbc,dk,cl,jiad->ijab', eris.oovv, t1, t1, l2)
    res +=    -0.500000 * einsum('klba,ck,dl,jidc->ijab', eris.oovv, t1, t1, l2)
    res +=     1.000000 * einsum('kicd,dl,ck,ljba->ijab', eris.oovv, t1, t1, l2)
    res +=    -1.000000 * einsum('kiac,cl,dk,ljbd->ijab', eris.oovv, t1, t1, l2)
    res +=     1.000000 * einsum('kibc,cl,dk,ljad->ijab', eris.oovv, t1, t1, l2)
    res +=    -1.000000 * einsum('kjcd,dl,ck,liba->ijab', eris.oovv, t1, t1, l2)
    res +=     1.000000 * einsum('kjac,cl,dk,libd->ijab', eris.oovv, t1, t1, l2)
    res +=    -1.000000 * einsum('kjbc,cl,dk,liad->ijab', eris.oovv, t1, t1, l2)
    res +=    -0.500000 * einsum('jicd,dk,cl,klba->ijab', eris.oovv, t1, t1, l2)
    return res


