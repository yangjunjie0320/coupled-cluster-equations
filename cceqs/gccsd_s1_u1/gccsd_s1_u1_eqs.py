
from numpy import einsum

def gccsd_s1_u1_ene(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e):
    res = 0.0
    res +=     1.000000 * einsum('I,I->'             , h1p, t1p)
    res +=     1.000000 * einsum('ia,ai->'           , h1e.ov, t1e)
    res +=     1.000000 * einsum('Iia,Iai->'         , h1e1p.ov, t1p1e)
    res +=     0.250000 * einsum('ijab,baji->'       , h2e.oovv, t2e)
    res +=     1.000000 * einsum('Iia,ai,I->'        , h1e1p.ov, t1e, t1p)
    res +=    -0.500000 * einsum('ijab,bi,aj->'      , h2e.oovv, t1e, t1e)
    return res



from numpy import einsum

def gccsd_s1_u1_r1e(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e):
    res = 0.0
    res +=     1.000000 * einsum('ai->ai'            , h1e.vo)
    res +=     1.000000 * einsum('I,Iai->ai'         , h1p, t1p1e)
    res +=    -1.000000 * einsum('ji,aj->ai'         , h1e.oo, t1e)
    res +=     1.000000 * einsum('ab,bi->ai'         , h1e.vv, t1e)
    res +=     1.000000 * einsum('Iai,I->ai'         , h1e1p.vo, t1p)
    res +=    -1.000000 * einsum('jb,abji->ai'       , h1e.ov, t2e)
    res +=    -1.000000 * einsum('Iji,Iaj->ai'       , h1e1p.oo, t1p1e)
    res +=     1.000000 * einsum('Iab,Ibi->ai'       , h1e1p.vv, t1p1e)
    res +=    -1.000000 * einsum('jaib,bj->ai'       , h2e.ovov, t1e)
    res +=     0.500000 * einsum('jkib,abkj->ai'     , h2e.ooov, t2e)
    res +=    -0.500000 * einsum('jabc,cbji->ai'     , h2e.ovvv, t2e)
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , h1e.ov, t1e, t1e)
    res +=    -1.000000 * einsum('Iji,aj,I->ai'      , h1e1p.oo, t1e, t1p)
    res +=     1.000000 * einsum('Iab,bi,I->ai'      , h1e1p.vv, t1e, t1p)
    res +=    -1.000000 * einsum('Ijb,aj,Ibi->ai'    , h1e1p.ov, t1e, t1p1e)
    res +=    -1.000000 * einsum('Ijb,bi,Iaj->ai'    , h1e1p.ov, t1e, t1p1e)
    res +=     1.000000 * einsum('Ijb,bj,Iai->ai'    , h1e1p.ov, t1e, t1p1e)
    res +=    -1.000000 * einsum('Ijb,abji,I->ai'    , h1e1p.ov, t2e, t1p)
    res +=    -1.000000 * einsum('jkib,aj,bk->ai'    , h2e.ooov, t1e, t1e)
    res +=     1.000000 * einsum('jabc,ci,bj->ai'    , h2e.ovvv, t1e, t1e)
    res +=    -0.500000 * einsum('jkbc,aj,cbki->ai'  , h2e.oovv, t1e, t2e)
    res +=    -0.500000 * einsum('jkbc,ci,abkj->ai'  , h2e.oovv, t1e, t2e)
    res +=     1.000000 * einsum('jkbc,cj,abki->ai'  , h2e.oovv, t1e, t2e)
    res +=    -1.000000 * einsum('Ijb,bi,aj,I->ai'   , h1e1p.ov, t1e, t1e, t1p)
    res +=     1.000000 * einsum('jkbc,ci,aj,bk->ai' , h2e.oovv, t1e, t1e, t1e)
    return res



from numpy import einsum

def gccsd_s1_u1_r2e(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e):
    res = 0.0
    res +=     1.000000 * einsum('baji->abij'        , h2e.vvoo)
    res +=     1.000000 * einsum('ki,bakj->abij'     , h1e.oo, t2e)
    res +=    -1.000000 * einsum('kj,baki->abij'     , h1e.oo, t2e)
    res +=     1.000000 * einsum('ac,bcji->abij'     , h1e.vv, t2e)
    res +=    -1.000000 * einsum('bc,acji->abij'     , h1e.vv, t2e)
    res +=     1.000000 * einsum('Iai,Ibj->abij'     , h1e1p.vo, t1p1e)
    res +=    -1.000000 * einsum('Iaj,Ibi->abij'     , h1e1p.vo, t1p1e)
    res +=    -1.000000 * einsum('Ibi,Iaj->abij'     , h1e1p.vo, t1p1e)
    res +=     1.000000 * einsum('Ibj,Iai->abij'     , h1e1p.vo, t1p1e)
    res +=    -1.000000 * einsum('kaji,bk->abij'     , h2e.ovoo, t1e)
    res +=     1.000000 * einsum('kbji,ak->abij'     , h2e.ovoo, t1e)
    res +=    -1.000000 * einsum('baic,cj->abij'     , h2e.vvov, t1e)
    res +=     1.000000 * einsum('bajc,ci->abij'     , h2e.vvov, t1e)
    res +=    -0.500000 * einsum('klji,balk->abij'   , h2e.oooo, t2e)
    res +=     1.000000 * einsum('kaic,bckj->abij'   , h2e.ovov, t2e)
    res +=    -1.000000 * einsum('kajc,bcki->abij'   , h2e.ovov, t2e)
    res +=    -1.000000 * einsum('kbic,ackj->abij'   , h2e.ovov, t2e)
    res +=     1.000000 * einsum('kbjc,acki->abij'   , h2e.ovov, t2e)
    res +=    -0.500000 * einsum('bacd,dcji->abij'   , h2e.vvvv, t2e)
    res +=    -1.000000 * einsum('kc,ak,bcji->abij'  , h1e.ov, t1e, t2e)
    res +=     1.000000 * einsum('kc,bk,acji->abij'  , h1e.ov, t1e, t2e)
    res +=     1.000000 * einsum('kc,ci,bakj->abij'  , h1e.ov, t1e, t2e)
    res +=    -1.000000 * einsum('kc,cj,baki->abij'  , h1e.ov, t1e, t2e)
    res +=    -1.000000 * einsum('Iki,ak,Ibj->abij'  , h1e1p.oo, t1e, t1p1e)
    res +=     1.000000 * einsum('Iki,bk,Iaj->abij'  , h1e1p.oo, t1e, t1p1e)
    res +=     1.000000 * einsum('Iki,bakj,I->abij'  , h1e1p.oo, t2e, t1p)
    res +=     1.000000 * einsum('Ikj,ak,Ibi->abij'  , h1e1p.oo, t1e, t1p1e)
    res +=    -1.000000 * einsum('Ikj,bk,Iai->abij'  , h1e1p.oo, t1e, t1p1e)
    res +=    -1.000000 * einsum('Ikj,baki,I->abij'  , h1e1p.oo, t2e, t1p)
    res +=     1.000000 * einsum('Iac,ci,Ibj->abij'  , h1e1p.vv, t1e, t1p1e)
    res +=    -1.000000 * einsum('Iac,cj,Ibi->abij'  , h1e1p.vv, t1e, t1p1e)
    res +=     1.000000 * einsum('Iac,bcji,I->abij'  , h1e1p.vv, t2e, t1p)
    res +=    -1.000000 * einsum('Ibc,ci,Iaj->abij'  , h1e1p.vv, t1e, t1p1e)
    res +=     1.000000 * einsum('Ibc,cj,Iai->abij'  , h1e1p.vv, t1e, t1p1e)
    res +=    -1.000000 * einsum('Ibc,acji,I->abij'  , h1e1p.vv, t2e, t1p)
    res +=     1.000000 * einsum('klji,bk,al->abij'  , h2e.oooo, t1e, t1e)
    res +=     1.000000 * einsum('kaic,cj,bk->abij'  , h2e.ovov, t1e, t1e)
    res +=    -1.000000 * einsum('kajc,ci,bk->abij'  , h2e.ovov, t1e, t1e)
    res +=    -1.000000 * einsum('kbic,cj,ak->abij'  , h2e.ovov, t1e, t1e)
    res +=     1.000000 * einsum('kbjc,ci,ak->abij'  , h2e.ovov, t1e, t1e)
    res +=     1.000000 * einsum('bacd,di,cj->abij'  , h2e.vvvv, t1e, t1e)
    res +=     1.000000 * einsum('Ikc,acji,Ibk->abij', h1e1p.ov, t2e, t1p1e)
    res +=    -1.000000 * einsum('Ikc,acki,Ibj->abij', h1e1p.ov, t2e, t1p1e)
    res +=     1.000000 * einsum('Ikc,ackj,Ibi->abij', h1e1p.ov, t2e, t1p1e)
    res +=    -1.000000 * einsum('Ikc,baki,Icj->abij', h1e1p.ov, t2e, t1p1e)
    res +=     1.000000 * einsum('Ikc,bakj,Ici->abij', h1e1p.ov, t2e, t1p1e)
    res +=    -1.000000 * einsum('Ikc,bcji,Iak->abij', h1e1p.ov, t2e, t1p1e)
    res +=     1.000000 * einsum('Ikc,bcki,Iaj->abij', h1e1p.ov, t2e, t1p1e)
    res +=    -1.000000 * einsum('Ikc,bckj,Iai->abij', h1e1p.ov, t2e, t1p1e)
    res +=     1.000000 * einsum('klic,ak,bclj->abij', h2e.ooov, t1e, t2e)
    res +=    -1.000000 * einsum('klic,bk,aclj->abij', h2e.ooov, t1e, t2e)
    res +=     0.500000 * einsum('klic,cj,balk->abij', h2e.ooov, t1e, t2e)
    res +=    -1.000000 * einsum('klic,ck,balj->abij', h2e.ooov, t1e, t2e)
    res +=    -1.000000 * einsum('kljc,ak,bcli->abij', h2e.ooov, t1e, t2e)
    res +=     1.000000 * einsum('kljc,bk,acli->abij', h2e.ooov, t1e, t2e)
    res +=    -0.500000 * einsum('kljc,ci,balk->abij', h2e.ooov, t1e, t2e)
    res +=     1.000000 * einsum('kljc,ck,bali->abij', h2e.ooov, t1e, t2e)
    res +=     0.500000 * einsum('kacd,bk,dcji->abij', h2e.ovvv, t1e, t2e)
    res +=    -1.000000 * einsum('kacd,di,bckj->abij', h2e.ovvv, t1e, t2e)
    res +=     1.000000 * einsum('kacd,dj,bcki->abij', h2e.ovvv, t1e, t2e)
    res +=    -1.000000 * einsum('kacd,dk,bcji->abij', h2e.ovvv, t1e, t2e)
    res +=    -0.500000 * einsum('kbcd,ak,dcji->abij', h2e.ovvv, t1e, t2e)
    res +=     1.000000 * einsum('kbcd,di,ackj->abij', h2e.ovvv, t1e, t2e)
    res +=    -1.000000 * einsum('kbcd,dj,acki->abij', h2e.ovvv, t1e, t2e)
    res +=     1.000000 * einsum('kbcd,dk,acji->abij', h2e.ovvv, t1e, t2e)
    res +=     0.500000 * einsum('klcd,adji,bclk->abij', h2e.oovv, t2e, t2e)
    res +=    -1.000000 * einsum('klcd,adki,bclj->abij', h2e.oovv, t2e, t2e)
    res +=    -0.500000 * einsum('klcd,baki,dclj->abij', h2e.oovv, t2e, t2e)
    res +=    -0.500000 * einsum('klcd,bdji,aclk->abij', h2e.oovv, t2e, t2e)
    res +=     1.000000 * einsum('klcd,bdki,aclj->abij', h2e.oovv, t2e, t2e)
    res +=     0.250000 * einsum('klcd,dcji,balk->abij', h2e.oovv, t2e, t2e)
    res +=    -0.500000 * einsum('klcd,dcki,balj->abij', h2e.oovv, t2e, t2e)
    res +=    -1.000000 * einsum('Ikc,ak,bcji,I->abij', h1e1p.ov, t1e, t2e, t1p)
    res +=     1.000000 * einsum('Ikc,bk,acji,I->abij', h1e1p.ov, t1e, t2e, t1p)
    res +=    -1.000000 * einsum('Ikc,ci,ak,Ibj->abij', h1e1p.ov, t1e, t1e, t1p1e)
    res +=     1.000000 * einsum('Ikc,ci,bk,Iaj->abij', h1e1p.ov, t1e, t1e, t1p1e)
    res +=     1.000000 * einsum('Ikc,ci,bakj,I->abij', h1e1p.ov, t1e, t2e, t1p)
    res +=     1.000000 * einsum('Ikc,cj,ak,Ibi->abij', h1e1p.ov, t1e, t1e, t1p1e)
    res +=    -1.000000 * einsum('Ikc,cj,bk,Iai->abij', h1e1p.ov, t1e, t1e, t1p1e)
    res +=    -1.000000 * einsum('Ikc,cj,baki,I->abij', h1e1p.ov, t1e, t2e, t1p)
    res +=    -1.000000 * einsum('klic,cj,bk,al->abij', h2e.ooov, t1e, t1e, t1e)
    res +=     1.000000 * einsum('kljc,ci,bk,al->abij', h2e.ooov, t1e, t1e, t1e)
    res +=    -1.000000 * einsum('kacd,di,cj,bk->abij', h2e.ovvv, t1e, t1e, t1e)
    res +=     1.000000 * einsum('kbcd,di,cj,ak->abij', h2e.ovvv, t1e, t1e, t1e)
    res +=    -1.000000 * einsum('klcd,ak,dl,bcji->abij', h2e.oovv, t1e, t1e, t2e)
    res +=    -0.500000 * einsum('klcd,bk,al,dcji->abij', h2e.oovv, t1e, t1e, t2e)
    res +=     1.000000 * einsum('klcd,bk,dl,acji->abij', h2e.oovv, t1e, t1e, t2e)
    res +=    -1.000000 * einsum('klcd,di,ak,bclj->abij', h2e.oovv, t1e, t1e, t2e)
    res +=     1.000000 * einsum('klcd,di,bk,aclj->abij', h2e.oovv, t1e, t1e, t2e)
    res +=    -0.500000 * einsum('klcd,di,cj,balk->abij', h2e.oovv, t1e, t1e, t2e)
    res +=     1.000000 * einsum('klcd,di,ck,balj->abij', h2e.oovv, t1e, t1e, t2e)
    res +=     1.000000 * einsum('klcd,dj,ak,bcli->abij', h2e.oovv, t1e, t1e, t2e)
    res +=    -1.000000 * einsum('klcd,dj,bk,acli->abij', h2e.oovv, t1e, t1e, t2e)
    res +=    -1.000000 * einsum('klcd,dj,ck,bali->abij', h2e.oovv, t1e, t1e, t2e)
    res +=     1.000000 * einsum('klcd,di,cj,bk,al->abij', h2e.oovv, t1e, t1e, t1e, t1e)
    return res



from numpy import einsum

def gccsd_s1_u1_r1p(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e):
    res = 0.0
    res +=     1.000000 * einsum('I->I'              , h1p)
    res +=     1.000000 * einsum('IJ,J->I'           , h2p, t1p)
    res +=     1.000000 * einsum('ia,Iai->I'         , h1e.ov, t1p1e)
    res +=     1.000000 * einsum('Iia,ai->I'         , h1e1p.ov, t1e)
    res +=     1.000000 * einsum('Jia,J,Iai->I'      , h1e1p.ov, t1p, t1p1e)
    res +=    -1.000000 * einsum('ijab,bi,Iaj->I'    , h2e.oovv, t1e, t1p1e)
    return res



from numpy import einsum

def gccsd_s1_u1_r1p1e(h1e, h2e, h1p, h2p, h1e1p, t1e, t2e, t1p, t1p1e):
    res = 0.0
    res +=     1.000000 * einsum('Iai->Iai'          , h1e1p.vo)
    res +=    -1.000000 * einsum('ji,Iaj->Iai'       , h1e.oo, t1p1e)
    res +=     1.000000 * einsum('ab,Ibi->Iai'       , h1e.vv, t1p1e)
    res +=     1.000000 * einsum('IJ,Jai->Iai'       , h2p, t1p1e)
    res +=    -1.000000 * einsum('Iji,aj->Iai'       , h1e1p.oo, t1e)
    res +=     1.000000 * einsum('Iab,bi->Iai'       , h1e1p.vv, t1e)
    res +=    -1.000000 * einsum('Ijb,abji->Iai'     , h1e1p.ov, t2e)
    res +=    -1.000000 * einsum('jaib,Ibj->Iai'     , h2e.ovov, t1p1e)
    res +=    -1.000000 * einsum('jb,aj,Ibi->Iai'    , h1e.ov, t1e, t1p1e)
    res +=    -1.000000 * einsum('jb,bi,Iaj->Iai'    , h1e.ov, t1e, t1p1e)
    res +=    -1.000000 * einsum('Ijb,bi,aj->Iai'    , h1e1p.ov, t1e, t1e)
    res +=    -1.000000 * einsum('Jji,J,Iaj->Iai'    , h1e1p.oo, t1p, t1p1e)
    res +=     1.000000 * einsum('Jab,J,Ibi->Iai'    , h1e1p.vv, t1p, t1p1e)
    res +=    -1.000000 * einsum('Jjb,Iaj,Jbi->Iai'  , h1e1p.ov, t1p1e, t1p1e)
    res +=    -1.000000 * einsum('Jjb,Ibi,Jaj->Iai'  , h1e1p.ov, t1p1e, t1p1e)
    res +=     1.000000 * einsum('Jjb,Ibj,Jai->Iai'  , h1e1p.ov, t1p1e, t1p1e)
    res +=    -1.000000 * einsum('jkib,aj,Ibk->Iai'  , h2e.ooov, t1e, t1p1e)
    res +=     1.000000 * einsum('jkib,bj,Iak->Iai'  , h2e.ooov, t1e, t1p1e)
    res +=     1.000000 * einsum('jabc,ci,Ibj->Iai'  , h2e.ovvv, t1e, t1p1e)
    res +=    -1.000000 * einsum('jabc,cj,Ibi->Iai'  , h2e.ovvv, t1e, t1p1e)
    res +=     1.000000 * einsum('jkbc,acji,Ibk->Iai', h2e.oovv, t2e, t1p1e)
    res +=     0.500000 * einsum('jkbc,ackj,Ibi->Iai', h2e.oovv, t2e, t1p1e)
    res +=     0.500000 * einsum('jkbc,cbji,Iak->Iai', h2e.oovv, t2e, t1p1e)
    res +=    -1.000000 * einsum('Jjb,aj,J,Ibi->Iai' , h1e1p.ov, t1e, t1p, t1p1e)
    res +=    -1.000000 * einsum('Jjb,bi,J,Iaj->Iai' , h1e1p.ov, t1e, t1p, t1p1e)
    res +=    -1.000000 * einsum('jkbc,aj,ck,Ibi->Iai', h2e.oovv, t1e, t1e, t1p1e)
    res +=     1.000000 * einsum('jkbc,ci,aj,Ibk->Iai', h2e.oovv, t1e, t1e, t1p1e)
    res +=    -1.000000 * einsum('jkbc,ci,bj,Iak->Iai', h2e.oovv, t1e, t1e, t1p1e)
    return res


