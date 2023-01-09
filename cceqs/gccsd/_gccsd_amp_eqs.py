
from numpy import einsum

def gccsd_ene(h1e, h2e, t1e, t2e):
    res = 0.0
    res +=     1.000000 * einsum('ia,ai->'           , h1e.ov, t1e, optimize = False)
    res +=     0.250000 * einsum('ijab,baji->'       , h2e.oovv, t2e, optimize = True)
    res +=    -0.500000 * einsum('ijab,bi,aj->'      , h2e.oovv, t1e, t1e, optimize = True)
    return res



from numpy import einsum

def gccsd_r1e(h1e, h2e, t1e, t2e):
    res = 0.0
    res +=     1.000000 * einsum('ai->ai'            , h1e.vo, optimize = False)
    res +=    -1.000000 * einsum('ji,aj->ai'         , h1e.oo, t1e, optimize = False)
    res +=     1.000000 * einsum('ab,bi->ai'         , h1e.vv, t1e, optimize = False)
    res +=    -1.000000 * einsum('jb,abji->ai'       , h1e.ov, t2e, optimize = False)
    res +=    -1.000000 * einsum('jaib,bj->ai'       , h2e.ovov, t1e, optimize = False)
    res +=     0.500000 * einsum('jkib,abkj->ai'     , h2e.ooov, t2e, optimize = True)
    res +=    -0.500000 * einsum('jabc,cbji->ai'     , h2e.ovvv, t2e, optimize = True)
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , h1e.ov, t1e, t1e, optimize = True)
    res +=    -1.000000 * einsum('jkib,aj,bk->ai'    , h2e.ooov, t1e, t1e, optimize = True)
    res +=     1.000000 * einsum('jabc,ci,bj->ai'    , h2e.ovvv, t1e, t1e, optimize = True)
    res +=    -0.500000 * einsum('jkbc,aj,cbki->ai'  , h2e.oovv, t1e, t2e, optimize = True)
    res +=    -0.500000 * einsum('jkbc,ci,abkj->ai'  , h2e.oovv, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('jkbc,cj,abki->ai'  , h2e.oovv, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('jkbc,ci,aj,bk->ai' , h2e.oovv, t1e, t1e, t1e, optimize = True)
    return res



from numpy import einsum

def gccsd_r2e(h1e, h2e, t1e, t2e):
    res = 0.0
    res +=     1.000000 * einsum('baji->abij'        , h2e.vvoo, optimize = False)
    res +=     1.000000 * einsum('ki,bakj->abij'     , h1e.oo, t2e, optimize = False)
    res +=    -1.000000 * einsum('kj,baki->abij'     , h1e.oo, t2e, optimize = False)
    res +=     1.000000 * einsum('ac,bcji->abij'     , h1e.vv, t2e, optimize = False)
    res +=    -1.000000 * einsum('bc,acji->abij'     , h1e.vv, t2e, optimize = False)
    res +=    -1.000000 * einsum('kaji,bk->abij'     , h2e.ovoo, t1e, optimize = False)
    res +=     1.000000 * einsum('kbji,ak->abij'     , h2e.ovoo, t1e, optimize = False)
    res +=    -1.000000 * einsum('baic,cj->abij'     , h2e.vvov, t1e, optimize = False)
    res +=     1.000000 * einsum('bajc,ci->abij'     , h2e.vvov, t1e, optimize = False)
    res +=    -0.500000 * einsum('klji,balk->abij'   , h2e.oooo, t2e, optimize = True)
    res +=     1.000000 * einsum('kaic,bckj->abij'   , h2e.ovov, t2e, optimize = True)
    res +=    -1.000000 * einsum('kajc,bcki->abij'   , h2e.ovov, t2e, optimize = True)
    res +=    -1.000000 * einsum('kbic,ackj->abij'   , h2e.ovov, t2e, optimize = True)
    res +=     1.000000 * einsum('kbjc,acki->abij'   , h2e.ovov, t2e, optimize = True)
    res +=    -0.500000 * einsum('bacd,dcji->abij'   , h2e.vvvv, t2e, optimize = True)
    res +=    -1.000000 * einsum('kc,ak,bcji->abij'  , h1e.ov, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('kc,bk,acji->abij'  , h1e.ov, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('kc,ci,bakj->abij'  , h1e.ov, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('kc,cj,baki->abij'  , h1e.ov, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('klji,bk,al->abij'  , h2e.oooo, t1e, t1e, optimize = True)
    res +=     1.000000 * einsum('kaic,cj,bk->abij'  , h2e.ovov, t1e, t1e, optimize = True)
    res +=    -1.000000 * einsum('kajc,ci,bk->abij'  , h2e.ovov, t1e, t1e, optimize = True)
    res +=    -1.000000 * einsum('kbic,cj,ak->abij'  , h2e.ovov, t1e, t1e, optimize = True)
    res +=     1.000000 * einsum('kbjc,ci,ak->abij'  , h2e.ovov, t1e, t1e, optimize = True)
    res +=     1.000000 * einsum('bacd,di,cj->abij'  , h2e.vvvv, t1e, t1e, optimize = True)
    res +=     1.000000 * einsum('klic,ak,bclj->abij', h2e.ooov, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('klic,bk,aclj->abij', h2e.ooov, t1e, t2e, optimize = True)
    res +=     0.500000 * einsum('klic,cj,balk->abij', h2e.ooov, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('klic,ck,balj->abij', h2e.ooov, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('kljc,ak,bcli->abij', h2e.ooov, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('kljc,bk,acli->abij', h2e.ooov, t1e, t2e, optimize = True)
    res +=    -0.500000 * einsum('kljc,ci,balk->abij', h2e.ooov, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('kljc,ck,bali->abij', h2e.ooov, t1e, t2e, optimize = True)
    res +=     0.500000 * einsum('kacd,bk,dcji->abij', h2e.ovvv, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('kacd,di,bckj->abij', h2e.ovvv, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('kacd,dj,bcki->abij', h2e.ovvv, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('kacd,dk,bcji->abij', h2e.ovvv, t1e, t2e, optimize = True)
    res +=    -0.500000 * einsum('kbcd,ak,dcji->abij', h2e.ovvv, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('kbcd,di,ackj->abij', h2e.ovvv, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('kbcd,dj,acki->abij', h2e.ovvv, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('kbcd,dk,acji->abij', h2e.ovvv, t1e, t2e, optimize = True)
    res +=     0.500000 * einsum('klcd,adji,bclk->abij', h2e.oovv, t2e, t2e, optimize = True)
    res +=    -1.000000 * einsum('klcd,adki,bclj->abij', h2e.oovv, t2e, t2e, optimize = True)
    res +=    -0.500000 * einsum('klcd,baki,dclj->abij', h2e.oovv, t2e, t2e, optimize = True)
    res +=    -0.500000 * einsum('klcd,bdji,aclk->abij', h2e.oovv, t2e, t2e, optimize = True)
    res +=     1.000000 * einsum('klcd,bdki,aclj->abij', h2e.oovv, t2e, t2e, optimize = True)
    res +=     0.250000 * einsum('klcd,dcji,balk->abij', h2e.oovv, t2e, t2e, optimize = True)
    res +=    -0.500000 * einsum('klcd,dcki,balj->abij', h2e.oovv, t2e, t2e, optimize = True)
    res +=    -1.000000 * einsum('klic,cj,bk,al->abij', h2e.ooov, t1e, t1e, t1e, optimize = True)
    res +=     1.000000 * einsum('kljc,ci,bk,al->abij', h2e.ooov, t1e, t1e, t1e, optimize = True)
    res +=    -1.000000 * einsum('kacd,di,cj,bk->abij', h2e.ovvv, t1e, t1e, t1e, optimize = True)
    res +=     1.000000 * einsum('kbcd,di,cj,ak->abij', h2e.ovvv, t1e, t1e, t1e, optimize = True)
    res +=    -1.000000 * einsum('klcd,ak,dl,bcji->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=    -0.500000 * einsum('klcd,bk,al,dcji->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('klcd,bk,dl,acji->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('klcd,di,ak,bclj->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('klcd,di,bk,aclj->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=    -0.500000 * einsum('klcd,di,cj,balk->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('klcd,di,ck,balj->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('klcd,dj,ak,bcli->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('klcd,dj,bk,acli->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=    -1.000000 * einsum('klcd,dj,ck,bali->abij', h2e.oovv, t1e, t1e, t2e, optimize = True)
    res +=     1.000000 * einsum('klcd,di,cj,bk,al->abij', h2e.oovv, t1e, t1e, t1e, t1e, optimize = True)
    return res


