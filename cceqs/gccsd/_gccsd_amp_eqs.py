
from numpy import einsum

def gccsd_ene(h1e, h2e, t1e, t2e):
    res = 0.0
    res +=     1.000000 * einsum('ia,ai->'           , h1e.ov, t1e)
    res +=     0.250000 * einsum('ijab,baji->'       , h2e.oovv, t2e)
    res +=    -0.500000 * einsum('ijab,bi,aj->'      , h2e.oovv, t1e, t1e)
    return res



from numpy import einsum

def gccsd_r1e(h1e, h2e, t1e, t2e):
    res = 0.0
    res +=     1.000000 * einsum('ai->ai'            , h1e.vo)
    res +=    -1.000000 * einsum('ji,aj->ai'         , h1e.oo, t1e)
    res +=     1.000000 * einsum('ab,bi->ai'         , h1e.vv, t1e)
    res +=    -1.000000 * einsum('jb,abji->ai'       , h1e.ov, t2e)
    res +=    -1.000000 * einsum('ajbi,bj->ai'       , h2e.vovo, t1e)
    res +=     0.500000 * einsum('jkbi,abkj->ai'     , -h2e.oovo, t2e)
    res +=    -0.500000 * einsum('ajbc,cbji->ai'     , -h2e.vovv, t2e)
    res +=    -1.000000 * einsum('jb,bi,aj->ai'      , h1e.ov, t1e, t1e)
    res +=    -1.000000 * einsum('jkbi,aj,bk->ai'    , -h2e.oovo, t1e, t1e)
    res +=     1.000000 * einsum('ajbc,ci,bj->ai'    , -h2e.vovv, t1e, t1e)
    res +=    -0.500000 * einsum('jkbc,aj,cbki->ai'  , h2e.oovv, t1e, t2e)
    res +=    -0.500000 * einsum('jkbc,ci,abkj->ai'  , h2e.oovv, t1e, t2e)
    res +=     1.000000 * einsum('jkbc,cj,abki->ai'  , h2e.oovv, t1e, t2e)
    res +=     1.000000 * einsum('jkbc,ci,aj,bk->ai' , h2e.oovv, t1e, t1e, t1e)
    return res



from numpy import einsum

def gccsd_r2e(h1e, h2e, t1e, t2e):
    res = 0.0
    res +=     1.000000 * einsum('jiba->abij'        , -h2e.oovv)
    res +=     1.000000 * einsum('ki,bakj->abij'     , h1e.oo, t2e)
    res +=    -1.000000 * einsum('kj,baki->abij'     , h1e.oo, t2e)
    res +=     1.000000 * einsum('ac,bcji->abij'     , h1e.vv, t2e)
    res +=    -1.000000 * einsum('bc,acji->abij'     , h1e.vv, t2e)
    res +=    -1.000000 * einsum('jiak,bk->abij'     , h2e.oovo, t1e)
    res +=     1.000000 * einsum('jibk,ak->abij'     , h2e.oovo, t1e)
    res +=    -1.000000 * einsum('ciba,cj->abij'     , h2e.vovv, t1e)
    res +=     1.000000 * einsum('cjba,ci->abij'     , h2e.vovv, t1e)
    res +=    -0.500000 * einsum('klji,balk->abij'   , h2e.oooo, t2e)
    res +=     1.000000 * einsum('akci,bckj->abij'   , h2e.vovo, t2e)
    res +=    -1.000000 * einsum('akcj,bcki->abij'   , h2e.vovo, t2e)
    res +=    -1.000000 * einsum('bkci,ackj->abij'   , h2e.vovo, t2e)
    res +=     1.000000 * einsum('bkcj,acki->abij'   , h2e.vovo, t2e)
    res +=    -0.500000 * einsum('bacd,dcji->abij'   , h2e.vvvv, t2e)
    res +=    -1.000000 * einsum('kc,ak,bcji->abij'  , h1e.ov, t1e, t2e)
    res +=     1.000000 * einsum('kc,bk,acji->abij'  , h1e.ov, t1e, t2e)
    res +=     1.000000 * einsum('kc,ci,bakj->abij'  , h1e.ov, t1e, t2e)
    res +=    -1.000000 * einsum('kc,cj,baki->abij'  , h1e.ov, t1e, t2e)
    res +=     1.000000 * einsum('klji,bk,al->abij'  , h2e.oooo, t1e, t1e)
    res +=     1.000000 * einsum('akci,cj,bk->abij'  , h2e.vovo, t1e, t1e)
    res +=    -1.000000 * einsum('akcj,ci,bk->abij'  , h2e.vovo, t1e, t1e)
    res +=    -1.000000 * einsum('bkci,cj,ak->abij'  , h2e.vovo, t1e, t1e)
    res +=     1.000000 * einsum('bkcj,ci,ak->abij'  , h2e.vovo, t1e, t1e)
    res +=     1.000000 * einsum('bacd,di,cj->abij'  , h2e.vvvv, t1e, t1e)
    res +=     1.000000 * einsum('klci,ak,bclj->abij', -h2e.oovo, t1e, t2e)
    res +=    -1.000000 * einsum('klci,bk,aclj->abij', -h2e.oovo, t1e, t2e)
    res +=     0.500000 * einsum('klci,cj,balk->abij', -h2e.oovo, t1e, t2e)
    res +=    -1.000000 * einsum('klci,ck,balj->abij', -h2e.oovo, t1e, t2e)
    res +=    -1.000000 * einsum('klcj,ak,bcli->abij', -h2e.oovo, t1e, t2e)
    res +=     1.000000 * einsum('klcj,bk,acli->abij', -h2e.oovo, t1e, t2e)
    res +=    -0.500000 * einsum('klcj,ci,balk->abij', -h2e.oovo, t1e, t2e)
    res +=     1.000000 * einsum('klcj,ck,bali->abij', -h2e.oovo, t1e, t2e)
    res +=     0.500000 * einsum('akcd,bk,dcji->abij', -h2e.vovv, t1e, t2e)
    res +=    -1.000000 * einsum('akcd,di,bckj->abij', -h2e.vovv, t1e, t2e)
    res +=     1.000000 * einsum('akcd,dj,bcki->abij', -h2e.vovv, t1e, t2e)
    res +=    -1.000000 * einsum('akcd,dk,bcji->abij', -h2e.vovv, t1e, t2e)
    res +=    -0.500000 * einsum('bkcd,ak,dcji->abij', -h2e.vovv, t1e, t2e)
    res +=     1.000000 * einsum('bkcd,di,ackj->abij', -h2e.vovv, t1e, t2e)
    res +=    -1.000000 * einsum('bkcd,dj,acki->abij', -h2e.vovv, t1e, t2e)
    res +=     1.000000 * einsum('bkcd,dk,acji->abij', -h2e.vovv, t1e, t2e)
    res +=     0.500000 * einsum('klcd,adji,bclk->abij', h2e.oovv, t2e, t2e)
    res +=    -1.000000 * einsum('klcd,adki,bclj->abij', h2e.oovv, t2e, t2e)
    res +=    -0.500000 * einsum('klcd,baki,dclj->abij', h2e.oovv, t2e, t2e)
    res +=    -0.500000 * einsum('klcd,bdji,aclk->abij', h2e.oovv, t2e, t2e)
    res +=     1.000000 * einsum('klcd,bdki,aclj->abij', h2e.oovv, t2e, t2e)
    res +=     0.250000 * einsum('klcd,dcji,balk->abij', h2e.oovv, t2e, t2e)
    res +=    -0.500000 * einsum('klcd,dcki,balj->abij', h2e.oovv, t2e, t2e)
    res +=    -1.000000 * einsum('klci,cj,bk,al->abij', -h2e.oovo, t1e, t1e, t1e)
    res +=     1.000000 * einsum('klcj,ci,bk,al->abij', -h2e.oovo, t1e, t1e, t1e)
    res +=    -1.000000 * einsum('akcd,di,cj,bk->abij', -h2e.vovv, t1e, t1e, t1e)
    res +=     1.000000 * einsum('bkcd,di,cj,ak->abij', -h2e.vovv, t1e, t1e, t1e)
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


