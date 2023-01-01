import itertools

PYTHON_FILE_TAB = "    "

ERIS_VO_6 = {
    "vvvv": ( 1, [0, 1, 2, 3]),
    "vvvo": ( 1, [0, 1, 2, 3]),
    "vvov": (-1, [0, 1, 3, 2]),
    "vvoo": ( 1, [0, 1, 2, 3]),
    "vovv": ( 1, [0, 1, 2, 3]),
    "vovo": ( 1, [0, 1, 2, 3]),
    "voov": (-1, [0, 1, 3, 2]),
    "vooo": ( 1, [0, 1, 2, 3]),
    "ovvv": (-1, [1, 0, 2, 3]),
    "ovvo": (-1, [1, 0, 2, 3]),
    "ovov": ( 1, [1, 0, 3, 2]),
    "ovoo": (-1, [1, 0, 2, 3]),
    "oovv": ( 1, [0, 1, 2, 3]),
    "oovo": ( 1, [0, 1, 2, 3]),
    "ooov": (-1, [0, 1, 3, 2]),
    "oooo": ( 1, [0, 1, 2, 3]),
}

ERIS_VO_9 = {
    "vvvv": ( 1, [0, 1, 2, 3]),
    "vvvo": (-1, [2, 3, 0, 1]),
    "vvov": ( 1, [3, 2, 0, 1]),
    "vvoo": (-1, [2, 3, 0, 1]),
    "vovv": ( 1, [0, 1, 2, 3]),
    "vovo": ( 1, [0, 1, 2, 3]),
    "voov": ( 1, [3, 2, 0, 1]),
    "vooo": (-1, [2, 3, 0, 1]),
    "ovvv": (-1, [1, 0, 2, 3]),
    "ovvo": (-1, [1, 0, 2, 3]),
    "ovov": ( 1, [1, 0, 3, 2]),
    "ovoo": ( 1, [2, 3, 1, 0]),
    "oovv": ( 1, [0, 1, 2, 3]),
    "oovo": ( 1, [0, 1, 2, 3]),
    "ooov": (-1, [0, 1, 3, 2]),
    "oooo": ( 1, [0, 1, 2, 3]),
}

def g_aint_full(eris):
    nb = eris.nb
    na = eris.na
    C = utils.block_diag(eris.ca, eris.cb)
    U = eris.umat()
    Ua = U - U.transpose((0,1,3,2))
    Ua_mo = numpy.einsum('pqrs,pw,qx,ry,sz->wxyz',Ua,C,C,C,C)
    temp = [i for i in range(2*eris.L)]
    oidx = temp[:na] + temp[eris.L:eris.L + nb] 
    vidx = temp[na:eris.L] + temp[eris.L + nb:]

    eris_blocks = {
        "vvvv": Ua_mo[numpy.ix_(vidx,vidx,vidx,vidx)],
        "vvvo": Ua_mo[numpy.ix_(vidx,vidx,vidx,oidx)],
        "vvov": Ua_mo[numpy.ix_(vidx,vidx,oidx,vidx)],
        "vovv": Ua_mo[numpy.ix_(vidx,oidx,vidx,vidx)],
        "ovvv": Ua_mo[numpy.ix_(oidx,vidx,vidx,vidx)],
        "vvoo": Ua_mo[numpy.ix_(vidx,vidx,oidx,oidx)],
        "vovo": Ua_mo[numpy.ix_(vidx,oidx,vidx,oidx)],
        "voov": Ua_mo[numpy.ix_(vidx,oidx,oidx,vidx)],
        "ovvo": Ua_mo[numpy.ix_(oidx,vidx,vidx,oidx)],
        "ovov": Ua_mo[numpy.ix_(oidx,vidx,oidx,vidx)],
        "oovv": Ua_mo[numpy.ix_(oidx,oidx,vidx,vidx)],
        "ooov": Ua_mo[numpy.ix_(oidx,oidx,oidx,vidx)],
        "oovo": Ua_mo[numpy.ix_(oidx,oidx,vidx,oidx)],
        "ovoo": Ua_mo[numpy.ix_(oidx,vidx,oidx,oidx)],
        "vooo": Ua_mo[numpy.ix_(vidx,oidx,oidx,oidx)],
        "oooo": Ua_mo[numpy.ix_(oidx,oidx,oidx,oidx)]
    }

    return two_e_blocks_full(**eris_blocks)

def space_idx_formatter(name, space_list):    
    s = ""
    for i, space in enumerate(space_list):
        if space == "nm":
            pass
        else:
            s += f"{space[0]}"

    if len(s) > 0:
        s = f"{name}." + s
    else:
        s = f"{name}" 

    return s

def einsum_str_formatter(sstr, istr, fstr, tstr):
    istr_list = istr.split(",")[:-1]
    tstr_list = tstr.split(",")[1:]

    nt = len(istr_list)
    assert nt == len(istr_list)
    assert nt == len(tstr_list)

    istr_list_new = []
    tstr_list_new = []

    for idx_t, istr_t, tstr_t in zip(range(nt), istr_list, tstr_list):
        tstr_t_split = tstr_t.split(".")
        
        if len(tstr_t_split) == 2:
            eris_idx = ERIS_VO_9.get(tstr_t_split[-1], None)
            if eris_idx is not None:
                sign_t, permu_t = eris_idx

                if sign_t == -1:
                    tstr_t_new = " -" + tstr_t_split[0].strip() + "." + "".join([tstr_t_split[1][i] for i in permu_t])
                else:
                    tstr_t_new = tstr_t_split[0] + "." + "".join([tstr_t_split[1][i] for i in permu_t])
                istr_t_new = "".join([istr_t[i] for i in permu_t])

                istr_list_new.append(istr_t_new)
                tstr_list_new.append(tstr_t_new)
            
            else:
                istr_list_new.append(istr_t)
                tstr_list_new.append(tstr_t)

        elif len(tstr_t_split) == 1:
            istr_list_new.append(istr_t)
            tstr_list_new.append(tstr_t)
        
        else:
            raise ValueError("The tstr is not valid.")

    istr = ",".join(istr_list_new) 
    tstr = "," + ",".join(tstr_list_new)
    
    einsum_str = "\'" + istr + "->" + fstr + "\'"
    return f"{float(sstr): 12.6f} * einsum({einsum_str:20s}{tstr})"

def gen_eris_vo_dict(sym=9):
    """Convert all the eris blocks to the vo format,
    where the virtual indices are always before the occupied indices. And will generate the corresponding sign caused by the permutation.

    The sym is the symmetry of the eris, which can be either 
    9 or 6, if sym == 9, then the eris will only permute p and q
    (r and s), which will lead to 3 * 3 = 9 independent blocks;
    if sym == 6, then the eris will not only permute p and q,
    r and s, and also pq and rs, which will lead to 9 - 6 / 2 = 6
    independent blocks.
    """

    idx_list = ["vv", "vo", "ov", "oo"]
    eris_vo_list = []
    eris_vo_dict = {}

    for pq_idx, pq in enumerate(idx_list):
        for rs_idx, rs in enumerate(idx_list):
            sign = 1

            mn = pq
            kl = rs

            permu_mn = [0, 1]
            permu_kl = [2, 3]

            if sym == 6:
                if pq_idx < rs_idx:
                    mn = rs
                    kl = pq
                    permu_mn = [2, 3]
                    permu_kl = [0, 1]
                    sign *= -1
            else:
                assert sym == 9

            if mn == "ov":
                mn = "vo"
                permu_mn = permu_mn[::-1]
                sign *= -1
            
            if kl == "ov":
                kl = "vo"
                permu_kl = permu_kl[::-1]
                sign *= -1

            eris_vo_old = pq + rs
            eris_vo_new = mn + kl
            permu = permu_mn + permu_kl

            eris_vo_old_permu = "".join([eris_vo_old[i] for i in permu])
            assert eris_vo_old_permu == eris_vo_new
            print(f"{eris_vo_old:4s} -> {eris_vo_new:4s} with permutation {permu} and sign {sign: 2d}.")
            
            if mn + kl not in eris_vo_list:
                eris_vo_list.append(mn + kl)

            eris_vo_dict[pq+rs] = (sign, permu)

    assert len(eris_vo_list) == sym

    return eris_vo_dict


if __name__ == "__main__":
    # test space_idx_formatter_vo_9
    eris_vo_9 = gen_eris_vo_dict(sym=9)
    eris_vo_6 = gen_eris_vo_dict(sym=6)

    print("eris_vo_9 = {")
    for key, value in eris_vo_9.items():
        print(f"    \"{key}\": ({value[0]: 2d}, {value[1]}),")
    print("}")

    print("eris_vo_6 = {")
    for key, value in eris_vo_6.items():
        print(f"    \"{key}\": ({value[0]: 2d}, {value[1]}),")
    print("}")