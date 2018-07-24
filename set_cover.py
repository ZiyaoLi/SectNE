import heapq as hq

VERBOSE = True


def set_cover(net, max_n, typ='dir', verbose=VERBOSE):
    if typ == 'dir':
        degree_heap = [(-v.out_degree, v) for v in net.vertices]
        # since our dear python only provide the smallest heap,
        # we use negative degree as index to ensure the top of the
        # heap is the max-degree vertex.
        hq.heapify(degree_heap)
        dominated_set = set()
        rst = []
        while len(rst) < max_n and len(degree_heap) > 0:
            v_top = hq.heappop(degree_heap)[1]
            if v_top.id not in dominated_set:
                dominated_set.add(v_top.id)
                rst.append(v_top.id)
                for out_id in v_top.out_prox:
                    dominated_set.add(out_id)
    elif typ == 'undir':
        degree_heap = [(-v.weight(), v) for v in net.vertices]
        # since our dear python only provide the smallest heap,
        # we use negative degree as index to ensure the top of the
        # heap is the max-degree vertex.
        hq.heapify(degree_heap)
        dominated_set = set()
        rst = []
        while len(rst) < max_n and len(degree_heap) > 0:
            v_top = hq.heappop(degree_heap)[1]
            if v_top.id not in dominated_set:
                dominated_set.add(v_top.id)
                rst.append(v_top.id)
                for out_id in v_top.out_prox:
                    dominated_set.add(out_id)
                for in_id in v_top.in_prox:
                    dominated_set.add(in_id)
    else:
        assert False, 'invalid type provided: %s' % typ
    if verbose:
        print('%d vertices selected among %d in total (%.2f). \n'
              'Covered %d / %d (%.2f) vertices in total.'
              % (len(rst), net.nVertices,
                 len(rst) / net.nVertices,
                 len(dominated_set), net.nVertices,
                 len(dominated_set) / net.nVertices))
    return rst
