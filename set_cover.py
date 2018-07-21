import heapq as hq


def set_cover(net, max_n):

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
    return rst
