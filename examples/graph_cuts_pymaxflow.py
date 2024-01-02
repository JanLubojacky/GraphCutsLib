import numpy as np
import maxflow

def get_affinity(p1, p2, sigma=20):
    diff_sq = (p1 - p2)**2
    aff = np.exp(-diff_sq / (2*sigma**2))

    return round(100 * aff)

def segment_with_pymaxflow(img, mask_fg, mask_bg, sigma=20, neighborhood_sz=4):

    assert img.shape == mask_fg.shape == mask_bg.shape

    num_nodes = img.shape[0] * img.shape[1]
    if neighborhood_sz == 4:
        num_edges = 2 * num_nodes - img.shape[0] - img.shape[1]
    elif neighborhood_sz == 8:
        num_edges = 4 * num_nodes - 2 * (img.shape[0] + img.shape[1])

    G = maxflow.Graph[int](num_nodes, num_edges)
    nodeids = G.add_nodes(num_nodes)

    # reshape to the image shape
    img_node_ids = nodeids.reshape(img.shape)

    # define neighbourhoods
    neighbours = []
    neighbour_4 = [(0, -1), (1, 0)]
    neighbour_8 = [(0, -1), (-1, 0), (1, 0), (0, 1)]

    if neighborhood_sz == 4:
        neighbours = neighbour_4
    elif neighborhood_sz == 8:
        neighbours = neighbour_8
    else:
        raise ValueError("Neighborhood size must be 4 or 8")

    # add edges between pixels
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            # four-neighborhood
            for (ii, jj) in neighbours:

                ii += i 
                jj += j

                # check if the neighbor is inside the image
                if ii >= 0 and ii < img.shape[0] and jj >= 0 and jj < img.shape[1]:

                    # edges between source and sink
                    if mask_fg[i,j] == 1:
                        # infinite edge from source
                        affinity = 1_000_000_000
                        G.add_tedge(img_node_ids[i,j], affinity, 0)
                    elif mask_bg[i,j] == 1:
                        # infinite edge to sink
                        affinity = 1_000_000_000
                        G.add_tedge(img_node_ids[i,j], 0, affinity)
                    elif mask_fg[i,j] + mask_bg[i,j] == 2:
                        raise ValueError("Pixel cannot be both foreground and background")

                    affinity = get_affinity(img[i, j], img[ii, jj], sigma=sigma)
                    G.add_edge(img_node_ids[i,j], img_node_ids[ii, jj], affinity, affinity)

    # compute the maxflow
    G.maxflow()

    # get the segmentation mask
    sgm_mask = G.get_grid_segments(nodeids)
    mask = np.int_(np.logical_not(sgm_mask))
    mask = mask.reshape(img.shape)

    return mask