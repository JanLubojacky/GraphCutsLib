import timeit

import matplotlib.pyplot as plt
# import max_flow as mf
import numpy as np
from graph_cuts_pymaxflow import segment_with_pymaxflow

import graph_cuts as gc

if __name__ == "__main__":
    # load the image
    img = plt.imread("test_imgs/lungs_ct_smaller.jpeg")
    mask_fg = np.load("test_imgs/lungs_small_fg.npy")
    mask_bg = np.load("test_imgs/lungs_small_bg.npy")

    # img = plt.imread("test_imgs/lungs_ct_larger.jpg")
    # mask_fg = np.load("test_imgs/lungs_larger_fg.npy")
    # mask_bg = np.load("test_imgs/lungs_larger_bg.npy")
    # print(img.shape)
    # print(mask_fg.shape)
    # print(mask_bg.shape)

    # convert to grayscale
    img = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 + img[:, :, 2] * 0.1140
    img = np.round(img)

    test_times = 5
    times_pymaxflow = np.zeros(test_times)
    times_my_graph_cuts = np.zeros(test_times)

    sigma = 10.0
    nsz = 4

    for i in range(test_times):
        print(f"iteration {i}")
        # start timer
        startpmx = timeit.default_timer()

        mask = segment_with_pymaxflow(img, mask_fg, mask_bg, sigma, nsz)

        # stop timer
        stoppmx = timeit.default_timer()

        times_pymaxflow[i] = stoppmx - startpmx

        # start timer
        start = timeit.default_timer()

        mask = gc.segment(img, mask_fg, mask_bg, sigma, nsz)

        # stop timer
        stop = timeit.default_timer()

        times_my_graph_cuts[i] = stop - start

    print(f"py max flow: {times_pymaxflow}")
    print(f"my graph cuts: {times_my_graph_cuts}")

    print(
        f"py max flow: {np.mean(times_pymaxflow)} ± {np.std(times_pymaxflow)}"
    )
    print(
        f"my graph cuts: {np.mean(times_my_graph_cuts)} ± {np.std(times_my_graph_cuts)}"
    )
