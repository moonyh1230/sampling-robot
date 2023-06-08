import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def extract_param(filepath):
    rgb_intrinsic = np.genfromtxt(filepath + 'rgb.cal', skip_footer=6)
    depth_intrinsic = np.genfromtxt(filepath + 'depth.cal', skip_footer=6)

    rgb_translation = np.genfromtxt(filepath + 'rgb.cal', skip_header=9, skip_footer=1)
    rgb_rotation = np.genfromtxt(filepath + 'rgb.cal', skip_footer=2, skip_header=5)

    return rgb_intrinsic, depth_intrinsic, rgb_rotation, rgb_translation


def decode_biwi_depth(filepath):
    f = open(filepath, 'rb')
    width = int.from_bytes(f.read(4), 'little')
    height = int.from_bytes(f.read(4), 'little')
    depth = np.zeros([width * height], dtype=np.uint16)
    i = 0
    while i < width * height:
        skip = int.from_bytes(f.read(4), 'little')
        read = int.from_bytes(f.read(4), 'little')
        for j in range(read):
            depth[i + skip + j] = int.from_bytes(f.read(2), 'little')
        i += skip + read
    f.close()
    depth = depth.reshape(height, width)

    d_ = np.zeros((depth.shape[0] + 40, depth.shape[1] + 40), dtype=depth.dtype)
    d_[40:, :640] = depth
    depth = cv2.resize(d_, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)

    return depth


def main():
    dataset_list = './651952/dataset.csv'
    data_path = './651952/'
    plot_path = data_path + 'plot/'

    dataframe = pd.read_csv(dataset_list, sep=' ', header=None)
    os.makedirs(plot_path, exist_ok=True)

    data = dataframe.to_numpy()

    x_data = data[:, 0:1404] * 1000
    x_data = x_data.reshape((x_data.shape[0], 468, 3))
    y_data = data[:, 1404:1410]
    y_data = y_data.reshape((y_data.shape[0], 2, 3))

    for i in range(data.shape[0]):
        fig_debug = plt.figure(figsize=(9, 9))
        ax_debug = fig_debug.add_subplot(111, projection='3d')
        ax_debug.scatter3D(x_data[i, :, 0], x_data[i, :, 1], x_data[i, :, 2], c='black')
        ax_debug.scatter3D(y_data[i, :, 0], y_data[i, :, 1], y_data[i, :, 2], c='red')
        ax_debug.view_init(-70, -90)

        # plt.savefig(plot_path + str(i) + ".png", dpi=300)
        plt.close(fig_debug)

    print("complete")


if __name__ == '__main__':
    main()
