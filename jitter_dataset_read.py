import numpy as np
import pandas as pd
import os


np.set_printoptions(precision=4)

var_list = np.zeros((0, 7))
std_list = np.zeros((0, 7))

dir_path = "./jittering_test/09191041"

csv_list = os.listdir(dir_path)

for i in csv_list:
    dl_set = np.zeros((0, 1))
    dataset = pd.read_csv(dir_path + "/{}".format(i), header=0, names=['counts', 'x', 'y', 'z', 'vec_x', 'vex_y', 'vec_z'],
                          index_col='counts')

    var = dataset.var()
    std = dataset.std()

    l_vec = np.array([dataset['x'].to_list(), dataset['y'].to_list(), dataset['z'].to_list()]).T

    for j in l_vec:
        dl_tmp = np.linalg.norm(j)
        dl_set = np.append(dl_set, np.array([dl_tmp])[np.newaxis, :], axis=0)

    dl_var = pd.DataFrame(dl_set).var()
    dl_std = pd.DataFrame(dl_set).std()

    var_list = np.append(var_list, np.append(var.values[np.newaxis, :], dl_var.values[np.newaxis, :], axis=1), axis=0)
    std_list = np.append(std_list, np.append(std.values[np.newaxis, :], dl_std.values[np.newaxis, :], axis=1), axis=0)

print(var_list)
print(std_list)

pd.DataFrame(var_list).to_csv(dir_path + "/var.csv")
pd.DataFrame(std_list).to_csv(dir_path + "/std.csv")
