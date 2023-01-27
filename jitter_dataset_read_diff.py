import numpy as np
import pandas as pd
import os


np.set_printoptions(precision=4)

var_list = np.zeros((0, 4))
std_list = np.zeros((0, 4))

dir_path = "./jittering_test/01191352"

csv_list = os.listdir(dir_path)

dl_set = np.zeros((0, 1))
dataset = pd.read_csv(dir_path + "/{}".format(csv_list[0]), header=0,
                      index_col=0)

dataset_np = dataset.to_numpy()
reshaped_dataset = dataset_np.reshape((dataset_np.shape[0], 468, 3))

x = np.zeros((468, 0))
y = np.zeros((468, 0))
z = np.zeros((468, 0))

for i in reshaped_dataset:
    x = np.concatenate((x, i[:, 0, np.newaxis]), axis=1)
    y = np.concatenate((y, i[:, 1, np.newaxis]), axis=1)
    z = np.concatenate((z, i[:, 2, np.newaxis]), axis=1)

x_var_mean = np.mean(np.var(x, axis=1))
y_var_mean = np.mean(np.var(y, axis=1))
z_var_mean = np.mean(np.var(z, axis=1))

x_std_mean = np.mean(np.std(x, axis=1))
y_std_mean = np.mean(np.std(y, axis=1))
z_std_mean = np.mean(np.std(z, axis=1))

print(x_var_mean, y_var_mean, z_var_mean)
print(x_std_mean, y_std_mean, z_std_mean)
