import keras.activations
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def make_dataset(idata, odata):
    x_list, y_list = list(), list()
    data_length = len(idata)

    for k in range(data_length):
        if k == data_length:
            break
        # x_ = idata[k:k + 1, :]
        x_ = idata[k]
        y_ = odata[k]
        # x = X[:, i:i + 1]
        # y = Y[i]

        x_list.append(x_)
        y_list.append(y_)

    return np.array(x_list), np.array(y_list)


dir_path = "./BIWI_fl_ear_dataset/"

dataset_list = os.listdir(dir_path)

x = np.zeros((0, 1404))
y = np.zeros((0, 3))
head_pose = np.zeros((0, 3))

for i in dataset_list:
    dataframe = pd.read_csv(dir_path + "/{}".format(i), sep=' ', header=None)
    data = dataframe.to_numpy()

    # fl_data = data[:, 0:1404]
    # euler = data[:, 1404:1407]
    # ear_pos = data[:, 1410:1413]

    x_data = data[:, 0:1404]
    y_data = data[:, 1410:1413]
    head_pose_data = data[:, 1404:1407]

    x = np.append(x, x_data, axis=0)
    y = np.append(y, y_data, axis=0)
    head_pose = np.append(head_pose, head_pose_data, axis=0)

# x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)

input_scaler = MinMaxScaler((-1, 1))
output_scaler = MinMaxScaler((-1, 1))

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

input_scaler.fit(xtrain)
xtrain = input_scaler.transform(xtrain)
xtest = input_scaler.transform(xtest)

xtrain = xtrain.reshape(xtrain.shape[0], 468, 3)
xtest = xtest.reshape(xtest.shape[0], 468, 3)

output_scaler.fit(ytrain)
ytrain = output_scaler.transform(ytrain)
ytest = output_scaler.transform(ytest)

ts_trainx, ts_trainy = make_dataset(xtrain, ytrain)
ts_testx, ts_testy = make_dataset(xtest, ytest)

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)

opti = Adam(learning_rate=0.0005)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation="relu", padding="same", input_shape=(468, 3)))
model.add(Conv1D(filters=32, kernel_size=2, activation="relu", padding="same"))
model.add(Conv1D(filters=32, kernel_size=2, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(filters=16, kernel_size=2, activation="relu", padding="same"))
model.add(Conv1D(filters=16, kernel_size=2, activation="relu", padding="same"))
model.add(Conv1D(filters=16, kernel_size=2, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="relu"))
model.compile(loss="mse", optimizer=opti)
model.summary()

blackbox = model.fit(ts_trainx, ts_trainy, batch_size=12, epochs=1000, verbose=2, callbacks=[es])

score_test = model.evaluate(ts_testx, ts_testy, verbose=0)
print(model.metrics_names)
print(score_test)

ypred = model.predict(ts_testx)
print("MSE: %.4f" % mean_squared_error(output_scaler.inverse_transform(ts_testy), output_scaler.inverse_transform(ypred)))

x_ax = range(len(ypred))

plt.subplot(3, 1, 1)
plt.scatter(x_ax, output_scaler.inverse_transform(ts_testy)[:, 0], s=5, color="blue", label="original_x")
plt.plot(x_ax, output_scaler.inverse_transform(ypred)[:, 0], lw=0.8, color="red", label="predicted_x")
plt.legend()

plt.subplot(3, 1, 2)
plt.scatter(x_ax, output_scaler.inverse_transform(ts_testy)[:, 1], s=5, color="blue", label="original_y")
plt.plot(x_ax, output_scaler.inverse_transform(ypred)[:, 1], lw=0.8, color="red", label="predicted_y")
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(x_ax, output_scaler.inverse_transform(ts_testy)[:, 2], s=5, color="blue", label="original_z")
plt.plot(x_ax, output_scaler.inverse_transform(ypred)[:, 2], lw=0.8, color="red", label="predicted_z")
plt.legend()

plt.show()

