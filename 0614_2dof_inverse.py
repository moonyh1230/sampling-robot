import numpy as np
from math import atan2, sqrt, sin, cos, degrees, acos
from matplotlib import pyplot as plt


def inverse(x, y):
    l_1 = 323.5
    l_2 = 66.5 + 350.13 + 168

    try:
        cos_2 = (x**2 + y**2 - l_1**2 - l_2**2) / (2 * l_1 * l_2)
        sin_2 = [sqrt(1 - cos_2**2), -sqrt(1 - cos_2**2)]

        theta_2 = [atan2(sin_2[0], cos_2), atan2(sin_2[1], cos_2)]

        b = acos((x**2 + y**2 + l_1**2 - l_2**2) / (2 * l_1 * sqrt(x**2 + y**2)))

        theta_1 = [atan2(y, x) - b, atan2(y, x) + b]

        return theta_1, theta_2

    except Exception as e:
        return None


if __name__ == "__main__":
    y1 = 900

    # while True:
    #     x1 = input("x = ")
    #     x1 = float(x1)
    #
    #     y1 = input("y = ")
    #     y1 = float(y1)
    #
    #     try:
    #         t1, t2 = inverse(x1, y1)
    #
    #         print(degrees(t1[0]), degrees(t2[0]))
    #         print(degrees(t1[1]), degrees(t2[1]))
    #
    #         # print(323.5 * cos(t1[0]) + 518 * cos(t1[0] + t2[0]))
    #         # print(323.5 * sin(t1[0]) + 518 * sin(t1[0] + t2[0]))
    #
    #     except:
    #         print("singular position")
    #         continue

    for x1 in range(20):
        try:
            t1, t2 = inverse(270 - x1 * 25, y1)

            print("x = {}".format(270 - x1 * 25))
            print(degrees(t1[0]), degrees(t2[0]))
            print(degrees(t1[1]), degrees(t2[1]))
            print("")

        except:
            print("x = {} is singular position".format(270 - x1 * 25))
            continue
