import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import math
import pandas as pd
import os

from datetime import datetime
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from RealSense_Utilities.realsense_api.realsense_api import mediapipe_detection

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_0 = mp.solutions.face_mesh

frame_height, frame_width, channels = (480, 640, 3)


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
    return a * x + (1 - a) * x_prev


def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)

    return result[0], result[1], result[2]


def zoom(img: np.ndarray, scale, center=None):
    height, width = img.shape[:2]
    rate = height / width

    if center is None:
        center_x = int(width / 2)
        center_y = int(height / 2)
        radius_x, radius_y = int(width / 2), int(height / 2)
    else:
        center_x, center_y = center

    if center_x < width * (1 - rate):
        center_x = width * (1 - rate)
    elif center_x > width * rate:
        center_x = width * rate

    if center_y < height * (1 - rate):
        center_y = height * (1 - rate)
    elif center_y > height * rate:
        center_y = height * rate

    center_x, center_y = int(center_x), int(center_y)
    left_x, right_x = center_x, int(width - center_x)
    up_y, down_y = int(height - center_y), center_y
    radius_x = min(left_x, right_x)
    radius_y = min(up_y, down_y)

    # Actual zoom code
    radius_x, radius_y = int(scale * radius_x), int(scale * radius_y)

    # size calculation
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y

    # Crop image to size
    cropped = img[min_y:max_y, min_x:max_x]
    # Return to original size
    # if scale >= 0:
    #     new_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
    # else:
    #     new_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)

    new_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)

    return new_cropped


class OneEuroFilter:
    def __init__(self, t0, x0: np.ndarray, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0.astype('float')

        fill_array = np.zeros(x0.shape)
        self.dx_prev = fill_array.astype('float')
        self.t_prev = float(t0)

    def __call__(self, t, x: np.ndarray) -> np.ndarray:
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def main():
    rs_main = None

    previous_timestamp = 0
    points_3d = None

    min_cutoff = 0.00001
    beta = 0.0
    first_iter = True

    zoom_scale = 1

    jitter_count = 0
    landmark_iterable = [4, 6, 9, 200]

    cameras = {}
    realsense_device = find_realsense()

    for serial, devices in realsense_device:
        if serial == '123622270472':
            cameras[serial] = RealSenseCamera(depth_stream_width=640, depth_stream_height=480,
                                              color_stream_width=640, color_stream_height=480,
                                              color_stream_fps=90, depth_stream_fps=90,
                                              device=devices, adv_mode_flag=True, device_type="d405")
        else:
            cameras[serial] = RealSenseCamera(depth_stream_width=640, depth_stream_height=480,
                                              color_stream_width=640, color_stream_height=480,
                                              color_stream_fps=30, depth_stream_fps=90,
                                              device=devices, device_type="d415", adv_mode_flag=True)

    for ser, dev in cameras.items():
        rs_main = dev

    if rs_main is None:
        print("can't initialize realsense cameras")

    with mp_face_mesh_0.FaceMesh(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh_0:

        try:
            while True:
                points_3d_iter = np.zeros((0, 3))
                rs_main.get_data()

                current_timestamp = rs_main.current_timestamp

                frameset = rs_main.frameset
                rs_main.get_aligned_frames(frameset, aligned_to_color=True)

                frameset = rs_main.depth_to_disparity.process(rs_main.frameset)
                frameset = rs_main.spatial_filter.process(frameset)
                frameset = rs_main.temporal_filter.process(frameset)
                frameset = rs_main.disparity_to_depth.process(frameset)
                frameset = rs_main.hole_filling_filter.process(frameset).as_frameset()

                rs_main.frameset = frameset

                rs_main.color_frame = frameset.get_color_frame()
                rs_main.depth_frame = frameset.get_depth_frame()

                rs_main.color_image = frame_to_np_array(rs_main.color_frame)

                img_rs0 = np.copy(rs_main.color_image)

                # img_rs0 = zoom(img_rs0.copy(), scale=zoom_scale)

                img_raw = np.copy(img_rs0)

                _, results = mediapipe_detection(img_rs0, face_mesh_0)
                multi_face_landmarks = results.multi_face_landmarks

                try:
                    if multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]

                        for i in range(468):
                            pixel_point = face_landmarks.landmark[i]
                            pixel_x = int(pixel_point.x * frame_width)
                            pixel_y = int(pixel_point.y * frame_height)
                            # _ = cv2.circle(img_rs0, (pixel_x, pixel_y), 2, (0, 0, 0), -1)

                            depth = rs_main.depth_frame.get_distance(pixel_x, pixel_y)
                            if depth == 0:
                                raise ValueError

                            x, y, z = convert_depth_to_phys_coord(
                                pixel_x,
                                pixel_y,
                                depth,
                                rs_main.color_intrinsics)

                            temporal_3d_point = np.array([x, y, z]) * 1000

                            points_3d_iter = np.append(points_3d_iter, temporal_3d_point[np.newaxis, :], axis=0)

                        points_3d_iter = points_3d_iter.reshape(1, 1404)

                        if first_iter:
                            points_3d = points_3d_iter
                            first_iter = False
                        else:
                            points_3d = np.concatenate((points_3d, points_3d_iter), axis=0)

                except ValueError:
                    print('value error')
                    continue

                except RuntimeError:
                    print('runtime error')
                    continue

                finally:
                    rs_main.depth_frame.keep()

                elapsed = current_timestamp - previous_timestamp

                # print('FPS:{} / z:{}\r'.format(1 / elapsed, points_3d_iter_hat[0, 0, 2]), end='')
                print('FPS:{}\r'.format(1 / elapsed), end='')

                previous_timestamp = current_timestamp

                resized_image = cv2.resize(img_rs0, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

                cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_front', resized_image)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

                if key & 0xFF == ord('s'):
                    # for jitter test
                    present_time = datetime.now()
                    if len(str(present_time.month)) == 1:
                        month = '0' + str(present_time.month)
                    else:
                        month = str(present_time.month)

                    if len(str(present_time.day)) == 1:
                        day = '0' + str(present_time.day)
                    else:
                        day = str(present_time.day)

                    if len(str(present_time.hour)) == 1:
                        hour = '0' + str(present_time.hour)
                    else:
                        hour = str(present_time.hour)

                    if len(str(present_time.minute)) == 1:
                        minute = '0' + str(present_time.minute)
                    else:
                        minute = str(present_time.minute)

                    os.mkdir("./jittering_test/{}/".format(month + day + hour + minute))

                    pd.DataFrame(points_3d).to_csv(
                        "./jittering_test/{}/points.csv".format(month + day + hour + minute)
                    )

                    print("test {} complete and data saved".format(jitter_count))
                    jitter_count += 1
                    first_iter = True

        finally:
            rs_main.stop()


if __name__ == '__main__':
    main()

