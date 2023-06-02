import os
import queue
import struct
import time
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import torch
import torch.backends.cudnn as cudnn
import timeit
import math
import psutil
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from RealSense_Utilities.realsense_api.realsense_api import mediapipe_detection
from threading import Thread
from torch.multiprocessing import Process, Pool, set_start_method, JoinableQueue, Queue, Lock
from Sampling_Socket import Receiver, Sender
from OneEuroFilter import OneEuroFilter

q = queue.Queue(maxsize=1)


def get_mid_pos(box, depth_frame, intr):
    mid_pos = [(int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2]

    dist = depth_frame.get_distance(mid_pos[0], mid_pos[1])
    pos = rs.rs2_deproject_pixel_to_point(intr, mid_pos, dist)

    return pos


def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)

    return result[0], result[1], result[2]


def homogeneous_rot(axis, angle):
    mat = np.zeros((4, 4))
    mat[3, 3] = 1
    rot = R.from_euler(axis, angle, degrees=True).as_matrix()
    mat[0:3, 0:3] = rot
    return mat


def homogeneous_trans(tx, ty, tz):
    mat = np.eye(4)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat


def make_transformation_matrix(theta_0, theta_1):
    R_0 = homogeneous_rot('z', theta_0)
    R_1 = homogeneous_rot('z', theta_1)
    # if robot arm layout rotated CCW, R_c rotate angle is positive.
    # R_c = homogeneous_rot('x', -20)
    R_c = homogeneous_rot('x', 20)
    # if robot arm layout rotated CCW, R_co euler angle is zyx(0, 0, -90)
    # R_co = homogeneous_rot('z', 180) @ homogeneous_rot('x', 90)
    R_co = homogeneous_rot('x', -90)
    T_0 = homogeneous_trans(0, 323.5, 0)
    # if robot arm layout rotated CCW, T_1 z value is negative.
    # T_1 = homogeneous_trans(0, 358.5, 99.85)
    T_1 = homogeneous_trans(0, 358.5, -99.85)
    # if robot arm layout rotated CCW, T_c x value is negative, z value is positive.
    # T_c = homogeneous_trans(35, 70.6968, -8.5806)
    T_c = homogeneous_trans(-35, 70.6968, 8.5806)

    transform_mat = R_0 @ T_0 @ R_1 @ T_1 @ T_c @ R_c @ R_co

    return transform_mat


def camera_transformation(x, y, z, rot_x, rot_y, rot_z):
    r_x = homogeneous_rot('x', rot_x)
    r_y = homogeneous_rot('y', rot_y)
    r_z = homogeneous_rot('z', rot_z)
    t = homogeneous_trans(x, y, z)

    transform_matrix = t @ r_x @ r_y @ r_z

    return transform_matrix


class InitializeYOLO:
    def __init__(self, weights_path, img_size=640, conf_threshold=0.25,
                 iou_threshold=0.45, max_det=1, classes=None, agnostic_nms=False,
                 augment=False, device_num='', line_thickness=3):
        self.weights = weights_path
        self.imgsz = img_size
        self.conf_thres = conf_threshold
        self.iou_thres = iou_threshold
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.line_thickness = line_thickness
        self.device_type = select_device(device_num)

        self.model = attempt_load(self.weights, device=self.device_type)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        cudnn.benchmark = False

        if self.device_type.type != 'cpu':
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device_type).type_as(next(self.model.parameters())))

    def detect_from_img(self, img_0, realsense_device: RealSenseCamera):
        return_pos = None
        s = np.stack([letterbox(x, self.imgsz, stride=self.stride)[0].shape for x in img_0], 0)
        rect = np.unique(s, axis=0).shape[0] == 1

        if not rect:
            print(
                'WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img_raw = img_0.copy()
        img_0 = img_0[np.newaxis, :, :, :]

        # Stack
        img_0 = np.stack(img_0, 0)

        # Convert
        img_0 = img_0[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img_0 = np.ascontiguousarray(img_0)

        img_0 = torch.from_numpy(img_0).to(self.device_type)
        img_0 = img_0.float()  # uint8 to fp16/32
        img_0 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_0.ndimension() == 3:
            img_0 = img_0.unsqueeze(0)

        # Inference
        pred = self.model(img_0, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_0.shape[2:], det[:, :4], img_raw.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    mid_pos = get_mid_pos(xyxy, realsense_device.depth_frame, realsense_device.color_intrinsics)
                    # print(mid_pos[2] * 1000)

                    return_pos = np.array(mid_pos)

                    return return_pos

        if return_pos is None:
            return None


class EarPositionCheck(Process):
    def __init__(self, task_queue, res_queue, lock, name: str):
        super(EarPositionCheck, self).__init__()
        self.name = name
        self.destroy_flag = False
        self.task_queue = task_queue
        self.res_queue = res_queue
        self.lock = lock

    def run(self):
        device = None
        cameras = {}
        transform_matrix = None
        weights_earhole = "earhole_0508s.pt"

        model = InitializeYOLO(weights_path=weights_earhole)
        realsense_device = find_realsense()

        if self.name == "rs_left":
            for serial, devices in realsense_device:
                if serial == "114222251096":
                    cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type='d455',
                                                      mp_lock=self.lock, color_stream_fps=30, depth_stream_fps=30)

                    _, device = cameras.popitem()

            transform_matrix = camera_transformation(406.65, -230, 699.65, 0, -90, 0)

        elif self.name == "rs_right":
            for serial, devices in realsense_device:
                if serial == "117122250913":
                    cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type='d455',
                                                      mp_lock=self.lock, color_stream_fps=30, depth_stream_fps=30)

                    _, device = cameras.popitem()

            transform_matrix = camera_transformation(-406.65, -230, 699.65, 0, 90, 0)

        else:
            print("cannot find required camera")
            self.destroy_flag = True

        while True:
            try:
                while not self.task_queue.empty():
                    current_task = self.task_queue.get()
                    self.task_queue.task_done()

                    if current_task:
                        pass
                    elif current_task is None:
                        self.destroy_flag = True
                        break

                    device.get_data()

                    timestamp = device.current_timestamp

                    frameset = device.frameset
                    device.get_aligned_frames(frameset, aligned_to_color=True)

                    frameset = device.depth_to_disparity.process(device.frameset)
                    frameset = device.spatial_filter.process(frameset)
                    frameset = device.temporal_filter.process(frameset)
                    frameset = device.disparity_to_depth.process(frameset)
                    frameset = device.hole_filling_filter.process(frameset).as_frameset()

                    device.frameset = frameset

                    device.color_frame = frameset.get_color_frame()
                    device.depth_frame = frameset.get_depth_frame()

                    device.color_image = frame_to_np_array(device.color_frame)

                    img_rs = np.copy(device.color_image)
                    img_raw = np.copy(img_rs)

                    earhole = model.detect_from_img(img_rs, device)

                    try:
                        if earhole is not None:
                            text = str(earhole[2] * 1000)
                            img_raw = cv2.putText(img_raw.copy(), text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

                            queued = [transform_matrix @ (earhole * 1000), timestamp, frameset]
                            self.res_queue.put(queued)
                            self.res_queue.join()

                    except Exception as e:
                        print(str(e) + self.name)

                    resized_image = cv2.resize(img_raw, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

                    # Show images from both cameras
                    cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.name, resized_image.shape[1], resized_image.shape[0])
                    cv2.imshow(self.name, resized_image)

                    cv2.waitKey(1)

                if self.destroy_flag:
                    break

            except RuntimeError as run_expt:
                print(run_expt)
                print(self.name + " can't initialize. Please turn it again.")

        try:
            cv2.destroyWindow(self.name)
        except Exception as e:
            print(e)

        device.stop()
        print(self.name + " process successfully closed")


class LandmarkMaker(Thread):
    def __init__(self, color_image, face_mesh, depth_frame, color_intrinsics):
        super(LandmarkMaker, self).__init__()
        self.color_image = color_image
        self.face_mesh = face_mesh
        self.depth_frame = depth_frame
        self.color_intrinsics = color_intrinsics
        self.ret = None

    def run(self):
        _, results = mediapipe_detection(self.color_image, self.face_mesh)
        multi_face_landmarks = results.multi_face_landmarks

        try:
            if multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                points_3d = np.zeros((0, 3))

                for i in range(468):
                    pixel_point = face_landmarks.landmark[i]

                    pixel_x = int(pixel_point.x * frame_width)
                    pixel_y = int(pixel_point.y * frame_height)

                    # if i == 4 or i == 9 or i == 200:
                    #     _ = cv2.circle(self.color_image, (pixel_x, pixel_y), 2, (0, 0, 0), -1)
                    # elif i == 6:
                    #     _ = cv2.circle(self.color_image, (pixel_x, pixel_y), 2, (0, 0, 255), -1)
                    # else:
                    #     _ = cv2.circle(self.color_image, (pixel_x, pixel_y), 1, (0, 255, 255), -1)

                    depth = self.depth_frame.get_distance(pixel_x, pixel_y)
                    if depth == 0:
                        raise ValueError

                    x, y, z = convert_depth_to_phys_coord(pixel_x, pixel_y, depth, self.color_intrinsics)

                    temporal_3d_point = np.array([x, y, z]) * 1000

                    points_3d = np.append(points_3d, temporal_3d_point[np.newaxis, :], axis=0)

                # for debug
                if q.full():
                    _ = q.get()

                q.put(points_3d.flatten())
                self.ret = True

            else:
                # for debug
                if q.full():
                    _ = q.get()

                q.put(np.zeros((0, 1)))
                self.ret = False

        except RuntimeError:
            print("thread_RuntimeError")
            self.ret = False
            pass
        except ValueError:
            print("thread_ValueError")
            self.ret = False
            pass

        finally:
            # self.depth_frame.keep()
            pass


frame_height, frame_width, channels = (480, 640, 3)


@torch.no_grad()
def main():
    left_task = JoinableQueue(maxsize=1)
    right_task = JoinableQueue(maxsize=1)
    left_res_task = JoinableQueue(maxsize=1)
    right_res_task = JoinableQueue(maxsize=1)
    rs_main = None

    collect_flag = False
    dataset = np.zeros((0, 1410))
    color_frame = np.zeros((0, 3))
    depth_frame = np.zeros((0, 3))

    lock = Lock()

    timestamps = np.zeros((0, 3))

    cameras = {}
    realsense_device = find_realsense()

    for serial, devices in realsense_device:
        if serial == "135222252454":
            cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type='d455',
                                              mp_lock=lock, color_stream_fps=30, depth_stream_fps=30)
            time.sleep(3)

    _, rs_main = cameras.popitem()

    if rs_main is None:
        print("can't initialize realsense cameras")

    left_ear_cam = EarPositionCheck(left_task, left_res_task, lock, "rs_left")
    right_ear_cam = EarPositionCheck(right_task, right_res_task, lock, "rs_right")

    left_ear_cam.start()
    time.sleep(3)
    right_ear_cam.start()
    time.sleep(3)

    mp_face_mesh_0 = mp.solutions.face_mesh

    with mp_face_mesh_0.FaceMesh(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            try:
                reset = False
                left_data = None
                right_data = None

                rs_main.get_data()
                if left_task.empty():
                    left_task.put(True)
                    # left_task.join()

                if right_task.empty():
                    right_task.put(True)
                    # right_task.join()

                frameset = rs_main.frameset
                main_timestamp = rs_main.current_timestamp

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
                img_raw = np.copy(img_rs0)
                img_depth = np.copy(np.asanyarray(rs_main.depth_frame.get_data()))

                resized_image = cv2.resize(img_raw, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

                p1 = LandmarkMaker(img_rs0, face_mesh, rs_main.depth_frame, rs_main.color_intrinsics)
                p1.start()
                p1.join()

                try:
                    while not q.full():
                        if not p1.is_alive():
                            reset = True
                            break

                    if reset:
                        continue

                    if not p1.ret:
                        _ = q.get()

                        cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                        cv2.imshow('RealSense_front', resized_image)

                        key = cv2.waitKey(1)

                        if key & 0xFF == ord('q') or key == 27:
                            left_task.put(None)
                            left_ear_cam.join()

                            right_task.put(None)
                            right_ear_cam.join()

                            cv2.destroyAllWindows()
                            break

                        continue

                except Exception as e:
                    print(e)
                    pass

                face_points = q.get()

                if not left_res_task.empty():
                    left_data = left_res_task.get()
                    left_res_task.task_done()

                if not right_res_task.empty():
                    right_data = right_res_task.get()
                    right_res_task.task_done()

                resized_image = cv2.resize(img_rs0, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

                if (left_data and right_data) is None:
                    cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                    cv2.imshow('RealSense_front', resized_image)

                    key = cv2.waitKey(1)

                    if key & 0xFF == ord('q') or key == 27:
                        left_task.put(None)
                        left_ear_cam.join()

                        right_task.put(None)
                        right_ear_cam.join()

                        cv2.destroyAllWindows()
                        break

                    continue

                timestamp_iter = np.array([main_timestamp, left_data[1], right_data[1]])
                timestamps = np.append(timestamps, timestamp_iter[np.newaxis, :], axis=0)

                if collect_flag:
                    dataset_iter = np.concatenate((face_points, left_data[0], right_data[0]), axis=1)
                    dataset = np.append(dataset, dataset_iter[np.newaxis, :], axis=0)
                    color_iter = np.array([img_raw.copy(),
                                           frame_to_np_array(left_data[2].get_color_frame()).copy(),
                                           frame_to_np_array(right_data[2].get_color_frame()).copy()])
                    color_frame = np.append(color_frame, color_iter[np.newaxis, :], axis=0)
                    depth_iter = np.array([img_depth,
                                           np.copy(np.asanyarray(left_data[2].get_depth_frame().get_data())),
                                           np.copy(np.asanyarray(right_data[2].get_depth_frame().get_data()))])
                    depth_frame = np.append(depth_frame, depth_iter[np.newaxis, :], axis=0)

                cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_front', resized_image)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    left_task.put(None)
                    left_ear_cam.join()

                    right_task.put(None)
                    right_ear_cam.join()

                    cv2.destroyAllWindows()
                    break
                elif key & 0xFF == ord('s'):
                    if not collect_flag:
                        collect_flag = True
                    else:
                        collect_flag = False
                        present_time = datetime.now()
                        save_path = ("./{}{}{}{}/".format(present_time.month, present_time.day, present_time.hour,
                                                          present_time.minute))
                        color_path = save_path + 'color_img/'
                        depth_path = save_path + 'depth_img/'
                        path_list = [save_path, color_path, depth_path]
                        os.makedirs(path_list, exist_ok=True)
                        np.savetxt(save_path + "dataset.csv", dataset, fmt='%1.5f')
                        for i in range(dataset.shape[1]):
                            cv2.imwrite(color_path + '{}_main.png'.format(i), color_frame[i, 0])
                            cv2.imwrite(color_path + '{}_left.png'.format(i), color_frame[i, 1])
                            cv2.imwrite(color_path + '{}_right.png'.format(i), color_frame[i, 2])
                            np.savetxt(depth_path + '{}_main.csv'.format(i), depth_frame[i, 0])
                            np.savetxt(depth_path + '{}_left.csv'.format(i), depth_frame[i, 1])
                            np.savetxt(depth_path + '{}_right.csv'.format(i), depth_frame[i, 2])

            except RuntimeError:
                print("frame skipped")
                continue

        rs_main.stop()
        print("main process closed")
        if timestamps.shape[0] > 1:
            np.savetxt("./timestamps.csv", timestamps, fmt='%1.5f')


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()
