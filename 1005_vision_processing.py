import queue
import struct
import threading

import cv2
import os
import random
import sys
import time
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import torch
import torch.backends.cudnn as cudnn
import timeit
import math
import pandas as pd
from datetime import datetime
from pathlib import Path
from models.experimental import attempt_load
from utils.augmentations import letterbox, random_perspective
from utils.general import check_img_size, check_imshow, non_max_suppression, \
    apply_classifier, scale_coords, increment_path
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from RealSense_Utilities.realsense_api.realsense_api import mediapipe_detection
from threading import Thread
from Sampling_Socket import Receiver, Sender

q = queue.Queue(maxsize=1)

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_mid_pos(box, depth_frame, intr):
    mid_pos = [(int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2]

    dist = depth_frame.get_distance(mid_pos[0], mid_pos[1])
    pos = rs.rs2_deproject_pixel_to_point(intr, mid_pos, dist)

    return pos


def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)

    return result[0], result[1], result[2]


def get_swab_pos(stroke):
    swab_pos = np.array([0, 58.14, 332.5 + stroke])

    return swab_pos


def make_transformation_matrix(width):
    calibration_path = "transform.npz"

    with np.load(calibration_path) as cal:
        rot_mat, trans_vec = [cal[i] for i in ('rot', 'trans')]

    m_T_o_inv = np.array([[-1, 0, 0, 780], [0, -1, 0, -73.7], [0, 0, -1, width], [0, 0, 0, 1]])
    c_T_m_inv = np.zeros((4, 4))
    c_T_m_inv[0:3, 0:3] = rot_mat.T
    c_T_m_inv[0:3, 3] = -rot_mat.T @ trans_vec
    c_T_m_inv[3, 0:4] = np.array([0, 0, 0, 1])

    transform_mat = m_T_o_inv @ c_T_m_inv

    return transform_mat


class InitializeYOLO:
    def __init__(self, weights_path, img_size=640, conf_threshold=0.25,
                 iou_threshold=0.35, max_det=1, classes=None, agnostic_nms=False,
                 augment=False, half=False, device_num='', line_thickness=3):
        self.weights = weights_path
        self.imgsz = img_size
        self.conf_thres = conf_threshold
        self.iou_thres = iou_threshold
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.half = half
        self.line_thickness = line_thickness
        self.device_type = select_device(device_num)
        self.half &= self.device_type != 'cpu'

        self.model = attempt_load(self.weights, device=self.device_type)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.half:
            self.model.half()

        cudnn.benchmark = True

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
        img_0 = img_0.half() if self.half else img_0.float()  # uint8 to fp16/32
        img_0 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_0.ndimension() == 3:
            img_0 = img_0.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = self.model(img_0, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        t2 = time_sync()

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


class SwabPositionCheck(Thread):
    def __init__(self, rs_device: RealSenseCamera, yolo_model: InitializeYOLO, sender: Sender):
        Thread.__init__(self)
        self.device = rs_device
        self.model = yolo_model
        self.udp_sender = sender

    def run(self):
        swab_pos = get_swab_pos(0)
        unit_vec = np.array([0, np.sin(math.pi * 30 / 180), np.cos(math.pi * 30 / 180)])
        camera_pos = np.array([0, -86.65, 246.95])

        flag_send = False
        frame_keep = 0

        try:
            while True:
                self.device.get_data()

                frameset = self.device.frameset
                self.device.get_aligned_frames(frameset, aligned_to_color=True)

                frameset = self.device.depth_to_disparity.process(self.device.frameset)
                frameset = self.device.spatial_filter.process(frameset)
                frameset = self.device.temporal_filter.process(frameset)
                frameset = self.device.disparity_to_depth.process(frameset)
                frameset = self.device.hole_filling_filter.process(frameset).as_frameset()

                self.device.frameset = frameset

                self.device.color_frame = frameset.get_color_frame()
                self.device.depth_frame = frameset.get_depth_frame()

                self.device.color_image = frame_to_np_array(self.device.color_frame)

                img_rs = np.copy(self.device.color_image)
                img_raw = np.copy(img_rs)

                ideal_camera_z_vec = np.dot(swab_pos - camera_pos, unit_vec) * unit_vec
                ideal_camera_y_vec = (swab_pos - camera_pos) - ideal_camera_z_vec

                ideal_camera_y = np.linalg.norm(ideal_camera_y_vec)
                ideal_camera_z = np.linalg.norm(ideal_camera_z_vec)

                swabs = self.model.detect_from_img(img_rs, self.device)

                offset = np.array([[0 - swabs[1] * 1000,
                                    ideal_camera_y + swabs[0] * 1000,
                                    ideal_camera_z - swabs[2] * 1000]]
                                  )

                list_offset = offset[0].tolist()

                text = "offset x:{:.3f} y:{:.3f} z:{:.3f}".format(offset[0][0], offset[0][1], offset[0][2])

                if np.linalg.norm(offset) > 40:
                    offset_over = "Need to retry swab gripping. max offset: {:.3f} mm".format(np.max(offset))
                    img_raw = cv2.putText(img_raw.copy(), offset_over, (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.7,
                                          (0, 0, 255), 2)

                    if not flag_send and frame_keep == 200:
                        udp_send = struct.pack("iiffffff", 1, 1, 0, 0, 0, 0, 0, 0)

                        self.udp_sender.send_messages(udp_send)

                        flag_send = True

                else:
                    if not flag_send and frame_keep == 200:
                        udp_send = struct.pack("iiffffff", 0, 1, list_offset[0], list_offset[1], list_offset[2], 0, 0, 0)

                        self.udp_sender.send_messages(udp_send)

                        flag_send = True

                img_raw = cv2.putText(img_raw.copy(), text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

                resized_image = cv2.resize(img_raw, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

                # Show images from both cameras
                cv2.namedWindow('RealSense_swab', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_swab', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_swab', resized_image)

                cv2.waitKey(1)

                frame_keep += 1
                if frame_keep > 500:
                    break

        finally:
            cv2.destroyWindow('RealSense_swab')
            self.device.stop()


class LandmarkMaker(Thread):
    def __init__(self, color_image, face_mesh, depth_frame, color_intrinsics):
        Thread.__init__(self)
        self.color_image = color_image
        self.face_mesh = face_mesh
        self.depth_frame = depth_frame
        self.color_intrinsics = color_intrinsics
        self.img_circled = None
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

                    if i == 4 or i == 9 or i == 200:
                        _ = cv2.circle(self.color_image, (pixel_x, pixel_y), 2, (0, 0, 0), -1)
                    # elif i == 93:
                    #     color_image = cv2.circle(color_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
                    elif i == 6:
                        _ = cv2.circle(self.color_image, (pixel_x, pixel_y), 2, (0, 0, 255), -1)
                    # else:
                    #     color_image = cv2.circle(color_image, (pixel_x, pixel_y), 1, (0, 255, 255), -1)

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
            self.depth_frame.keep()


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_0 = mp.solutions.face_mesh

frame_height, frame_width, channels = (480, 640, 3)

UDP_vision_ip = "169.254.84.185"
UDP_main_ip = "161.122.55.149"

UDP_vision_port = 61456
UDP_main_port = 61440


@torch.no_grad()
def main():
    rs_main = None
    rs_swab = None
    weights_swab = "swab_0801.pt"
    weights_main = "ear_0829_x.pt"

    cameras = {}
    realsense_device = find_realsense()

    for serial, devices in realsense_device:
        if serial == '105322250965':
            cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True)

        elif serial == '123622270472':
            cameras[serial] = RealSenseCamera(depth_stream_width=640, depth_stream_height=480,
                                              color_stream_width=640, color_stream_height=480,
                                              color_stream_fps=30, depth_stream_fps=30,
                                              device=devices, adv_mode_flag=True, device_type="d405")

    for ser, dev in cameras.items():
        if ser == '105322250965':
            rs_main = dev

        elif ser == '123622270472':
            rs_swab = dev

    if rs_main is None or rs_swab is None:
        print("can't initialize realsense cameras")

    mean_temp = np.zeros((0, 6))
    mean_flag = False

    udp_receiver = Receiver("udp_receiver", UDP_vision_ip, UDP_vision_port)
    udp_sender = Sender("udp_sender", UDP_main_ip, UDP_main_port)

    udp_receiver.start()

    flag_path_calc = False
    flag_swab_check = False

    yolo_main = InitializeYOLO(weights_path=weights_main)
    yolo_swab = InitializeYOLO(weights_path=weights_swab)

    face_center_prev = None

    with mp_face_mesh_0.FaceMesh(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh_0:
        try:
            while True:
                if not flag_path_calc and not flag_swab_check:
                    recv_msg = udp_receiver.get_recv_data()

                    clustered_data = recv_msg.decode('utf-8')
                    recv_protocol = list(map(int, clustered_data.split('[')[1].split(']')[0].split(',')))
                    if recv_protocol is not None and recv_protocol[0] == 0:
                        flag_swab_check = True

                    elif recv_protocol is not None and recv_protocol[0] == 1:
                        flag_path_calc = True
                        robot_width = recv_protocol[1]

                reset = False

                start_time = timeit.default_timer()
                rs_main.get_data()

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

                img_raw = np.copy(img_rs0)

                p1 = LandmarkMaker(img_rs0, face_mesh_0, rs_main.depth_frame, rs_main.color_intrinsics)
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
                        terminate_time = timeit.default_timer()

                        print('FPS:{}\r'.format(1 / (terminate_time - start_time)), end='')

                        resized_image = cv2.resize(img_raw, dsize=(0, 0), fx=1, fy=1,
                                                   interpolation=cv2.INTER_AREA)

                        resized_image = cv2.rotate(resized_image.copy(), cv2.ROTATE_90_CLOCKWISE)

                        # Show images from both cameras
                        cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                        cv2.imshow('RealSense_front', resized_image)

                        key = cv2.waitKey(1)

                        if key & 0xFF == ord('q') or key == 27:
                            cv2.destroyAllWindows()
                            break

                        continue

                except Exception as e:
                    print(e)
                    pass

                face_points = q.get()

                if flag_path_calc:
                    if not mean_flag:
                        mean_flag = True

                    pos_ear = yolo_main.detect_from_img(img_rs0, rs_main)

                    if len(face_points) == 1404:
                        x_reshaped = face_points.reshape(468, 3)

                        nose_point = np.mean([x_reshaped[20], x_reshaped[60], x_reshaped[75],
                                              x_reshaped[79], x_reshaped[166], x_reshaped[238]],
                                             axis=0)

                        try:
                            if pos_ear.sum() == 0:
                                raise ValueError
                            else:
                                vec_ear = pos_ear * 1000
                                ear_pixel = rs.rs2_project_point_to_pixel(rs_main.color_intrinsics, vec_ear)

                        except ValueError as e:
                            print(e, ": successfully detecting ears but can't find ear's position")
                            continue

                        except NameError:
                            print("didn't detect ears")
                            continue

                        vec_no_9 = x_reshaped[9]
                        vec_no_200 = x_reshaped[200]
                        vec_no_4 = x_reshaped[4]

                        plane_norm_vec = np.cross((vec_no_200 - vec_no_4), (vec_no_9 - vec_no_4))
                        plane_norm_unit_vec = plane_norm_vec / np.linalg.norm(plane_norm_vec)

                        if plane_norm_unit_vec[2] <= 0:
                            img_disp = img_raw.copy()

                        else:
                            swab_vec = (nose_point - vec_ear) / np.linalg.norm(nose_point - vec_ear)

                            proj_norm_vec = np.dot(swab_vec, plane_norm_unit_vec) * plane_norm_unit_vec
                            proj_vec = (swab_vec - proj_norm_vec) / np.linalg.norm(swab_vec - proj_norm_vec)

                            swab_visualize = nose_point + proj_vec * 100

                            send_obj = np.append(nose_point, -proj_vec)

                            # print(send_obj)

                            if mean_flag and send_obj.sum != 0:
                                mean_temp = np.append(mean_temp, send_obj[np.newaxis, :], axis=0)
                                # print(mean_temp[-1])

                                if len(mean_temp) > 100:
                                    mean_obj = np.mean(mean_temp, axis=0)[np.newaxis, :]

                                    trans_vec_temp = np.append(mean_obj[:, 0:314], np.array([[1]]))[np.newaxis, :]
                                    ang_vec_temp = np.append(mean_obj[:, 3:6], np.array([[0]]))[np.newaxis, :]

                                    trans_mat = make_transformation_matrix(robot_width)

                                    trans_vec_rot = trans_mat @ trans_vec_temp.T
                                    ang_vec_rot = trans_mat @ ang_vec_temp.T

                                    udp_send_array = np.append(trans_vec_rot.T[0, 0:3], ang_vec_rot.T[0, 0:3])
                                    udp_send = struct.pack("iiffffff", 0, 2, udp_send_array[0], udp_send_array[1],
                                                           udp_send_array[2], udp_send_array[3], udp_send_array[4],
                                                           udp_send_array[5])

                                    print(udp_send)

                                    udp_sender.send_messages(udp_send)

                                    mean_flag = False
                                    mean_temp = np.zeros((0, 6))

                                    flag_path_calc = False

                            swab_point_0 = rs.rs2_project_point_to_pixel(rs_main.color_intrinsics, nose_point)
                            swab_point_1 = rs.rs2_project_point_to_pixel(rs_main.color_intrinsics, swab_visualize)

                            img_circle = cv2.circle(img_raw.copy(), list(map(int, swab_point_0)), 2, (255, 0, 0), -1)

                            img_circle = cv2.circle(img_circle.copy(), list(map(int, ear_pixel)), 2, (255, 0, 0), -1)

                            img_disp = cv2.line(img_circle.copy(), list(map(int, swab_point_0)),
                                                list(map(int, swab_point_1)),
                                                color=(0, 0, 255), thickness=2)

                        resized_image = cv2.resize(img_disp, dsize=(0, 0), fx=1, fy=1,
                                                   interpolation=cv2.INTER_AREA)

                elif flag_swab_check:
                    swab_check_thread = SwabPositionCheck(rs_swab, yolo_swab, udp_sender)

                    swab_check_thread.start()
                    swab_check_thread.join()

                    flag_swab_check = False

                rearranged_face = face_points.reshape(468, 3)
                face_center_current = np.average(rearranged_face, axis=0)

                if face_center_prev is not None:
                    offset_norm = np.linalg.norm(face_center_prev - face_center_current)

                    if offset_norm > 50:
                        print("invalid motion detected")
                        udp_sender.send_messages(struct.pack("iiffffff", 1, 0, 0, 0, 0, 0, 0, 0))

                face_center_prev = face_center_current

                terminate_time = timeit.default_timer()

                print('FPS:{}\r'.format(1 / (terminate_time - start_time)), end='')

                # for debug
                # arm_pos = np.array([550, 240, 0, 1])
                #
                # rel_arm_pos = np.linalg.inv(transform_mat) @ arm_pos[:, np.newaxis]
                # rel_arm_pixel = rs.rs2_project_point_to_pixel(rs_main.color_intrinsics, rel_arm_pos.T[:, 0:3][0])
                #
                # resized_image = cv2.circle(resized_image, list(map(int, rel_arm_pixel)), 3, (255, 255, 255), -1)
                #
                # resized_image = cv2.rotate(resized_image.copy(), cv2.ROTATE_90_CLOCKWISE)

                # Show images from both cameras
                cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_front', resized_image)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

        finally:
            rs_main.stop()
            rs_swab.stop()
            udp_sender.close_sender_socket()
            udp_receiver.close_receiver_socket()


if __name__ == '__main__':
    main()
