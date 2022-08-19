import queue
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

q = queue.Queue(maxsize=1)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_mid_pos(box, depth_frame, intr):
    mid_pos = [(int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2]

    dist = depth_frame.get_distance(mid_pos[0], mid_pos[1])
    pos = rs.rs2_deproject_pixel_to_point(intr, mid_pos, dist)

    return pos


def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)

    return result[0], result[1], result[2]


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

                    if i == 4 or i == 9 or i == 152:
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

                q.put(points_3d.flatten())
                self.ret = True

            else:
                q.put(np.zeros((0, 1)))
                self.ret = False

        except RuntimeError:
            self.ret = False
            pass
        except ValueError:
            self.ret = False
            pass


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_0 = mp.solutions.face_mesh

frame_height, frame_width, channels = (480, 640, 3)

transform_pointset = np.empty((0, 3))

w = 500
h = 700

calibration_path = "transform.npz"

with np.load(calibration_path) as cal:
    rot_mat, trans_vec = [cal[i] for i in ('rot', 'trans')]

m_T_o_inv = np.array([[-1, 0, 0, 780], [0, -1, 0, h - 946.3], [0, 0, -1, w], [0, 0, 0, 1]])
c_T_m_inv = np.zeros((4, 4))
c_T_m_inv[0:3, 0:3] = rot_mat.T
c_T_m_inv[0:3, 3] = -rot_mat.T @ trans_vec
c_T_m_inv[3, 0:4] = np.array([0, 0, 0, 1])

transform_mat = m_T_o_inv @ c_T_m_inv


@torch.no_grad()
def run():
    weights = 'ear_0805.pt'  # model.pt path(s)
    imgsz = 640  # inference size (pixels)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.35  # NMS IOU threshold
    max_det = 1  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    stride = 32
    device_num = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = False  # show results
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    update = False  # update all models
    name = 'exp'  # save results to project/name

    # Initialize
    device_type = select_device(device_num)
    half &= device_type.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, device=device_type)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device_type)['model']).to(device_type).eval()

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Run inference
    if device_type.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device_type).type_as(next(model.parameters())))  # run once

    cameras = {}
    realsense_device = find_realsense()

    for serial, devices in realsense_device:
        cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True)

    _, device = cameras.popitem()

    with mp_face_mesh_0.FaceMesh(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh_0:
        try:
            while True:
                global w
                reset = False
                send_obj = np.zeros((0, 6))

                start_time = timeit.default_timer()
                device.get_data()

                frameset = device.frameset
                device.get_aligned_frames(frameset, aligned_to_color=True)

                frameset = device.depth_to_disparity.process(device.frameset).as_frameset()
                frameset = device.spatial_filter.process(frameset).as_frameset()
                frameset = device.temporal_filter.process(frameset).as_frameset()
                frameset = device.disparity_to_depth.process(frameset).as_frameset()
                frameset = device.hole_filling_filter.process(frameset).as_frameset()

                device.frameset = frameset

                device.color_frame = frameset.get_color_frame()
                device.depth_frame = frameset.get_depth_frame()

                device.color_image = frame_to_np_array(device.color_frame)

                img_rs0 = np.copy(device.color_image)

                p1 = LandmarkMaker(img_rs0, face_mesh_0, device.depth_frame, device.color_intrinsics)
                p1.start()
                p1.join()

                s = np.stack([letterbox(x, imgsz, stride=stride)[0].shape for x in img_rs0], 0)  # shapes
                rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
                if not rect:
                    print(
                        'WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

                # Letterbox
                img_raw = img_rs0.copy()
                img_rs0 = img_rs0[np.newaxis, :, :, :]

                # Stack
                img_rs0 = np.stack(img_rs0, 0)

                # Convert
                img_rs0 = img_rs0[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
                img_rs0 = np.ascontiguousarray(img_rs0)

                img_rs0 = torch.from_numpy(img_rs0).to(device_type)
                img_rs0 = img_rs0.half() if half else img_rs0.float()  # uint8 to fp16/32
                img_rs0 /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img_rs0.ndimension() == 3:
                    img_rs0 = img_rs0.unsqueeze(0)

                # Inference
                t1 = time_sync()
                pred = model(img_rs0, augment=augment,
                             visualize=increment_path(save_dir / 'features', mkdir=True) if visualize else False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                t2 = time_sync()

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img_rs0, img_raw)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    s = f'{i}: '
                    s += '%gx%g ' % img_rs0.shape[2:]  # print string
                    annotator = Annotator(img_raw, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img_rs0.shape[2:], det[:, :4], img_raw.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for *xyxy, conf, cls in reversed(det):
                            mid_pos = get_mid_pos(xyxy, device.depth_frame, device.color_intrinsics)
                            # print(mid_pos[2] * 1000)
                            c = int(cls)  # integer class
                            # label = None if hide_labels else (
                            #     names[c] if hide_conf else f'{names[c]} {mid_pos[2] * 1000:.2f}')
                            # annotator.box_label(xyxy, label, color=colors(c, True))

                            ears = np.array(mid_pos)

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

                        # Show images from both cameras
                        cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                        cv2.imshow('RealSense_front', resized_image)

                        key = cv2.waitKey(1)

                        if key & 0xFF == ord('q') or key == 27:
                            cv2.destroyAllWindows()
                            break

                        if key & 0xFF == ord('s'):
                            present_time = datetime.now()
                            img_name = 'mesh_calc_{}{}{}{}.png'.format(present_time.month, present_time.day,
                                                                       present_time.hour,
                                                                       present_time.minute)
                            cv2.imwrite(img_name, resized_image)
                            print("Save")

                        continue

                except Exception as e:
                    print(e)
                    pass

                face_points = q.get()

                if len(face_points) == 1404:
                    x_reshaped = face_points.reshape(468, 3)

                    nose_point = np.mean([x_reshaped[20], x_reshaped[60], x_reshaped[75],
                                          x_reshaped[79], x_reshaped[166], x_reshaped[238]],
                                         axis=0)

                    vec_no_9 = x_reshaped[9]
                    vec_no_152 = x_reshaped[152]
                    try:
                        if ears.sum() == 0:
                            raise ValueError
                        else:
                            vec_ear = ears

                    except ValueError as e:
                        print(e, ": successfully detecting ears but can't find ear's position")
                        continue

                    except NameError:
                        print("didn't detect ears")
                        continue

                    vec_no_4 = x_reshaped[4]

                    plane_norm_vec = np.cross((vec_no_152 - vec_no_4), (vec_no_9 - vec_no_4))
                    plane_norm_unit_vec = plane_norm_vec / np.linalg.norm(plane_norm_vec)

                    if plane_norm_unit_vec[2] <= 0:
                        img_disp = img_raw.copy()

                    else:
                        swab_vec = (nose_point - vec_ear) / np.linalg.norm(nose_point - vec_ear)

                        proj_norm_vec = np.dot(swab_vec, plane_norm_unit_vec) * plane_norm_unit_vec
                        proj_vec = (swab_vec - proj_norm_vec) / np.linalg.norm(swab_vec - proj_norm_vec)

                        swab_visualize = nose_point + proj_vec * 100

                        send_obj[:, 0:3] = nose_point
                        send_obj[:, 3:6] = proj_vec

                        swab_point_0 = rs.rs2_project_point_to_pixel(device.color_intrinsics, nose_point)
                        swab_point_1 = rs.rs2_project_point_to_pixel(device.color_intrinsics, swab_visualize)

                        img_circle = cv2.circle(img_raw.copy(), list(map(int, swab_point_0)), 2, (255, 0, 0), -1)

                        img_disp = cv2.line(img_circle.copy(), list(map(int, swab_point_0)),
                                            list(map(int, swab_point_1)),
                                            color=(0, 0, 255), thickness=2)

                    resized_image = cv2.resize(img_disp, dsize=(0, 0), fx=1, fy=1,
                                               interpolation=cv2.INTER_AREA)

                terminate_time = timeit.default_timer()

                print('FPS:{}\r'.format(1 / (terminate_time - start_time)), end='')

                # for debug
                arm_pos = np.array([550, 240, 0, 1])

                rel_arm_pos = np.linalg.inv(transform_mat) @ arm_pos[:, np.newaxis]
                rel_arm_pixel = rs.rs2_project_point_to_pixel(device.color_intrinsics, rel_arm_pos)

                resized_image = cv2.circle(resized_image, list(map(int, rel_arm_pixel)), 3, (255, 255, 255), -1)

                # Show images from both cameras
                cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_front', resized_image)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

                if key & 0xFF == ord('s'):
                    present_time = datetime.now()
                    img_name = 'mesh_calc_{}{}{}{}.png'.format(present_time.month, present_time.day, present_time.hour,
                                                               present_time.minute)
                    cv2.imwrite(img_name, resized_image)
                    print("Save")

        finally:
            device.stop()


if __name__ == '__main__':
    run()
