import cv2
import os
import random
import sys
import time
import numpy as np
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


frame_height, frame_width, channels = (480, 640, 3)

w = 500
h = 700


@torch.no_grad()
def main():
    weights = 'swab_0801.pt'  # model.pt path(s)
    imgsz = 640  # inference size (pixels)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.35  # NMS IOU threshold
    max_det = 1  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    half = False  # use FP16 half-precision inference
    device_num = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    line_thickness = 3  # bounding box thickness (pixels)

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
        cameras[serial] = RealSenseCamera(depth_stream_width=640, depth_stream_height=480,
                                          color_stream_width=640, color_stream_height=480,
                                          depth_stream_fps=60,
                                          device=devices, adv_mode_flag=True, device_type="d405")

    _, device = cameras.popitem()

    swab_stroke = 138

    unit_vec = np.array([0, np.sin(math.pi * 30 / 180), np.cos(math.pi * 30 / 180)])
    camera_pos = np.array([0, -86.65, 246.95])
    swab_pos = np.array([0, 58.14, 332.5 + swab_stroke])

    mean_temp = np.zeros((0, 6))
    mean_flag = False

    try:
        while True:
            swabs = None
            start_time = timeit.default_timer()
            device.get_data()

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

            img_rs0 = np.copy(device.color_image)

            ideal_camera_z_vec = np.dot(swab_pos - camera_pos, unit_vec) * unit_vec
            ideal_camera_y_vec = (swab_pos - camera_pos) - ideal_camera_z_vec

            ideal_camera_y = np.linalg.norm(ideal_camera_y_vec)
            ideal_camera_z = np.linalg.norm(ideal_camera_z_vec)

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
            pred = model(img_rs0, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_sync()

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
                        label = f'{names[c]} {mid_pos[2] * 1000:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        swabs = np.array(mid_pos)

                        offset = np.array([[0 - swabs[1] * 1000,
                                            ideal_camera_y + swabs[0] * 1000,
                                            ideal_camera_z - swabs[2] * 1000]]
                                          )

                        text = "offset x:{:.3f} y:{:.3f} z:{:.3f}".format(offset[0][0], offset[0][1], offset[0][2])

                        if np.linalg.norm(offset) > 40:
                            offset_over = "Need to retry swab gripping. max offset: {:.3f} mm".format(np.max(offset))
                            img_raw = cv2.putText(img_raw.copy(), offset_over, (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)

                        img_raw = cv2.putText(img_raw.copy(), text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

            resized_image = cv2.resize(img_raw, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

            terminate_time = timeit.default_timer()
            if swabs is not None:
                print('FPS:{} offset x:{} y:{} z:{}\r'.format(1 / (terminate_time - start_time),
                                                              0 - swabs[1] * 1000,
                                                              ideal_camera_y + swabs[0] * 1000,
                                                              ideal_camera_z - swabs[2] * 1000), end='')

            else:
                print('FPS:{}\r'.format(1 / (terminate_time - start_time)), end='')

            # Show images from both cameras
            cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
            cv2.imshow('RealSense_front', resized_image)

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

            if key & 0xFF == ord('s'):
                if not mean_flag:
                    mean_flag = True

    finally:
        device.stop()


if __name__ == '__main__':
    main()
