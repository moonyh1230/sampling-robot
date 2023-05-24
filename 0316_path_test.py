import queue
import struct
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
from Sampling_Socket import Receiver, Sender
from OneEuroFilter import OneEuroFilter

q = queue.Queue(maxsize=1)

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def memory_usage(message: str = 'debug'):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")


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


# def make_transformation_matrix(width):
#     calibration_path = "transform.npz"
#
#     with np.load(calibration_path) as cal:
#         rot_mat, trans_vec = [cal[i] for i in ('rot', 'trans')]
#
#     m_T_o_inv = np.array([[-1, 0, 0, 780], [0, -1, 0, -73.7], [0, 0, -1, width], [0, 0, 0, 1]])
#     c_T_m_inv = np.zeros((4, 4))
#     c_T_m_inv[0:3, 0:3] = rot_mat.T
#     c_T_m_inv[0:3, 3] = -rot_mat.T @ trans_vec
#     c_T_m_inv[3, 0:4] = np.array([0, 0, 0, 1])
#
#     transform_mat = m_T_o_inv @ c_T_m_inv
#
#     return transform_mat


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


def drawing_coordinates(h_mat_cam, h_mat=np.eye(4)):
    h_mat = h_mat @ homogeneous_rot('x', -90)
    x = np.concatenate([h_mat[0:3, 3].reshape((1, 3)),
                        h_mat[0:3, 3].reshape((1, 3)) + np.array([h_mat[0:3, 0] * 100]).reshape((1, 3))],
                       axis=0)
    y = np.concatenate([h_mat[0:3, 3].reshape((1, 3)),
                        h_mat[0:3, 3].reshape((1, 3)) + np.array([h_mat[0:3, 1] * 100]).reshape((1, 3))],
                       axis=0)
    z = np.concatenate([h_mat[0:3, 3].reshape((1, 3)),
                        h_mat[0:3, 3].reshape((1, 3)) + np.array([h_mat[0:3, 2] * 100]).reshape((1, 3))],
                       axis=0)

    x_c = np.concatenate([h_mat_cam[0:3, 3].reshape((1, 3)),
                          h_mat_cam[0:3, 3].reshape((1, 3)) + np.array([h_mat_cam[0:3, 0] * 100]).reshape((1, 3))],
                         axis=0)
    y_c = np.concatenate([h_mat_cam[0:3, 3].reshape((1, 3)),
                          h_mat_cam[0:3, 3].reshape((1, 3)) + np.array([h_mat_cam[0:3, 1] * 100]).reshape((1, 3))],
                         axis=0)
    z_c = np.concatenate([h_mat_cam[0:3, 3].reshape((1, 3)),
                          h_mat_cam[0:3, 3].reshape((1, 3)) + np.array([h_mat_cam[0:3, 2] * 100]).reshape((1, 3))],
                         axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()

    ax.plot3D([x[0, 0], x[1, 0]], [x[0, 1], x[1, 1]], [x[0, 2], x[1, 2]], 'red')
    ax.plot3D([x_c[0, 0], x_c[1, 0]], [x_c[0, 1], x_c[1, 1]], [x_c[0, 2], x_c[1, 2]], 'red')
    ax.plot3D([y[0, 0], y[1, 0]], [y[0, 1], y[1, 1]], [y[0, 2], y[1, 2]], 'green')
    ax.plot3D([y_c[0, 0], y_c[1, 0]], [y_c[0, 1], y_c[1, 1]], [y_c[0, 2], y_c[1, 2]], 'green')
    ax.plot3D([z[0, 0], z[1, 0]], [z[0, 1], z[1, 1]], [z[0, 2], z[1, 2]], 'blue')
    ax.plot3D([z_c[0, 0], z_c[1, 0]], [z_c[0, 1], z_c[1, 1]], [z_c[0, 2], z_c[1, 2]], 'blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim3d([0, 1000])
    ax.set_ylim3d([-500, 500])
    ax.set_zlim3d([-1000, 0])

    plt.show()


def zoom(img: np.ndarray, scale, center=None):
    height, width = img.shape[:2]
    rate = height / width

    if center is None:
        center_x = int(width / 2)
        center_y = int(height / 2)
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


class InitializeYOLO:
    def __init__(self, weights_path, img_size=640, conf_threshold=0.05,
                 iou_threshold=0.15, max_det=1, classes=None, agnostic_nms=False,
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
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
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
                try:
                    offset = np.array([[0 - swabs[1] * 1000,
                                        ideal_camera_y + swabs[0] * 1000,
                                        ideal_camera_z - swabs[2] * 1000]]
                                      )

                    list_offset = offset[0].tolist()

                    text = "offset x:{:.3f} y:{:.3f} z:{:.3f}".format(offset[0][0], offset[0][1], offset[0][2])

                    if np.linalg.norm(offset) > 1000:
                        offset_over = "Need to retry swab gripping. max offset: {:.3f} mm".format(np.max(offset))
                        img_raw = cv2.putText(img_raw.copy(), offset_over, (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.7,
                                              (0, 0, 255), 2)

                        if not flag_send:
                            udp_send = struct.pack("ffffffff", 1, 1, 0, 0, 0, 0, 0, 0)

                            self.udp_sender.send_messages(udp_send)

                            flag_send = True

                    else:
                        if not flag_send and frame_keep == 200:
                            udp_send = struct.pack("ffffffff", 0, 1,
                                                   list_offset[0], list_offset[1], list_offset[2], 0, 0, 0)

                            self.udp_sender.send_messages(udp_send)

                            flag_send = True

                except Exception:
                    print("swab_thread: can't find swab")
                    text = "can't find swab"

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


class EndEffectTracking(Thread):
    def __init__(self, rs_device: RealSenseCamera):
        Thread.__init__(self)
        self.device = rs_device
        self.close_flag = False
        self.run_check = False

    def run(self):
        self.run_check = True

        try:
            while True:
                self.device.get_data()

                frameset = self.device.frameset
                self.device.get_aligned_frames(frameset, aligned_to_color=True)

                self.device.frameset = frameset

                self.device.color_frame = frameset.get_color_frame()
                self.device.depth_frame = frameset.get_depth_frame()

                self.device.color_image = frame_to_np_array(self.device.color_frame)

                img_rs = np.copy(self.device.color_image)
                img_raw = np.copy(img_rs)

                resized_image = cv2.resize(img_raw, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                resized_image = cv2.rotate(resized_image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Show images from both cameras
                cv2.namedWindow('RealSense_end-effector', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_end-effector', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_end-effector', resized_image)

                cv2.waitKey(1)

                if self.close_flag:
                    self.run_check = False
                    break

        finally:
            cv2.destroyWindow('RealSense_swab')

    def __bool__(self):
        return self.run_check

    def close_switch(self):
        self.close_flag = True


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
            # if not multi_face_landmarks:
            #     img_zoom = zoom(self.color_image, scale=self.zoom_scale)
            #     self.zoom_flag = True
            #
            #     _, results = mediapipe_detection(img_zoom, self.face_mesh)
            #     multi_face_landmarks = results.multi_face_landmarks

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
                    else:
                        color_image = cv2.circle(self.color_image, (pixel_x, pixel_y), 1, (0, 255, 255), -1)

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
            pass


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_0 = mp.solutions.face_mesh

frame_height, frame_width, channels = (480, 640, 3)

UDP_vision_ip = "169.254.84.185"
UDP_main_ip = "169.254.84.181"

UDP_vision_port = 61496
UDP_main_port = 61480


@torch.no_grad()
def main():
    rs_main = None

    first_iter = True

    total_time = 0

    min_cutoff = 0.0001
    beta = 0.001

    weights_main = "ear_0413.pt"

    cameras = {}
    realsense_device = find_realsense()

    udp_sender = Sender("udp_sender", UDP_main_ip, UDP_main_port)

    for serial, devices in realsense_device:
        cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type='d415')

    for ser, dev in cameras.items():
        rs_main = dev

    if rs_main is None:
        print("can't initialize realsense cameras")

    rs_main.get_data()

    mean_temp = np.zeros((0, 6))
    mean_flag = False

    yolo_main = InitializeYOLO(weights_path=weights_main)

    face_center_prev = None
    flag_path_calc = False

    prev_timestamp = rs_main.current_timestamp

    with mp_face_mesh_0.FaceMesh(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh_0:
        try:
            while True:
                reset = False

                start_time = timeit.default_timer()
                rs_main.get_data()

                current_timestamp = rs_main.current_timestamp

                elapsed = current_timestamp - prev_timestamp
                total_time += elapsed

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

                resized_image = cv2.resize(img_raw, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

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

                        resized_image = cv2.rotate(resized_image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)

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

                        # nose_point = np.mean([x_reshaped[20], x_reshaped[60], x_reshaped[75],
                        #                       x_reshaped[79], x_reshaped[166], x_reshaped[238]],
                        #                      axis=0)

                        nose_point = np.mean([x_reshaped[250], x_reshaped[290], x_reshaped[305],
                                              x_reshaped[309], x_reshaped[392], x_reshaped[458]],
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

                        except AttributeError:
                            print("didn't detect ears")
                            continue

                        vec_no_9 = x_reshaped[9]
                        vec_no_200 = x_reshaped[200]
                        vec_no_4 = x_reshaped[4]

                        # backup current iteration points -> 3d array(number_of_points, 1, 3)
                        points_3d_iter = np.concatenate(
                            (nose_point[np.newaxis, :],
                             vec_ear[np.newaxis, :],
                             vec_no_4[np.newaxis, :],
                             vec_no_9[np.newaxis, :],
                             vec_no_200[np.newaxis, :]),
                            axis=0
                        )

                        number_of_points = points_3d_iter.shape[0]

                        points_3d_iter = points_3d_iter.reshape((number_of_points, 1, 3))

                        # initializing iteration block
                        if first_iter:
                            points_3d = points_3d_iter
                        else:
                            points_3d = np.concatenate((points_3d, points_3d_iter), axis=1)

                        if first_iter:
                            if np.sum(points_3d_iter) != 0:
                                one_euro_x = OneEuroFilter(current_timestamp, points_3d_iter[:, :, 0].T, min_cutoff=min_cutoff,
                                                           beta=beta)
                                one_euro_y = OneEuroFilter(current_timestamp, points_3d_iter[:, :, 1].T, min_cutoff=min_cutoff,
                                                           beta=beta)
                                one_euro_z = OneEuroFilter(current_timestamp, points_3d_iter[:, :, 2].T, min_cutoff=min_cutoff,
                                                           beta=beta)

                                first_iter = False

                            points_3d_hat = points_3d_iter
                        else:
                            points_3d_x_hat = one_euro_x(current_timestamp, points_3d_iter[:, :, 0].T)
                            points_3d_y_hat = one_euro_y(current_timestamp, points_3d_iter[:, :, 1].T)
                            points_3d_z_hat = one_euro_z(current_timestamp, points_3d_iter[:, :, 2].T)

                            points_3d_iter_hat = np.concatenate(
                                (points_3d_x_hat.reshape((number_of_points, 1, 1)),
                                 points_3d_y_hat.reshape((number_of_points, 1, 1)),
                                 points_3d_z_hat.reshape((number_of_points, 1, 1))),
                                axis=2
                            )

                            points_3d_hat = np.concatenate((points_3d_hat, points_3d_iter_hat), axis=1)

                        vec_nose = points_3d_hat[0][-1]
                        vec_ear = points_3d_hat[1][-1]
                        vec_no_4 = points_3d_hat[2][-1]
                        vec_no_9 = points_3d_hat[3][-1]
                        vec_no_200 = points_3d_hat[4][-1]

                        plane_norm_vec = np.cross((vec_no_200 - vec_no_4), (vec_no_9 - vec_no_4))
                        plane_norm_unit_vec = plane_norm_vec / np.linalg.norm(plane_norm_vec)

                        # if plane_norm_unit_vec[2] <= 0:
                        #     img_disp = img_raw.copy()
                        #
                        # else:
                        swab_vec = (vec_nose - vec_ear) / np.linalg.norm(vec_nose - vec_ear)

                        proj_norm_vec = np.dot(swab_vec, plane_norm_unit_vec) * plane_norm_unit_vec
                        proj_vec = (swab_vec - proj_norm_vec) / np.linalg.norm(swab_vec - proj_norm_vec)

                        swab_visualize = vec_nose + proj_vec * 100

                        send_obj = np.append(vec_nose, -proj_vec)

                        # print(send_obj)

                        if mean_flag and send_obj.sum != 0:
                            mean_temp = np.append(mean_temp, send_obj[np.newaxis, :], axis=0)
                            # print(mean_temp[-1])

                            if len(mean_temp) > 100:
                                mean_obj = np.mean(mean_temp, axis=0)[np.newaxis, :]

                                trans_vec_temp = np.append(mean_obj[:, 0:3], np.array([[1]]))[np.newaxis, :]
                                ang_vec_temp = np.append(mean_obj[:, 3:6], np.array([[0]]))[np.newaxis, :]

                                trans_mat = make_transformation_matrix(-45.539, -58.496)

                                trans_vec_rot = trans_mat @ trans_vec_temp.T
                                ang_vec_rot = trans_mat @ ang_vec_temp.T

                                udp_send_array = np.append(trans_vec_rot.T[0, 0:3], ang_vec_rot.T[0, 0:3])
                                udp_send = struct.pack("ffffffff", 0, 2, udp_send_array[0], udp_send_array[1],
                                                       udp_send_array[2], udp_send_array[3], udp_send_array[4],
                                                       udp_send_array[5])

                                udp_sender.send_messages(udp_send)

                                print(udp_send_array)

                                mean_flag = False
                                mean_temp = np.zeros((0, 6))

                            swab_point_0 = rs.rs2_project_point_to_pixel(rs_main.color_intrinsics, vec_nose)
                            swab_point_1 = rs.rs2_project_point_to_pixel(rs_main.color_intrinsics, swab_visualize)

                            img_circle = cv2.circle(img_raw.copy(), list(map(int, swab_point_0)), 2, (255, 0, 0), -1)

                            img_circle = cv2.circle(img_circle.copy(), list(map(int, ear_pixel)), 2, (255, 0, 0), -1)

                            img_disp = cv2.line(img_circle.copy(), list(map(int, swab_point_0)),
                                                list(map(int, swab_point_1)),
                                                color=(0, 0, 255), thickness=2)

                        resized_image = cv2.resize(img_disp, dsize=(0, 0), fx=1, fy=1,
                                                   interpolation=cv2.INTER_AREA)

                rearranged_face = face_points.reshape(468, 3)
                face_center_current = np.average(rearranged_face, axis=0)

                if face_center_prev is not None:
                    offset_norm = np.linalg.norm(face_center_prev - face_center_current)

                    if offset_norm > 75:
                        print("invalid motion detected")

                face_center_prev = face_center_current

                prev_timestamp = current_timestamp

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

                resized_image = cv2.rotate(resized_image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)

                cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_front', resized_image)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
                elif key & 0xFF == ord('p'):
                    if flag_path_calc is False:
                        flag_path_calc = True
                        first_iter = True
                    else:
                        flag_path_calc = False

        finally:
            rs_main.stop()
            udp_sender.close_sender_socket()


if __name__ == '__main__':
    main()
