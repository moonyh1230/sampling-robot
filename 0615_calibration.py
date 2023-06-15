from cv2 import aruco
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from json_tricks.np import dumps


def make_gripper2base(theta_0, theta_1):
    t_0 = homogeneous_trans(0, 323.5, 0)
    t_1 = homogeneous_trans(0, 416.63, 0)
    r_0 = homogeneous_rot('z', -theta_0)
    r_1 = homogeneous_rot('z', -theta_1)

    transform_matrix = r_0 @ t_0 @ r_1 @ t_1
    r_g2b = transform_matrix[0:3, 0:3]
    t_g2b = transform_matrix[0:3, 3]

    return r_g2b, t_g2b


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


def main():
    ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    ch_board = aruco.CharucoBoard((7, 5), 0.035, 0.027, ar_dict)
    ch_params = aruco.DetectorParameters()

    frame_height, frame_width, channels = (720, 1280, 3)

    cameras = {}
    realsense_device = find_realsense()

    rs_main = None
    rs_sub = None

    for serial, devices in realsense_device:
        if serial == "135222252454":
            cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type='d455',
                                              color_stream_height=frame_height, color_stream_width=frame_width,
                                              depth_stream_height=frame_height, depth_stream_width=1280,
                                              color_stream_fps=30, depth_stream_fps=30)

            _, rs_main = cameras.popitem()
            time.sleep(3)

    if (rs_main and rs_sub) is None:
        print("can't initialize realsense cameras")

    cam_mat = rs_main.camera_matrix
    dist = rs_main.dist_coeffs
    intr = rs_main.color_intrinsics

    r_gripper_to_base = []
    t_gripper_to_base = []

    r_target_to_cam = []
    t_target_to_cam = []

    record_count = 0

    while True:
        try:
            tvec = None
            rvec = None
            rs_main.get_data()

            frameset_main = rs_main.frameset
            rs_main.get_aligned_frames(frameset_main, aligned_to_color=True)

            img_color_main = np.copy(np.asanyarray(rs_main.color_frame_aligned.get_data()))
            img_gray_main = cv2.cvtColor(img_color_main.copy(), cv2.COLOR_RGB2GRAY)

            cr_main, ids_main, _ = aruco.detectMarkers(img_gray_main.copy(), ar_dict,
                                                       parameters=ch_params)

            if ids_main is not None and len(ids_main) > 0:
                _retval, ch_cr_main, ch_ids_main = aruco.interpolateCornersCharuco(cr_main,
                                                                                   ids_main,
                                                                                   img_gray_main.copy(),
                                                                                   ch_board)

                if ch_ids_main is not None and len(ch_ids_main) > 0:
                    val, rvec, tvec = aruco.estimatePoseCharucoBoard(ch_cr_main,
                                                                     ch_ids_main,
                                                                     ch_board,
                                                                     cam_mat,
                                                                     dist,
                                                                     rvec,
                                                                     tvec)

                    if val:
                        img_main = cv2.drawFrameAxes(img_color_main.copy(),
                                                     ch_cr_main,
                                                     ch_ids_main,
                                                     rvec,
                                                     tvec,
                                                     0.03)

                        img_detected = cv2.resize(img_main.copy(), dsize=(0, 0), fx=0.5, fy=0.5,
                                                  interpolation=cv2.INTER_AREA)

                        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("test", img_detected.shape[1], img_detected.shape[0])
                        cv2.imshow("test", img_detected)

                        key = cv2.pollKey()

                        if key & 0xFF == ord('q') or key == 27:
                            cv2.destroyAllWindows()
                            break
                        elif key & 0xFF == ord('r'):
                            t0 = float(input("input joint 0 angle: "))
                            t1 = float(input("input joint 1 angle: "))

                            r_g, t_g = make_gripper2base(t0, t1)
                            r_gripper_to_base.append(r_g)
                            t_gripper_to_base.append(t_g)

                            r_target_to_cam.append(rvec)
                            t_target_to_cam.append(tvec)

                            print("{} pose recorded".format(record_count))

                            record_count += 1

                        continue

            img_show = cv2.resize(img_color_main.copy(), dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", img_show.shape[1], img_show.shape[0])
            cv2.imshow("test", img_show)

            key = cv2.pollKey()

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('c'):
                if record_count > 3:
                    cv2.destroyAllWindows()

                    print("start calibration with {} records".format(record_count))

                    try:
                        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(r_gripper_to_base,
                                                                            t_gripper_to_base,
                                                                            r_target_to_cam,
                                                                            t_target_to_cam,
                                                                            method=cv2.CALIB_HAND_EYE_TSAI)

                        print(r_cam2gripper, t_cam2gripper)
                        np.savez_compressed("cam_to_gripper.npz", rot=r_cam2gripper, trans=t_cam2gripper)

                    except Exception as e:
                        print(str(e))
                        break

                    break

        except Exception as e:
            print(str(e))
            pass


if __name__ == "__main__":
    main()
