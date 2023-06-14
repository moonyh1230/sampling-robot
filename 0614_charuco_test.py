from cv2 import aruco
import cv2
import numpy as np
import time
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense


def main():
    ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    ch_board = aruco.CharucoBoard((7, 5), 0.035, 0.027, ar_dict)
    ch_params = aruco.DetectorParameters()
    chessboard_corners = ch_board.getChessboardCorners()

    obj_corners = []
    main_corners = []
    sub_corners = []

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

        elif serial == "123622270472":
            cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type='d405',
                                              color_stream_height=frame_height, color_stream_width=frame_width,
                                              depth_stream_height=frame_height, depth_stream_width=1280,
                                              color_stream_fps=30, depth_stream_fps=30)

            _, rs_sub = cameras.popitem()
            time.sleep(3)

    if (rs_main and rs_sub) is None:
        print("can't initialize realsense cameras")

    cam_mat = rs_main.camera_matrix
    dist = rs_main.dist_coeffs
    intr = rs_main.color_intrinsics

    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)

    while True:
        try:
            rs_main.get_data()
            rs_sub.get_data()

            frameset_main = rs_main.frameset
            rs_main.get_aligned_frames(frameset_main, aligned_to_color=True)

            frameset_sub = rs_sub.frameset
            rs_sub.get_aligned_frames(frameset_sub, aligned_to_color=True)

            img_color_main = np.copy(np.asanyarray(rs_main.color_frame_aligned.get_data()))
            img_gray_main = cv2.cvtColor(img_color_main.copy(), cv2.COLOR_RGB2GRAY)

            img_color_sub = np.copy(np.asanyarray(rs_sub.color_frame_aligned.get_data()))
            img_gray_sub = cv2.cvtColor(img_color_sub.copy(), cv2.COLOR_RGB2GRAY)

            # found_main, ch_corners_main = cv2.findChessboardCorners(img_color_main.copy(), (6, 4),
            #                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
            # found_sub, ch_corners_sub = cv2.findChessboardCorners(img_color_sub.copy(), (6, 4),
            #                                                       cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
            #
            # ch_corners_main = cv2.cornerSubPix(img_gray_main.copy(), ch_corners_main, (5, 5), (-1, -1), term_crit)
            # ch_corners_sub = cv2.cornerSubPix(img_gray_sub.copy(), ch_corners_sub, (5, 5), (-1, -1), term_crit)
            #
            # img_main = cv2.drawChessboardCorners(img_gray_main.copy(),
            #                                      (6, 4),
            #                                      ch_corners_main,
            #                                      found_main)
            #
            # img_sub = cv2.drawChessboardCorners(img_gray_sub.copy(),
            #                                     (6, 4),
            #                                     ch_corners_sub,
            #                                     found_sub)

            cr_main, ids_main, _ = aruco.detectMarkers(img_color_main.copy(), ar_dict,
                                                               parameters=ch_params)
            cr_sub, ids_sub, _ = aruco.detectMarkers(img_color_sub.copy(), ar_dict,
                                                             parameters=ch_params)

            if ids_main is not None and len(ids_main) > 0 and ids_sub is not None and len(ids_sub) > 0:
                _retval, ch_cr_main, ch_ids_main = aruco.interpolateCornersCharuco(cr_main,
                                                                                   ids_main,
                                                                                   img_color_main.copy(),
                                                                                   ch_board)
                _retval, ch_cr_sub, ch_ids_sub = aruco.interpolateCornersCharuco(cr_sub,
                                                                                 ids_sub,
                                                                                 img_color_sub.copy(),
                                                                                 ch_board)

                if ch_ids_main is not None and len(ch_ids_main) > 0 and ch_ids_sub is not None and len(ch_ids_sub) > 0:
                    img_main = aruco.drawDetectedCornersCharuco(img_color_main.copy(), ch_cr_main, ch_ids_main)
                    img_sub = aruco.drawDetectedCornersCharuco(img_color_sub.copy(), ch_cr_sub, ch_ids_sub)

                    img_detected = np.hstack([img_main, img_sub])
                    img_detected = cv2.resize(img_detected.copy(), dsize=(0, 0), fx=0.5, fy=0.5,
                                              interpolation=cv2.INTER_AREA)

                    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("test", img_detected.shape[1], img_detected.shape[0])
                    cv2.imshow("test", img_detected)

                    key = cv2.pollKey()

                    if key & 0xFF == ord('q') or key == 27:
                        cv2.destroyAllWindows()
                        break
                    elif key & 0xFF == ord('r'):
                        if len(ch_ids_main) == len(ch_ids_sub):

                            pass
                    continue

            img_show = np.hstack([img_color_main, img_color_sub])
            img_show = cv2.resize(img_show.copy(), dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", img_show.shape[1], img_show.shape[0])
            cv2.imshow("test", img_show)

            key = cv2.pollKey()

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

        except Exception as e:
            print(str(e))
            pass


if __name__ == "__main__":
    main()
