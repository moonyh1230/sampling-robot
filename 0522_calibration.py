import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array

ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)

squareLength = 0.067
markerLength = 0.040

name = "transform.npz"


def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)

    return result[0], result[1], result[2]


frame_height, frame_width, channels = (1280, 720, 3)


def main():
    cameras = {}
    realsense_device = find_realsense()

    for serial, devices in realsense_device:
        if serial == '135222252454':
            cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type="d455",
                                              color_stream_width=1280, color_stream_height=720,
                                              depth_stream_width=1280, depth_stream_height=720,
                                              depth_stream_fps=30, color_stream_fps=30)

    ser, device = cameras.popitem()


    try:
        while True:
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

            img_color = device.color_image.copy()
            img_gray = cv2.cvtColor(img_color.copy(), cv2.COLOR_RGB2GRAY)

            corners, ids, _ = aruco.detectMarkers(img_gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            if ids is not None and len(ids) == 4:
                dia_corners, dia_ids = aruco.detectCharucoDiamond(img_gray, corners, ids, squareLength / markerLength)
                if len(dia_corners) >= 1:
                    img_draw_dia = aruco.drawDetectedDiamonds(img_color.copy(), dia_corners, dia_ids, (0, 255, 0))
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(dia_corners,
                                                                    squareLength,
                                                                    device.camera_matrix,
                                                                    device.dist_coeffs)
                    img_draw_dia = aruco.drawAxis(img_draw_dia,
                                                  device.camera_matrix,
                                                  device.dist_coeffs,
                                                  rvec,
                                                  tvec,
                                                  0.05)

                    tvec = tvec[0][0]

                    x_sum = dia_corners[0][0][0][0] + dia_corners[0][1][0][0] + \
                            dia_corners[0][2][0][0] + dia_corners[0][3][0][0]
                    y_sum = dia_corners[0][0][0][1] + dia_corners[0][1][0][1] + \
                            dia_corners[0][2][0][1] + dia_corners[0][3][0][1]

                    x_cen = x_sum * .25
                    y_cen = y_sum * .25

                    dist = device.depth_frame_aligned.get_distance(int(x_cen), int(y_cen))

                    if dist != 0:
                        x, y, z = convert_depth_to_phys_coord(x_cen, y_cen, dist, device.color_intrinsics)
                        dx = (tvec[0] - x) * 1000
                        dy = (tvec[1] - y) * 1000
                        dz = (tvec[2] - z) * 1000

                        print("dx = {}, dy = {}, dz = {}".format(dx, dy, dz))

                else:
                    img_draw_dia = img_color

            else:
                img_draw_dia = img_color

            resized_image = cv2.resize(img_draw_dia.copy(), dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)

            cv2.namedWindow('{}_RealSense_front'.format(ser), cv2.WINDOW_NORMAL)
            cv2.resizeWindow('{}_RealSense_front'.format(ser), resized_image.shape[1], resized_image.shape[0])
            cv2.imshow('{}_RealSense_front'.format(ser), resized_image)

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        device.stop()


if __name__ == '__main__':
    main()
