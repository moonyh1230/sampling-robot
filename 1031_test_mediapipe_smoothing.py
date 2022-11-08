import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import cv2

from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from RealSense_Utilities.realsense_api.realsense_api import mediapipe_detection

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh_0 = mp.solutions.face_mesh

frame_height, frame_width, channels = (480, 640, 3)


def main():
    rs_main = None

    cameras = {}
    realsense_device = find_realsense()

    for serial, devices in realsense_device:
        if serial == '105322250965':
            cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True)

    for ser, dev in cameras.items():
        if ser == '105322250965':
            rs_main = dev

    with mp_face_mesh_0.FaceMesh(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh_0:

        try:
            while True:
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

                _, results = mediapipe_detection(img_rs0, face_mesh_0)
                multi_face_landmarks = results.multi_face_landmarks

                try:
                    if multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]

                        points_3d = np.zeros((0, 3))

                        for i in range(468):
                            pixel_point = face_landmarks.landmark[i]
                            pixel_x = int(pixel_point.x * frame_width)
                            pixel_y = int(pixel_point.y * frame_height)

                            _ = cv2.circle(img_rs0, (pixel_x, pixel_y), 2, (0, 0, 255), -1)

                finally:
                    pass

                resized_image = cv2.rotate(img_rs0.copy(), cv2.ROTATE_90_CLOCKWISE)
                cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
                cv2.imshow('RealSense_front', resized_image)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

        finally:
            rs_main.stop()


if __name__ == '__main__':
    main()
