import os
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation
from camera_handler import CameraHandler

# camera 1
CAMERA_INTRINSICS_MAT = np.array(
    [[610.331, 0, 312.435], [0, 608.768, 246.99], [0, 0, 1]], dtype=np.float32)
# camera 2
# CAMERA_INTRINSICS_MAT = np.array(
# [[614.182, 0, 315.91], [0, 614.545, 244.167], [0, 0, 1]], dtype=np.float32)
CAMERA_DISTORTION_COEFF_MAT = np.array([0, 0, 0, 0, 0], dtype=np.float32)
ARUCO_NAME = cv2.aruco.DICT_4X4_50
MARKER_SIDE_LENGTH_MM = 93


class HandEyeCalibration():
    """
    To solve the pose from camera to robot base X, we need to solve a function 
    AX = XB, A = inv(T_base_ee_next) * T_base_ee_current, 
    B = T_marker_cam_next * inv(T_marker_cam_current)
    """

    def __init__(self, camera_intrinsics_mat, camera_distortion_coeff_mat):
        self.cam_intr_mat = camera_intrinsics_mat
        self.cam_dist_coeff_mat = camera_distortion_coeff_mat
        self.marker_side_length_mm = MARKER_SIDE_LENGTH_MM
        self.aruco_dict = cv2.aruco.Dictionary_get(ARUCO_NAME)
        self.detector_parameters = cv2.aruco.DetectorParameters_create()
        self.marker_to_cam_poses_list = []
        self.R_marker_to_cam_poses_list = []
        self.t_marker_to_cam_poses_list = []
        self.ee_to_base_poses_list = []
        self.R_ee_to_base_poses_list = []
        self.t_ee_to_base_poses_list = []
        self.realsense_handler = CameraHandler((640, 480))

    def draw_marker(self):
        img = cv2.aruco.drawMarker(self.aruco_dict, 0, 700)
        plt.imshow(img, cmap=mpl.cm.gray)
        plt.show()

    def read_ee_to_base_poses_from_file(self, filename):
        self.ee_to_base_poses_list = []
        with open(filename, 'r') as f:
            for line_id, line in enumerate(f):
                row = line.strip("\n").split(",")
                row = np.array([float(x) for x in row], dtype=np.float32)
                A_i = row.reshape((4, 4))
                A_i = A_i.T
                self.ee_to_base_poses_list.append(A_i)
                self.R_ee_to_base_poses_list.append(A_i[0:3,0:3])
                self.t_ee_to_base_poses_list.append(A_i[0:3,3])

    def save_images_with_key(self, save_dir):
        self.realsense_handler.save_left_camera_images(save_dir)

    def get_images_list_from_dir(self, images_dir):
        assert(os.path.exists(images_dir))
        image_paths_list = sorted(os.listdir(images_dir))
        images_list = []
        for image_idx, image_name in enumerate(image_paths_list):
            image_path = os.path.join(images_dir, image_name)
            image = cv2.imread(image_path)
            images_list.append(image)
        return images_list

    def compute_A(self):
        A = np.zeros((4, (len(self.ee_to_base_poses_list)-1)*4))
        for pose_idx, (current_pose, next_pose) in enumerate(zip(self.ee_to_base_poses_list)):
            A_i = np.linalg.inv(next_pose) @ current_pose
            A[:, pose_idx * 4: pose_idx * 4 + 4] = A_i
        return A

    def compute_B(self):
        B = np.zeros((4, (len(self.marker_to_cam_poses_list)-1)*4))
        for pose_idx, (current_pose, next_pose) in enumerate(zip(self.marker_to_cam_poses_list[:-1], self.marker_to_cam_poses_list[1:])):
            B_i = next_pose @ np.linalg.inv(current_pose)
            B[:, pose_idx * 4: pose_idx * 4 + 4] = B_i
        return B

    def get_poses_from_images(self, images_list):
        self.marker_to_cam_poses_list = []
        for image_idx, image in enumerate(images_list):
            self.marker_to_cam_poses_list.append(
                self.get_pose_from_one_image(image))
            self.R_marker_to_cam_poses_list.append(
                self.get_pose_from_one_image(image)[0:3,0:3]
            )
            self.t_marker_to_cam_poses_list.append(
                self.get_pose_from_one_image(image)[0:3,-1]
            )

    def get_pose_from_one_image(self, image):
        marker_corners = self.detect_aruco_corners(image)
        rot_vec, trans_vec, _ = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners, self.marker_side_length_mm, self.cam_intr_mat, self.cam_dist_coeff_mat)
        cv2.aruco.estimatePoseSingleMarkers()
        sci_rotation = Rotation.from_rotvec(rot_vec)
        rot_mat = sci_rotation.as_dcm()
        trans_mat = np.concatenate(
            [rot_mat, trans_vec.reshape((3, 1))], axis=1)
        trans_mat = np.concatenate(
            [trans_mat, np.array([0, 0, 0, 1]).reshape((1, 4))], axis=0)
        return trans_mat

    def detect_aruco_corners(self, image):
        marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(
            image, self.aruco_dict, parameters=self.detector_parameters)
        return marker_corners

    def test_opencv_key_press(self):
        cap = cv2.VideoCapture(0)
        counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
            k = cv2.waitKey(33)
            if k == ord('s'):
                cv2.imwrite('{0:06}.png'.format(counter), frame)
                counter += 1
            elif k == ord('q'):
                break

        cap.release()

if __name__ == '__main__':
    # calib = HandEyeCalibration(
        # CAMERA_INTRINSICS_MAT, CAMERA_DISTORTION_COEFF_MAT)
    # calib.save_images_with_key("./")
    # handler = CameraHandler((640, 480))
    # # handler.save_left_camera_images("./")
    # handler.save_right_camera_images("./")
    
    calib = HandEyeCalibration(
        CAMERA_INTRINSICS_MAT, CAMERA_DISTORTION_COEFF_MAT)

    # Get R,t ee_to_base 
    calib.read_ee_to_base_poses_from_file("pose.txt")
    T_ee_base = calib.ee_to_base_poses_list
    R_ee_base = calib.R_ee_to_base_poses_list
    t_ee_base = calib.t_ee_to_base_poses_list
 
    # Get R,T c_to_marker
    img_list = calib.get_images_list_from_dir("")
    calib.get_poses_from_images(img_list)
    R_marker_cam = calib.R_marker_to_cam_poses_list
    t_marker_cam = calib.t_marker_to_cam_poses_list
    R_cam_marker = []
    t_cam_marker = []
    for r,t in zip(R_marker_cam,t_marker_cam):
        R_cam_marker.append(r.T) 
        t_cam_marker.append(-r.T.dot(t))

    R_marker_to_ee, t_marker_to_ee = cv2.calibrateHandEye(R_ee_base,t_ee_base,R_cam_marker,t_cam_marker,cv2.CALIB_HAND_EYE_DANIILIDIS)

    # T_cam_marker
    T_cam_marker = np.concatenate(
            [R_cam_marker[0], t_cam_marker[0].reshape((3, 1))], axis=1)
    T_cam_marker = np.concatenate(
            [T_cam_marker, np.array([0, 0, 0, 1]).reshape((1, 4))], axis=0)

    # T_marker_to_ee
    T_marker_to_ee = np.concatenate(
            [R_marker_to_ee, t_marker_to_ee.reshape((3, 1))], axis=1)
    T_marker_to_ee = np.concatenate(
            [T_marker_to_ee, np.array([0, 0, 0, 1]).reshape((1, 4))], axis=0)
    
    T_cam_base = T_ee_base[0].dot(T_marker_to_ee).dot(T_cam_marker)
    print(T)