import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
import threading


class CameraHandler:

    __instance = None

    @staticmethod
    def getInstance(resolution, device_id=None):
        """ Static access method. """
        if CameraHandler.__instance == None:
            CameraHandler(resolution, device_id)
        return CameraHandler.__instance

    def __init__(self, resolution, device_id=None):
        if CameraHandler.__instance != None:
            raise Exception("This class: CameraHandler is a singleton!")
        else:
            CameraHandler.__instance = self

            # Start Camera device for the observations
            self.pipeline = rs.pipeline()
            self.__color_frame = None
            self.__depth_frame = None
            self.__resolution = resolution

            # find the id via devices = rs.context().query_devices()
            #devices[0], devices[1]
            if device_id is None:
                # observation: "828112071102"   "829212070352"robot right  "943222073921" robot_left
                self.device_id = "943222073921"
                self.reward_devices_id = "829212070352"
                self.observation_device_id = "828112071102"
            else:
                self.device_id = device_id

            # For observation camera
            self.obesrvation_pipeline = rs.pipeline()
            self.obesrvation_config = rs.config()
            self.obesrvation_config.enable_device(self.observation_device_id)
            self.obesrvation_config.enable_stream(
                rs.stream.depth, 640, 480, rs.format.z16, 60)
            self.obesrvation_config.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 60)

            self.observation_color = None
            self.observation_depth = None

            # For reward cameras
            self.config = rs.config()
            self.config.enable_device(self.device_id)
            self.config.enable_stream(
                rs.stream.depth, 640, 480, rs.format.z16, 60)
            self.config.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 60)

            # Start the second camera
            self.reward_pipeline = rs.pipeline()
            self.reward_config = rs.config()
            self.reward_config.enable_device(self.reward_devices_id)
            self.reward_config.enable_stream(
                rs.stream.depth, 640, 480, rs.format.z16, 60)
            self.reward_config.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 60)

            self.reward_frame = None
            self.reward_depth_frame = None

            self.color_image = None
            self.color_frame = None
            self.depth_image = None
            self.depth_frame = None
            self.frame_count = 0

            # New buffers for color , depth and the distance(object to target)
            self.color_buffer = deque(maxlen=10)
            self.depth_buffer = deque(maxlen=10)
            self.distance_buffer = deque([1], maxlen=10)

            # Aruco marker part
            # Load the dictionary that was used to generate the markers.
            # camera observation
            self.obj_position = None
            self.goal_position = None
            # camera reward
            self.obj_position_reward = None
            self.goal_position_reward = None

            # position in base frame
            self.goal_position_base = None
            self.obj_position_base = None

            self.marker_0_base = None
            self.marker_1_base = None
            self.marker_3_base = None
            self.marker_4_base = None
            self.marker_5_base = None

            self.dictionary = cv2.aruco.Dictionary_get(
                cv2.aruco.DICT_ARUCO_ORIGINAL)

            # Initialize the detector parameters using default values
            self.parameters = cv2.aruco.DetectorParameters_create()

            self.camera_thread = threading.Thread(
                name="camera_thread", target=self.start_pipeline)

    def get_color_frame(self):
        return self.color_buffer[-1]

    def get_depth_frame(self):
        return self.depth_buffer[-1]

    def grab_distance(self):
        return self.distance_buffer[-1]

    def get_current_obj(self):
        return self.obj_position_base, self.marker_0_base, self.marker_1_base

    def get_targetmarkers(self):
        return self.marker_3_base, self.marker_4_base, self.marker_5_base

    def start_pipeline(self):
        # self.pipeline.start()
        # align_to = rs.stream.color
        # align = rs.align(align_to)
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        # start Observation Camera
        self.obesrvation_pipeline.start(self.obesrvation_config)
        obesrvation_align_to = rs.stream.color
        self.obesrvation_align = rs.align(obesrvation_align_to)

        # start Reward Camera
        profile = self.pipeline.start(self.config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.reward_pipeline.start(self.reward_config)
        reward_align_to = rs.stream.color
        self.reward_align = rs.align(reward_align_to)

        try:
            while True:
                # Observation Camera
                observation_frames = self.obesrvation_pipeline.wait_for_frames(
                    200 if (self.frame_count > 1) else 10000)
                observation_aligned_frames = self.obesrvation_align.process(
                    observation_frames)
                observation_color_frame = np.asanyarray(
                    observation_aligned_frames.get_color_frame().get_data())
                observation_depth_frame = np.asanyarray(
                    observation_aligned_frames.get_depth_frame().get_data())

                self.observation_color = observation_color_frame
                self.color_buffer.append(self.observation_color)
                self.observation_depth = observation_depth_frame
                self.depth_buffer.append(self.observation_depth)

                # Reward Camera left
                frames = self.pipeline.wait_for_frames(
                    200 if (self.frame_count > 1) else 10000)  # wait 10 seconds for first frame
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                self.depth_frame = depth_frame
                self.__depth_frame = np.asanyarray(depth_frame.get_data())

                self.color_frame = np.asanyarray(
                    aligned_frames.get_color_frame().get_data())
                self.color_image = self.color_frame
                self.__color_frame = cv2.resize(
                    self.color_frame, self.__resolution)

                # Reward Camera right
                reward_frames = self.reward_pipeline.wait_for_frames(
                    200 if (self.frame_count > 1) else 10000)
                reward_aligned_frames = self.reward_align.process(
                    reward_frames)
                reward_color_frame = np.asanyarray(
                    reward_aligned_frames.get_color_frame().get_data())

                self.reward_frame = reward_color_frame
                self.reward_depth_frame = reward_aligned_frames.get_depth_frame()

                # Compute the distance and store them in the buffer
                distance_temp = self.cal_distance()

                if (distance_temp == 1):
                    self.distance_buffer.append(self.distance_buffer[-1])

                if (distance_temp <= 1):
                    self.distance_buffer.append(distance_temp)

                # Filter the outlier
                if self.distance_buffer[-2] != 1 and abs(self.distance_buffer[-1]-self.distance_buffer[-2]) > 0.1:
                    self.distance_buffer.append(self.distance_buffer[-2])

                    # if self.distance_buffer[-1]!=1 and abs(distance_temp-self.distance_buffer[-1])>0.2:
                    # if (distance_temp < 0.8):
                    #     self.distance_buffer.append(distance_temp)
                    # else:
                    #     self.distance_buffer.append(self.distance_buffer[-1])

                # if self.color_image is not None:
                #     cv2.imshow('right',self.color_image)
                #     cv2.imshow('left', self.reward_frame)
                #     cv2.imshow('observation', self.depth_buffer[-1])
                # key = cv2.waitKey(1)
                # if key & 0xFF == ord('q') or key == 27:
                #     cv2.destroyAllWindows()
                #     break

        except KeyboardInterrupt:
            self.camera_thread.join()

    # Capture current frame  (like shooting a picture)

    def _capture(self):

        # get the frames
        frames = self.pipeline.wait_for_frames(
            200 if (self.frame_count > 1) else 10000)  # wait 10 seconds for first frame

        # convert camera frames to images

        # Align the depth frame to color frame
        # if self.enable_depth and self.enable_rgb else None
        aligned_frames = self.align.process(frames)
        # if aligned_frames is not None else frames.get_depth_frame()
        depth_frame = aligned_frames.get_depth_frame()
        # if aligned_frames is not None else frames.get_color_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        self.depth_frame = depth_frame
        self.color_frame = color_frame
        # if self.enable_depth else None
        self.depth_image = np.asanyarray(depth_frame.get_data())
        # if self.enable_rgb else None
        self.color_image = np.asanyarray(color_frame.get_data())

        # return original images including color image and depth image(CV form) along wiht color,depth frame(for pyrealsense)
        return self.color_image, self.depth_image, self.depth_frame, self.color_frame

    def get_marker_position(self):

        # Option 1: Capture the frame each time to get a series of frame
        # color_image,depth_image,depth_frame,color_frame=self._capture()

        # Option 2: Camera start_pipeline runs in the background and get frame each time
        color_image = self.color_image
        # depth_frame = self.depth_frame
        # Add new camera for detection
        reward_color = self.reward_frame

        # Aruco marker part
        # Detect the markers in the image
        markerIds_temp = None

        # Transformation robotbase to two cameras
        "left"
        "               [-0.391273,   0.27366, -0.878642, 0.0620607]"
        " base_T_EE     [0.0879567,  0.961503,  0.260304,  0.315562]"
        "               [ 0.916051, 0.0245676, -0.400288,  0.943767]"
        "               [        0,         0,         0,         1]"
        "               [0,0,-1,0]"
        " EE_T_camera   [-1,0,0,0]"
        "               [0,1,0,0]"
        "               [0,0,0,1] "
        # EE_T_camera_left = np.array(
        #     [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        # base_T_EE_left = np.array([[-0.391273, 0.27366, -0.878642, 0.0620607], [0.0879567, 0.961503,
        #                                                                         0.260304, 0.315562], [0.916051, 0.0245676, -0.400288, 0.943767], [0, 0, 0, 1]])
        # base_T_camera_left = base_T_EE_left.dot(EE_T_camera_left)
        base_T_camera_left = np.array([ [-2.92152513e-01, -8.60829435e-01,  4.16656445e-01, -8.20919629e-02],
                                        [-9.56361983e-01,  2.63047919e-01, -1.27119770e-01,  3.63598337e-01],
                                        [-1.71864357e-04, -4.35626443e-01, -9.00125638e-01,  9.31717567e-01],
                                        [0.         , 0.         , 0.         , 1.        ]])
        base_R_left = base_T_camera_left[0:3, 0:3]
        base_T_left = base_T_camera_left[0:3, 3]
        "right"
        "               [ -0.40339,-0.182166,-0.89671,0.0503344]"
        " base_T_EE     [ -0.0714206,0.983251,-0.167621,-0.296183]"
        "               [  0.912226,-0.00357328,-0.409652,0.943524]"
        "               [       0,         0,         0,         1]"
        "               [0,0,-1,0]"
        " EE_T_camera   [-1,0,0,0]"
        "               [0,1,0,0]"
        "               [0,0,0,1]  "
        # EE_T_camera_right = np.array(
        #     [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        # base_T_EE_right = np.array([[-0.40339, -0.182166, -0.89671, 0.0503344], [-0.0714206, 0.983251, -
        #                                                                          0.167621, -0.296183], [0.912226, -0.00357328, -0.409652, 0.943524], [0, 0, 0, 1]])
        # base_T_camera_right = base_T_EE_right.dot(EE_T_camera_right)
        # base_T_camera_right = np.array([[ 0.09433898, -0.8902616 ,   0.44555711, -0.11539241],
        #                                 [-0.99553126, -0.08357619,  0.04380608 ,-0.29629556],
        #                                 [-0.00176641, -0.44770838, -0.89417077 , 0.92629916],
        #                                 [0.         , 0.         , 0.         , 1.        ]])
        base_T_camera_right = np.array([[ 0.14873457, -0.89726813,  0.4156554 , -0.10166122],
                                        [-0.98851075, -0.1235298 ,  0.08705028, -0.30935999],
                                        [-0.0267716 , -0.42382986, -0.90534073,  0.94275621],
                                        [0.         , 0.         , 0.         , 1.        ]])                                
        base_R_right = base_T_camera_right[0:3, 0:3]
        base_T_right = base_T_camera_right[0:3, 3]

        # Camera left
        if color_image is not None:
            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
                color_image, self.dictionary, parameters=self.parameters)
            if markerIds is not None:
                depth_frame = self.depth_frame
                markerIds_temp = markerIds
                markerCorners_temp = markerCorners
                aruco_list = {}
                # centre= {}
                result_center = {}
                # orient_centre= {}
                if markerIds_temp is not None:
                    # Print corners and ids to the console
                    # result=zip(markerIds, markerCorners)
                    for k in range(len(markerCorners_temp)):
                        temp_1 = markerCorners_temp[k]
                        temp_1 = temp_1[0]
                        temp_2 = markerIds_temp[k]
                        temp_2 = temp_2[0]
                        aruco_list[temp_2] = temp_1
                    key_list = aruco_list.keys()
                    # print(key_list)
                    for key in key_list:
                        dict_entry = aruco_list[key]
                        centre = dict_entry[0] + dict_entry[1] + \
                            dict_entry[2] + dict_entry[3]
                        centre[:] = [int(x / 4) for x in centre]
                        centre = tuple(centre)
                        result_center[key] = centre
                # print(result_center)
                # try:
                point_obj = None
                point_target = None

                point_marker_0 = None
                point_marker_1 = None
                point_marker_3 = None
                point_marker_4 = None
                point_marker_5 = None
                if result_center.get(0) != None:
                    x_id0 = result_center[0][0]
                    y_id0 = result_center[0][1]
                    p_0 = [x_id0, y_id0]
                    # Deproject pixel to 3D point
                    # point_obj = self.pixel2point(depth_frame, p_0)
                    # modified 30,07
                    point_marker_0 = self.pixel2point(depth_frame, p_0)

                if result_center.get(1) != None:
                    x_id1 = result_center[1][0]
                    y_id1 = result_center[1][1]
                    p_1 = [x_id1, y_id1]
                    # Deproject pixel to 3D point
                    # point_obj = self.pixel2point(depth_frame, p_1)
                    # modified 30,07
                    point_marker_1 = self.pixel2point(depth_frame, p_1)

                if(result_center.get(5) != None):
                    x_id5 = result_center[5][0]
                    y_id5 = result_center[5][1]
                    p_5 = [x_id5, y_id5]
                    # Deproject pixel to 3D point
                    point_5 = self.pixel2point(depth_frame, p_5)
                    point_marker_5 = point_marker_5 = np.array(point_5)

                # Dual ID-4 and ID-3
                if(result_center.get(4) != None):
                    x_id4 = result_center[4][0]
                    y_id4 = result_center[4][1]
                    p_4 = [x_id4, y_id4]
                    # Deproject pixel to 3D point
                    point_4 = self.pixel2point(depth_frame, p_4)
                    point_marker_4 = point_marker_4 = np.array(point_4)

                if(result_center.get(3) != None):
                    x_id3 = result_center[3][0]
                    y_id3 = result_center[3][1]
                    p_3 = [x_id3, y_id3]
                    # Deproject pixel to 3D point
                    point_3 = self.pixel2point(depth_frame, p_3)
                    point_marker_3 = point_marker_3 = np.array(point_3)

                # Calculate target point
                if (result_center.get(5) != None and result_center.get(4) != None and result_center.get(3) != None):
                    point_target = [point_4[0]+point_3[0]-point_5[0], point_4[1] +
                                    point_3[1]-point_5[1], point_4[2]+point_3[2]-point_5[2]]

                # store the target /obj position if detected
                self.goal_position = point_target

                # modified 30,07
                if point_marker_0 is not None and point_marker_1 is not None:
                    point_obj = (np.array(point_marker_0) +
                                 np.array(point_marker_1))/2
                    self.obj_position = point_obj

                # In case the target 3 markers cannot be detected at the same time, store the markers that can be detected
                # modified 30,07
                if point_marker_0 is not None:
                    self.marker_0_base = base_R_left.dot(
                        point_marker_0) + base_T_left
                if point_marker_1 is not None:
                    self.marker_1_base = base_R_left.dot(
                        point_marker_1) + base_T_left
                if point_marker_3 is not None:
                    self.marker_3_base = base_R_left.dot(
                        point_marker_3) + base_T_left
                if point_marker_4 is not None:
                    self.marker_4_base = base_R_left.dot(
                        point_marker_4) + base_T_left
                if point_marker_5 is not None:
                    self.marker_5_base = base_R_left.dot(
                        point_marker_5) + base_T_left

                # Transform to base
                if self.goal_position != None:
                    self.goal_position = np.array(self.goal_position)
                    self.goal_position_base = base_R_left.dot(
                        self.goal_position) + base_T_left

                if self.obj_position is not None:
                    self.obj_position = np.array(self.obj_position)
                    self.obj_position_base = base_R_left.dot(
                        self.obj_position) + base_T_left

        # Camera right
        if reward_color is not None:
            markerCorners_reward, markerIds_reward, rejectedCandidates_reward = cv2.aruco.detectMarkers(
                reward_color, self.dictionary, parameters=self.parameters)
            if markerIds_reward is not None:

                depth_frame = self.reward_depth_frame
                markerIds_temp = markerIds_reward
                markerCorners_temp = markerCorners_reward
                aruco_list = {}
                # centre= {}
                result_center = {}
                # orient_centre= {}
                if markerIds_temp is not None:
                    # Print corners and ids to the console
                    # result=zip(markerIds, markerCorners)
                    for k in range(len(markerCorners_temp)):
                        temp_1 = markerCorners_temp[k]
                        temp_1 = temp_1[0]
                        temp_2 = markerIds_temp[k]
                        temp_2 = temp_2[0]
                        aruco_list[temp_2] = temp_1
                    key_list = aruco_list.keys()
                    # print(key_list)
                    for key in key_list:
                        dict_entry = aruco_list[key]
                        centre = dict_entry[0] + dict_entry[1] + \
                            dict_entry[2] + dict_entry[3]
                        centre[:] = [int(x / 4) for x in centre]
                        centre = tuple(centre)
                        result_center[key] = centre

                point_obj = None
                point_target = None
                # modified 30,07
                point_marker_0 = None
                point_marker_1 = None
                point_marker_3 = None
                point_marker_4 = None
                point_marker_5 = None
                if result_center.get(0) != None:
                    x_id0 = result_center[0][0]
                    y_id0 = result_center[0][1]
                    p_0 = [x_id0, y_id0]
                    # Deproject pixel to 3D point
                    # point_obj = self.pixel2point(depth_frame, p_0)
                    # modified 30,07
                    point_marker_0 = self.pixel2point(depth_frame, p_0)

                # modified 30,07
                if result_center.get(1) != None:
                    x_id1 = result_center[1][0]
                    y_id1 = result_center[1][1]
                    p_1 = [x_id1, y_id1]
                    # Deproject pixel to 3D point
                    point_marker_1 = self.pixel2point(depth_frame, p_1)

                if(result_center.get(5) != None):
                    x_id5 = result_center[5][0]
                    y_id5 = result_center[5][1]
                    p_5 = [x_id5, y_id5]
                    # Deproject pixel to 3D point
                    point_5 = self.pixel2point(depth_frame, p_5)
                    point_marker_5 = np.array(point_5)

                # Dual ID-4 and ID-3
                if(result_center.get(4) != None):
                    x_id4 = result_center[4][0]
                    y_id4 = result_center[4][1]
                    p_4 = [x_id4, y_id4]
                    # Deproject pixel to 3D point
                    point_4 = self.pixel2point(depth_frame, p_4)
                    point_marker_4 = np.array(point_4)

                if(result_center.get(3) != None):
                    x_id3 = result_center[3][0]
                    y_id3 = result_center[3][1]
                    p_3 = [x_id3, y_id3]
                    # Deproject pixel to 3D point
                    point_3 = self.pixel2point(depth_frame, p_3)
                    point_marker_3 = np.array(point_3)

                # Calculate target point
                if (result_center.get(5) != None and result_center.get(4) != None and result_center.get(3) != None):
                    point_target = [point_4[0]+point_3[0]-point_5[0], point_4[1] +
                                    point_3[1]-point_5[1], point_4[2]+point_3[2]-point_5[2]]

                # store the target /obj position if detected
                self.goal_position_reward = point_target

                # modified 30,07
                if point_marker_0 is not None and point_marker_1 is not None:
                    point_obj = (np.array(point_marker_0) +
                                 np.array(point_marker_1))/2
                    self.obj_position_reward = point_obj

                # In case the target  markers cannot be detected at the same time, store the markers that can be detected
                # modified 30,07
                if point_marker_0 is not None:
                    self.marker_0_base = base_R_right.dot(
                        point_marker_0) + base_T_right
                if point_marker_1 is not None:
                    self.marker_1_base = base_R_right.dot(
                        point_marker_1) + base_T_right
                if point_marker_3 is not None:
                    self.marker_3_base = base_R_right.dot(
                        point_marker_3) + base_T_right
                if point_marker_4 is not None:
                    self.marker_4_base = base_R_right.dot(
                        point_marker_4) + base_T_right
                if point_marker_5 is not None:
                    self.marker_5_base = base_R_right.dot(
                        point_marker_5) + base_T_right

                # Transform to base

                if self.goal_position_reward != None:
                    self.goal_position_reward = np.array(
                        self.goal_position_reward)
                    self.goal_position_base = base_R_right.dot(
                        self.goal_position_reward) + base_T_right

                if self.obj_position_reward is not None:
                    self.obj_position_reward = np.array(
                        self.obj_position_reward)
                    self.obj_position_base = base_R_right.dot(
                        self.obj_position_reward) + base_T_right

        return self.obj_position_base, self.goal_position_base

    def cal_distance(self):
        # In base frame
        obj, target = self.get_marker_position()
        # In case the target cannot be detected with single camera,use the markers could be detected by mutiple cameras
        if target is None and self.marker_3_base is not None and self.marker_4_base is not None and self.marker_5_base is not None:
            target = self.marker_3_base + self.marker_4_base - self.marker_5_base
        if obj is None and self.marker_0_base is not None and self.marker_1_base is not None:
            obj = self.marker_0_base/2 + self.marker_1_base/2
        if obj is not None and target is not None:
            dis_obj2target_goal = np.linalg.norm(target-obj)
        # In case the target or obj cannot be detected by both cameras
        else:
            dis_obj2target_goal = 1
        return dis_obj2target_goal

    def pixel2point(self, frame, u):

        u_x = int(u[0])
        u_y = int(u[1])
        # Get depth from pixels
        dis2cam_u = frame.get_distance(u_x, u_y)
        dis2cam_u_alongx = 0
        dis2cam_u_alongy = 0
        # Make the depth value stable use more pixels
        # along x axis
        for i in range(1, 3):
            dis2cam_u_alongx = dis2cam_u_alongx + \
                frame.get_distance(u_x+i, u_y)
            dis2cam_u_alongx = dis2cam_u_alongx + \
                frame.get_distance(u_x-i, u_y)
        # along x axis
        for i in range(1, 3):
            dis2cam_u_alongy = dis2cam_u_alongy + \
                frame.get_distance(u_x, u_y+i)
            dis2cam_u_alongy = dis2cam_u_alongy + \
                frame.get_distance(u_x, u_y-i)

        sum_depth = dis2cam_u_alongx + dis2cam_u_alongy + dis2cam_u

        depth_average = sum_depth / 9
        # Convert pixels to 3D coordinates in camera frame(deprojection)
        depth_intrin = frame.profile.as_video_stream_profile().intrinsics
        u_pos = rs.rs2_deproject_pixel_to_point(
            depth_intrin, [u_x, u_y], depth_average)

        return u_pos

    # Distance computation through pixels
    def distance_pixel(self, frame, u, v):

        # Copy pixels into the arrays (to match rsutil signatures)
        u_x = u[0]
        u_y = u[1]
        v_x = v[0]
        v_y = v[1]
        # Get depth from pixels
        dis2cam_u = frame.get_distance(u_x, u_y)
        dis2cam_v = frame.get_distance(v_x, v_y)
        # Convert pixels to 3D coordinates in camera frame(deprojection)
        depth_intrin = frame.profile.as_video_stream_profile().intrinsics
        u_pos = rs.rs2_deproject_pixel_to_point(
            depth_intrin, [u_x, u_y], dis2cam_u)
        v_pos = rs.rs2_deproject_pixel_to_point(
            depth_intrin, [v_x, v_y], dis2cam_v)

        # Calculate distance between two points
        dis_obj2target = np.sqrt(
            pow(u_pos[0]-v_pos[0], 2)+pow(u_pos[1]-v_pos[1], 2)+pow(u_pos[2]-v_pos[2], 2))

        return dis_obj2target

    # Distance computation through 3d points
    def distance_3dpoints(self, u, v):

        dis_obj2target = np.sqrt(
            pow(u[0]-v[0], 2)+pow(u[1]-v[1], 2)+pow(u[2]-v[2], 2))

        return dis_obj2target

    def shutdown(self):
        self.running = False
        time.sleep(0.1)
        if self.pipeline is not None:
            self.pipeline.stop()


if __name__ == '__main__':
    camera_handler = CameraHandler.getInstance((128, 128))

    # ch.start_pipeline()
    t = threading.Thread(name='display', target=camera_handler.start_pipeline)
    t.start()
    # time needed for camera to warm up to continue getting frames (When running the camera in the background)
    time.sleep(2)
    a, b, c = camera_handler.get_targetmarkers()
    print(a, b, c)

    d, e, f = camera_handler.get_current_obj()
    print(d, e, f)

    # while True:
    #     color_frame = camera_handler.color_frame
    #     reward_frame = camera_handler.reward_frame
    #     cv2.imshow('Observation',color_frame)
    #     cv2.imshow('Reward', reward_frame)
    #     key = cv2.waitKey(1)
    #     if key & 0xFF == ord('q') or key == 27:
    #         cv2.destroyAllWindows()
    #         break

# plot
    # count = 2000
    # dis = []
    # while(count != 0):
    #     time.sleep(0.01)
    #     print(camera_handler.grab_distance())
    #     # print(ch.get_current_obj())
    #     dis.append(camera_handler.grab_distance())
    #     count = count - 1

    # data_size = len(dis)
    # axis = np.arange(0, 2000, 1)
    # lablesize = 18
    # fontsize = 16
    # plt.plot(axis, dis, color="steelblue", linewidth=1.0, label='distance')
    # plt.xlabel('Count', fontsize=lablesize)
    # plt.ylabel('Distance[m]', fontsize=lablesize)
    # # plt.xticks(fontsize=fontsize)
    # # plt.yticks(fontsize=fontsize)
    # # plt.legend(loc='lower right',fontsize=18)
    # plt.grid(ls='--')
    # plt.show()

    # test average time to get distance
    # count = 0
    # sumss = 0
    # while count<1000:
    #     time.sleep(0.005)
    #     a=0
    #     start= time.time()
    #     while (a==0):
    #         a=ch.get_distance()
    #         end= time.time()
    #     print(end-start)
    #     sumss+=end-start
    #     count=count+1
    # print (count)
    # print (sumss)

    # Show distance in cv window
    # cv2.namedWindow('update', cv2.WINDOW_AUTOSIZE)
    # while True:
    #     a = ch.markerprocess()
    #     if a is not None:
    #         cv2.imshow('update',a)
    #     key = cv2.waitKey(1)
    #     if key & 0xFF == ord('q') or key == 27:
    #         cv2.destroyAllWindows()
    #         break
