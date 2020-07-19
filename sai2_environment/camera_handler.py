import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
from collections import deque
import matplotlib.pyplot as plt


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
                self.device_id = "829212070352" # observation: "828112071102"   "829212070352"robot right  "943222073921" robot_left
                self.reward_devices_id = "943222073921"
            else:
                self.device_id = device_id

            self.config = rs.config()
            self.config.enable_device(self.device_id)
            self.config.enable_stream(
                rs.stream.depth, 640, 480, rs.format.z16, 60)
            self.config.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 60)

            # Start the second camera that computes the rewards
            self.reward_pipeline = rs.pipeline()
            self.reward_config = rs.config()
            self.reward_config.enable_device(self.reward_devices_id)
            self.reward_config.enable_stream(
                rs.stream.depth, 640, 480, rs.format.z16, 60)
            self.reward_config.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 60)

            self.reward_frame = None
            self.reward_depth_frame = None
            # Choose which camera to use
            self.camera_flag = 1

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

            self.dictionary = cv2.aruco.Dictionary_get(
                cv2.aruco.DICT_ARUCO_ORIGINAL)

            # Initialize the detector parameters using default values
            self.parameters = cv2.aruco.DetectorParameters_create()

    def get_color_frame(self):
        return self.color_buffer[-1]

    def get_depth_frame(self):
        return self.depth_buffer[-1]

    def grab_distance(self):
        return self.distance_buffer[-1]

    def get_current_obj(self):
        return self.obj_position

    def start_pipeline(self):
        # self.pipeline.start()
        # align_to = rs.stream.color
        # align = rs.align(align_to)
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        #start Observation Camera
        profile = self.pipeline.start(self.config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        #start Reward Camera
        reward_align_to = rs.stream.color
        self.reward_align = rs.align(reward_align_to)
        self.reward_pipeline.start(self.reward_config)


        while True:
            #Observation Camera
            frames = self.pipeline.wait_for_frames(
                200 if (self.frame_count > 1) else 10000)  # wait 10 seconds for first frame
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()

            self.depth_frame = depth_frame
            self.__depth_frame = np.asanyarray(depth_frame.get_data())
            self.depth_buffer.append(self.__depth_frame)
  

            self.color_frame = np.asanyarray(aligned_frames.get_color_frame().get_data())
            self.color_image = self.color_frame 
            self.__color_frame = cv2.resize(self.color_frame, self.__resolution)

            self.color_buffer.append(self.__color_frame)

            # Compute the distance and store them in the buffer
            distance_temp = self.cal_distance()

            if (distance_temp == 1):
                self.distance_buffer.append(self.distance_buffer[-1])

            if (distance_temp != 1):
                if (distance_temp < 0.5):
                    self.distance_buffer.append(distance_temp)
                else:
                    self.distance_buffer.append(self.distance_buffer[-1])

            #Reward Camera
            reward_frames = self.reward_pipeline.wait_for_frames(200 if (self.frame_count > 1) else 10000)
            reward_aligned_frames = self.reward_align.process(reward_frames)
            reward_color_frame = np.asanyarray(reward_aligned_frames.get_color_frame().get_data())

            self.reward_frame = reward_color_frame
            self.reward_depth_frame = reward_aligned_frames.get_depth_frame()

            if self.color_image is not None:
                cv2.imshow('Observation',self.color_image)
                cv2.imshow('Reward', self.reward_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    # Capture current frame  (like shooting a picture)
    def _capture(self):

        # get the frames
        # try:
        frames = self.pipeline.wait_for_frames(
            200 if (self.frame_count > 1) else 10000)  # wait 10 seconds for first frame
        # except Exception as e:
        #     logging.error(e)
        #     return
        #
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

    # For displaying to test the marker detection visually
    def markerprocess(self):

        # Option 1: Capture the frame each time to get a series of frame
        color_image, depth_image, depth_frame, color_frame = self._capture()

        # Option 2: Camera start_pipeline runs in the background and get frame each time
        # color_image = self.color_image
        # depth_frame = self.depth_frame
        # color_frame = self.color_frame

        # Aruco marker part
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
            color_image, self.dictionary, parameters=self.parameters)

        aruco_list = {}
        # centre= {}
        result_center = {}
        # orient_centre= {}
        if markerIds is not None:
            # Print corners and ids to the console
            # result=zip(markerIds, markerCorners)
            for k in range(len(markerCorners)):
                temp_1 = markerCorners[k]
                temp_1 = temp_1[0]
                temp_2 = markerIds[k]
                temp_2 = temp_2[0]
                aruco_list[temp_2] = temp_1
            key_list = aruco_list.keys()
            font = cv2.FONT_HERSHEY_SIMPLEX
            # print(key_list)
            for key in key_list:
                dict_entry = aruco_list[key]
                centre = dict_entry[0] + dict_entry[1] + \
                    dict_entry[2] + dict_entry[3]
                centre[:] = [int(x / 4) for x in centre]
                # orient_centre = centre + [0.0,5.0]
                centre = tuple(centre)
                result_center[key] = centre
                # orient_centre = tuple((dict_entry[0]+dict_entry[1])/2)
                cv2.circle(color_image, centre, 1, (0, 0, 255), 8)

            # Compute distance when matching the conditions
            if len(result_center) < 4:
                print("No enough marker detected")

            if len(result_center) >= 4:
                # To avoid keyerror
                # start = time.time()
                try:
                    # Moving object localization marker
                    x_id0 = result_center[0][0]
                    y_id0 = result_center[0][1]
                    p_0 = [x_id0, y_id0]

                    # Target object localization marker
                    # Single ID-5
                    x_id5 = result_center[5][0]
                    y_id5 = result_center[5][1]
                    p_5 = [x_id5, y_id5]

                    # Dual ID-4 and ID-3
                    x_id4 = result_center[4][0]
                    y_id4 = result_center[4][1]
                    p_4 = [x_id4, y_id4]

                    x_id3 = result_center[3][0]
                    y_id3 = result_center[3][1]
                    p_3 = [x_id3, y_id3]

                    # Deproject pixel to 3D point
                    point_0 = self.pixel2point(depth_frame, p_0)
                    point_5 = self.pixel2point(depth_frame, p_5)
                    point_4 = self.pixel2point(depth_frame, p_4)
                    point_3 = self.pixel2point(depth_frame, p_3)
                    # Calculate target point
                    point_target = [point_4[0]+point_3[0]-point_5[0], point_4[1] +
                                    point_3[1]-point_5[1], point_4[2]+point_3[2]-point_5[2]]
                    # Compute distance
                    # dis=distance_3dpoints(point_target,point_0)

                    # Display target and draw a line between them
                    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                    target_pixel = rs.rs2_project_point_to_pixel(
                        color_intrin, point_target)
                    target_pixel[0] = int(target_pixel[0])
                    target_pixel[1] = int(target_pixel[1])
                    cv2.circle(color_image, tuple(
                        target_pixel), 1, (0, 0, 255), 8)
                    cv2.line(color_image, tuple(p_0), tuple(
                        target_pixel), (0, 255, 0), 2)

                    # Euclidean distance
                    dis_obj2target = self.distance_3dpoints(
                        point_0, point_target)
                    dis_obj2target_goal = dis_obj2target * \
                        np.sin(np.arccos(0.02/dis_obj2target))

                    # print(dis_obj2target_goal)
                    # images = cv2.aruco.drawDetectedMarkers(images, markerCorners, borderColor=(0, 0, 255))
                    # end = time.time()
                    # print(str(end-start))

                except KeyError:
                    print("Keyerror!!!")

            # Outline all of the markers detected in our image
            # images = cv2.aruco.drawDetectedMarkers(images, markerCorners, borderColor=(0, 0, 255))
                # self.images = color_image
        self.images = cv2.aruco.drawDetectedMarkers(
            color_image, markerCorners, borderColor=(0, 0, 255))

        return self.images

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
        # markerIds = None
        # markerIds_reward = None
        markerIds_temp = None
        
        # Decide which camera frame to use
        if color_image is not None:
            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
                color_image, self.dictionary, parameters=self.parameters)
            if markerIds is not None:
                if 0 in markerIds:
                    depth_frame = self.depth_frame
                    self.camera_flag = 1
                    markerIds_temp = markerIds
                    markerCorners_temp = markerCorners       
        if reward_color is not None:
            markerCorners_reward, markerIds_reward, rejectedCandidates_reward = cv2.aruco.detectMarkers(
                reward_color, self.dictionary, parameters=self.parameters)
            if markerIds_reward is not None:
                if 0 in markerIds_reward:
                    depth_frame = self.reward_depth_frame
                    markerIds_temp = markerIds_reward
                    markerCorners_temp = markerCorners_reward
                    self.camera_flag = 2

        # print(markerIds_temp)

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
                centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]
                centre[:] = [int(x / 4) for x in centre]
                centre = tuple(centre)
                result_center[key] = centre

        # print(result_center)
        try:
            point_obj = None
            point_target = None
            if result_center.get(0) != None:
                x_id0 = result_center[0][0]
                y_id0 = result_center[0][1]
                p_0 = [x_id0, y_id0]
                # Deproject pixel to 3D point
                point_obj = self.pixel2point(depth_frame, p_0)
            if result_center.get(1) != None:
                x_id1 = result_center[1][0]
                y_id1 = result_center[1][1]
                p_1 = [x_id1, y_id1]
                # Deproject pixel to 3D point
                point_obj = self.pixel2point(depth_frame, p_1)

            if(result_center.get(5) != None):
                x_id5 = result_center[5][0]
                y_id5 = result_center[5][1]
                p_5 = [x_id5, y_id5]
                # Deproject pixel to 3D point
                point_5 = self.pixel2point(depth_frame, p_5)

            # Dual ID-4 and ID-3
            if(result_center.get(4) != None):
                x_id4 = result_center[4][0]
                y_id4 = result_center[4][1]
                p_4 = [x_id4, y_id4]
                # Deproject pixel to 3D point
                point_4 = self.pixel2point(depth_frame, p_4)

            if(result_center.get(3) != None):
                x_id3 = result_center[3][0]
                y_id3 = result_center[3][1]
                p_3 = [x_id3, y_id3]
                # Deproject pixel to 3D point
                point_3 = self.pixel2point(depth_frame, p_3)

            # Calculate target point
            if (result_center.get(5) != None and result_center.get(4) != None and result_center.get(3) != None):
                point_target = [point_4[0]+point_3[0]-point_5[0], point_4[1] +
                                point_3[1]-point_5[1], point_4[2]+point_3[2]-point_5[2]]

            
            # Transformation between two cameras
            # left_R_right = 
            # left_T_right =  
            
            if self.camera_flag == 1:
                # store the goal position
                self.goal_position = point_target
                old_goal = point_target
                
                # share the goal position with the other camera
                # self.goal_position = np.array(point_target)
                # self.goal_position_reward = R * self.goal_position + T
                
                
                # store the obj position
                self.obj_position = point_obj
                old_obj = point_obj
            
            if self.camera_flag == 2:
                # store the goal position
                self.goal_position_reward = point_target
                old_goal_reward  = point_target


                # share the goal position with the other camera
                # self.goal_position_reward = np.array(point_target)
                # self.goal_position = R.T * self.goal_position - R.T*T
                
                
                
                
                # store the obj position
                self.obj_position_reward  = point_obj
                old_obj_reward  = point_obj
            
            # print (self.goal_position_reward)


        except KeyError:
            self.obj_position = old_obj
            self.goal_position = old_goal
            self.goal_position_reward = old_goal_reward
            self.obj_position_reward = old_obj_reward

        return result_center

    def cal_distance(self):
        # TODO more robust
        start_time = time.time()
        temp = self.get_marker_position()
        if self.camera_flag == 1:
            depth_frame = self.depth_frame
            obj_position = self.obj_position
            goal_position = self.goal_position 
        elif self.camera_flag == 2:
            depth_frame = self.reward_depth_frame
            obj_position = self.goal_position_reward
            goal_position = self.goal_position_reward

        while ( obj_position== None or goal_position == None):
            temp = self.get_marker_position()
            end = time.time()
            if(end-start_time > 0.005):
                # print("No enough markers detected for distance computation")
                return 1
        old_value = 1

        
        try:
            # Moving object localization marker
            if (temp.get(0) != None):
                x_id0 = temp[0][0]
                y_id0 = temp[0][1]
                p_0 = [x_id0, y_id0]
                # Deproject pixel to 3D point
                point_0 = self.pixel2point(depth_frame, p_0)

            if (temp.get(1) != None):
                x_id1 = temp[1][0]
                y_id1 = temp[1][1]
                p_1 = [x_id1, y_id1]
                # Deproject pixel to 3D point
                point_1 = self.pixel2point(depth_frame, p_1)

            if self.camera_flag == 1:
                point_target = np.array(self.goal_position)
            if self.camera_flag == 2:
                point_target = np.array(self.goal_position_reward)

            # print (point_target)
            
            # Euclidean distance
            # dis_obj2target_goal=0
            if (temp.get(0) != None and temp.get(1) != None):
                point_temp = [(point_0[0]+point_1[0])/2,
                              (point_0[1]+point_1[1])/2, (point_0[2]+point_1[2])/2]
                # dis_obj2target_goal = self.distance_3dpoints(point_temp, point_target)
                point_temp = np.array(point_temp)
                dis_obj2target_goal = np.linalg.norm(point_target-point_temp)
            if (temp.get(0) != None and temp.get(1) == None):
                # dis_obj2target = self.distance_3dpoints(point_0, point_target)
                # dis_obj2target_goal = dis_obj2target #*np.sin(np.arccos(0.02/dis_obj2target))
                point_0 = np.array(point_0)
                dis_obj2target_goal = np.linalg.norm(point_target-point_0)
            if (temp.get(0) == None and temp.get(1) != None):
                # dis_obj2target_goal = self.distance_3dpoints(point_1, point_target)
                point_1 = np.array(point_1)
                dis_obj2target_goal = np.linalg.norm(point_target-point_1)
            
            old_value = dis_obj2target_goal
            # print (dis_obj2target_goal)
            
            return dis_obj2target_goal

        except KeyError:
            print("Keyerror!!!")
            return old_value

    def pixel2point(self, frame, u):

        u_x = u[0]
        u_y = u[1]
        # Get depth from pixels
        dis2cam_u = frame.get_distance(u_x, u_y)
        # Convert pixels to 3D coordinates in camera frame(deprojection)
        depth_intrin = frame.profile.as_video_stream_profile().intrinsics
        u_pos = rs.rs2_deproject_pixel_to_point(
            depth_intrin, [u_x, u_y], dis2cam_u)

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
    camera_handler = CameraHandler.getInstance((128,128))

    # ch.start_pipeline()
    t = threading.Thread(name='display', target=camera_handler.start_pipeline)
    t.start()
    # time needed for camera to warm up to continue getting frames (When running the camera in the background)
    time.sleep(2)

    # while True:
    #     color_frame = camera_handler.color_frame
    #     reward_frame = camera_handler.reward_frame
    #     cv2.imshow('Observation',color_frame)
    #     cv2.imshow('Reward', reward_frame)
    #     key = cv2.waitKey(1)
    #     if key & 0xFF == ord('q') or key == 27:
    #         cv2.destroyAllWindows()
    #         break

    count = 2000
    dis = []
    while(count != 0):
        time.sleep(0.001)
        print(camera_handler.grab_distance())
        # print(ch.get_current_obj())
        dis.append(camera_handler.grab_distance())
        count = count - 1

    data_size = len(dis)
    axis = np.arange(0, 2000, 1)
    lablesize = 18
    fontsize = 16
    plt.plot(axis, dis, color="steelblue", linewidth=1.0, label='distance')
    plt.xlabel('Count', fontsize=lablesize)
    plt.ylabel('Distance[m]', fontsize=lablesize)
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.legend(loc='lower right',fontsize=18)
    plt.grid(ls='--')
    plt.show()

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
