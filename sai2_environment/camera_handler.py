import cv2
import numpy as np
import pyrealsense2 as rs

class CameraHandler:

    __instance = None

    @staticmethod 
    def getInstance(resolution):
        """ Static access method. """
        if CameraHandler.__instance == None:
            CameraHandler(resolution)
        return CameraHandler.__instance

    def __init__(self, resolution):
        if CameraHandler.__instance != None:
            raise Exception("This class: CameraHandler is a singleton!")
        else:
            CameraHandler.__instance = self
            self.pipeline = rs.pipeline()
            self.__color_frame = None
            self.__depth_frame = None
            self.__resolution = resolution

    def get_color_frame(self):
        return self.__color_frame

    def get_depth_frame(self):
        return self.__depth_frame

    def start_pipeline(self):
        self.pipeline.start()
        align_to = rs.stream.color
        align = rs.align(align_to)
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            self.__depth_frame = np.asanyarray(depth_frame.get_data())
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            self.__color_frame = cv2.resize(color_frame, self.__resolution)


if __name__ == '__main__':
    ch = CameraHandler.getInstance((128,128))
    ch.start_pipeline()
