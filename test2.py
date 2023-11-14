import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os
import sys

class Realsense2:
    def __init__(self, camera_id_list = [0], camera_width=1280, camera_height=720, camera_fps=30):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.camera_id_list = camera_id_list
        

    def camera_config(self):
        self.connect_device = []
        # get all realsense serial number
        for d in rs.context().devices:
            print('Found device: ',
                d.get_info(rs.camera_info.name), ' ',
                d.get_info(rs.camera_info.serial_number))
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                self.connect_device.append(d.get_info(rs.camera_info.serial_number))
        # config realsense
        self.pipeline_list = [rs.pipeline() for i in range(len(self.camera_id_list))]
        self.config_list = [rs.config() for i in range(len(self.camera_id_list))]
        if len(self.camera_id_list) == 1: # one realsense
                self.config_list[0].enable_device(self.connect_device[0])
                self.config_list[0].enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.camera_fps)
                self.config_list[0].enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.camera_fps)
                self.pipeline_list[0].start(self.config_list[0])
        elif len(self.camera_id_list) >= 2: # two realsense
            if len(self.connect_device) < 2:
                print('Registrition needs two camera connected.But got one.Please run realsense-viewer to check your camera status.')
                exit()
            # enable config
            for n, config in enumerate(self.config_list):
                config.enable_device(self.connect_device[n])
                config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.camera_fps)
                config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.camera_fps)
                # self.config_list[n] = config
            # start pipeline
            for n, pipeline in enumerate(self.pipeline_list):
                pipeline.start(self.config_list[n])

    def wait_frames(self, frame_id=None):
        '''
        camera_id:
            @ = camera number , get all frame
            @ = id , get specific id camera frame 
        '''
        self.frame_list = [None for i in range(len(self.camera_id_list))]
        if frame_id != None: # give a frame id
            self.frame_list[n] = self.pipeline_list[frame_id].wait_for_frames()
        else: # get all frame
            if len(self.camera_id_list) == 1:
                self.frame_list.append(self.pipeline_list[0].wait_for_frames())
            else:
                for n, camera_id in enumerate(self.camera_id_list):
                    self.frame_list[n] = self.pipeline_list[n].wait_for_frames()

    def rgb_image(self, camera_id=0):
        color_frame = self.frame_list[camera_id].get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def depth_frame(self, frames):
        depth_frame = frames.get_depth_frame()
        return depth_frame

    def stop(self):
        for pipeline in self.pipeline_list:
            pipeline.stop()
        print("camera exit sucessfully.")

if __name__ == '__main__':
    cap = Realsense2(camera_id_list=[0,1], camera_width=640, camera_height=480) # 
    cap.camera_config()
    while True:
        start = time.time()
        cap.wait_frames()
        img0 = cap.rgb_image(0)
        img1 = cap.rgb_image(1)
        cv2.imshow("img0", img0)
        cv2.imshow("img1", img1)
        print("FPS:", 1/(time.time()-start), "f/s")
        if cv2.waitKey(1) == ord("q"):
            break

    cap.stop()
