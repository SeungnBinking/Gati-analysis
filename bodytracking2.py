########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import time 
import math
import matplotlib.pyplot as plt


def calcurate_angle_3d(a, b, c):
    
    # Calculate dierction vector
    v1 = np.subtract(b,a)
    v2 = np.subtract(b,c)

    # Claculate dot_product 
    dot_product = np.dot(v1,v2)

    # Claculate denominator
    v1_2 = np.power(v1,2)
    v2_2 = np.power(v2,2)

    v1_2 = v1_2.tolist()
    v2_2 = v2_2.tolist()

    v1_2_s = sum(v1_2)
    v2_2_s = sum(v2_2)


    v1_2_s = math.sqrt(v1_2_s)
    v2_2_s = math.sqrt(v2_2_s)

    den_angle = v1_2_s * v2_2_s
    
    output = dot_product/den_angle
    

    radians = np.arccos(output)
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle


    return angle

def calculate_distance_3d(a, b):

    v = np.subtract(b, a)
    v_2 = np.power(v,2)
    v = v_2.tolist()
    dis_v = sum(v)
    dis_v = math.sqrt(dis_v)

    return dis_v


if __name__ == "__main__":
    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
   
    init_params.svo_real_time_mode = False
    svo_filepath= "C:/Users/msdl/Desktop/svo_videos/cut-video/2052-60/1_l_lr.svo"
    
 

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get svofile information 
    svo_inf = zed.get_svo_information()
    framerate = zed.get_camera_fps()
    totalframes = svo_inf[sl.SVO_POSITION_TOTAL_FRAME_COUNT]
    playtime = totalframes/framerate
    print(playtime)


    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_MEDIUM 
    obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
                 , display_resolution.height / camera_info.camera_resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()
    right_elbow_angle = []
    ankle_distance_point = []
    right_knee_flexion = []
    left_knee_flexion = []

    prevTime = 0

    while viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)

            # Update GL view
            viewer.update_view(image, bodies) 
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            cv2.waitKey(10)
            # Extrackt landmarks
           
            # Get coordinates
              
            for obj in bodies.object_list:
                
                neck = obj.keypoint[sl.BODY_PARTS.NECK.value]
                right_ankle = obj.keypoint[sl.BODY_PARTS.RIGHT_ANKLE.value]
                left_ankle = obj.keypoint[sl.BODY_PARTS.LEFT_ANKLE.value]

                # Right elbow angle
                right_shoulder = obj.keypoint[sl.BODY_PARTS.RIGHT_SHOULDER.value]
                right_elbow = obj.keypoint[sl.BODY_PARTS.RIGHT_ELBOW.value]
                right_wrist = obj.keypoint[sl.BODY_PARTS.RIGHT_WRIST.value]

                right_shoulder = np.array(right_shoulder)
                right_elbow = np.array(right_elbow)
                right_wrist = np.array(right_wrist)

                right_shoulder = right_shoulder*10
                right_elbow = right_elbow*10
                right_wrist = right_wrist*10

                # Right knee angle 
                right_hip = obj.keypoint[sl.BODY_PARTS.RIGHT_HIP.value]
                right_knee = obj.keypoint[sl.BODY_PARTS.RIGHT_KNEE.value]
                right_ankle = obj.keypoint[sl.BODY_PARTS.RIGHT_ANKLE.value]

                right_hip = np.array(right_hip)
                right_knee = np.array(right_knee)
                right_ankle = np.array(right_ankle)

                right_hip = right_hip*10
                right_knee = right_knee*10
                right_ankle = right_ankle*10

                # Left knee angle
                left_hip = obj.keypoint[sl.BODY_PARTS.LEFT_HIP.value]
                left_knee = obj.keypoint[sl.BODY_PARTS.LEFT_KNEE.value]
                left_ankle = obj.keypoint[sl.BODY_PARTS.LEFT_ANKLE.value]
            
                left_hip = np.array(left_hip)
                left_knee = np.array(left_knee)
                left_ankle = np.array(left_ankle)

                left_hip = left_hip*10
                left_knee = left_knee*10
                left_ankle = left_ankle*10
 


                # Calculate right hand angle
                right_hand_angle = calcurate_angle_3d(right_shoulder, right_elbow, right_wrist)
                right_hand_angle = round(right_hand_angle, 2)
                right_elbow_angle.append(right_hand_angle)

                # Calculate right knee flexion
                right_knee_angle = calcurate_angle_3d(right_hip, right_knee, right_ankle)
                right_knee_angle = 180 - right_knee_angle
                right_knee_angle = round(right_knee_angle, 2)
                right_knee_flexion.append(right_knee_angle)

                # Calculate left knee flexion
                left_knee_angle = calcurate_angle_3d(left_hip, left_knee, left_ankle)
                left_knee_angle = 180 - left_knee_angle
                left_knee_angle = round(left_knee_angle, 2)
                left_knee_flexion.append(left_knee_angle)
                 


                # The distance between two points
                distance_point = calculate_distance_3d(left_knee, right_knee)
                distance_point = distance_point/10
                distance_point = round(distance_point, 4)
                ankle_distance_point.append(distance_point)

                  
            

    viewer.exit()
    

    image.free(sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()

   