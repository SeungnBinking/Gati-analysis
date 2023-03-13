import pyzed.sl as sl
import cv2

# Initialize the camera
zed = sl.Camera()

# Create a init parameter object and set the SVO file path
init_params = sl.InitParameters(input_t=sl.InputType.SVO)
init_params.svo_input_filename = "C:/Users/msdl/Desktop/project/gvs/svo_videos/cut-video/3009/1_l_lr.svo"

# Open the camera with the specified parameters
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Error opening camera: {err}")
    exit()

# Enable body tracking
tracking_params = sl.TrackingParameters()
tracking_params.enable_body_fitting = True
tracking_params.initial_world_transform = sl.Transform()
err = zed.enable_tracking(tracking_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Error enabling body tracking: {err}")
    exit()

# Create a new view to retrieve left/right image and depth map
image_size = zed.get_camera_information().camera_resolution
image_left = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
image_right = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
depth_map = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.F32_C1)

# Start the camera and tracking
runtime_parameters = sl.RuntimeParameters()
person_id = 0  # set the ID of the person to track
while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left/right image and depth map
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        # Retrieve the body tracking data for the selected person
        tracking_data = zed.get_position_data()
        person_data = tracking_data.get_person_by_id(person_id)

        # Draw a bounding box around the selected person
        bb = person_data.bounding_box_2d
        cv2.rectangle(image_left.get_data(), (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)

        # Display the left image with bounding box
        cv2.imshow("Body Tracking", image_left.get_data())

        # Check for user input to quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

# Stop the camera and close the window
zed.disable_tracking()
zed.close()
cv2.destroyAllWindows()
