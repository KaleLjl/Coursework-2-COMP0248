import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Output paths
base_dir = "./data/RealSense_TEST4"
path_color = os.path.join(base_dir, "image")
path_depth = os.path.join(base_dir, "depth")
os.makedirs(path_color, exist_ok=True)
os.makedirs(path_depth, exist_ok=True)

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# # Set color sensor
# color_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
# color_sensor.set_option(rs.option.exposure, 1000)
# color_sensor.set_option(rs.option.gain, 32)

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

frame_count = 0
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if frame_count == 0:
            # Get depth sensor intrinsics
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            print("Depth Intrinsics: ", depth_intrinsics)
            K = np.array([
                [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0, 0, 1]
            ])
            np.savetxt(os.path.join(base_dir, "intrinsics.txt"), K, fmt="%.6f")
            print("Saved camera intrinsics.")
            
        cv2.imwrite("%s/%06d.png" % \
                        (path_depth, frame_count), depth_image)
        cv2.imwrite("%s/%06d.jpg" % \
                        (path_color, frame_count), color_image)
        print("Saved color + depth image %06d" % frame_count)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), 
                                           cv2.COLORMAP_JET)
        
        cv2.imshow('RealSense Depth Stream', depth_colormap)
        cv2.imshow('RealSense Color Stream', color_image)

        frame_count += 1
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()