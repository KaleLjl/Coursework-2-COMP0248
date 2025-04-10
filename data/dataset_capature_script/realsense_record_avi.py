import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Output paths
base_dir = "./data/RealSense_video_TEST"
path_color = os.path.join(base_dir, 'RealSense_rgb.avi')
path_depth = os.path.join(base_dir, 'RealSense_depth.avi')
os.makedirs(base_dir, exist_ok=True)

color_writer = cv2.VideoWriter(path_color, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
depth_writer = cv2.VideoWriter(path_depth, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)

profile = pipeline.start(config)

frame_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        #convert images to numpy arrays
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

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), 
                                           cv2.COLORMAP_JET)
        
        color_writer.write(color_image)
        depth_writer.write(depth_image)
        
        cv2.imshow('Stream_depth', depth_colormap)
        cv2.imshow('Stream_color', color_image)

        frame_count += 1
        
        if cv2.waitKey(1) == ord("q"):
            break
finally:
    color_writer.release()
    depth_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Recording complete. Files saved.")