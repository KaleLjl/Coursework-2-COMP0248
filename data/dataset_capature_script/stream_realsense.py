import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

MAX_DISTANCE_M = 4

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Streaming loop
try:
    rgb_ax    = plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.tight_layout()
    depth_ax  = plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.tight_layout()
    rgb_fig   = rgb_ax.imshow(np.zeros((480,640)))
    depth_fig = depth_ax.imshow(np.zeros((480,640)),vmin=0,vmax=255)
    colorizer = rs.colorizer()

    plt.ion()
    for fid in range(150):
        # Get frameset of color and depth
        frames     = pipeline.wait_for_frames()
        color      = frames.get_color_frame()
        color_np   = np.asanyarray(color.get_data())
        depth      = frames.get_depth_frame()
        depth_np   = np.asanyarray(depth.get_data())*depth.get_units()
        depth_img  = np.maximum(np.minimum((depth_np*255/MAX_DISTANCE_M).astype(np.uint8),255),0)
        depth_cmap = np.asanyarray(colorizer.colorize(depth).get_data())
        depth_fig.set_data(depth_img)
        rgb_fig.set_data(color_np)

        plt.pause(0.05)
    plt.ioff()
    plt.close('all')
finally:
    pipeline.stop()