import cv2
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np

# Directory containing the raster images
raster_dir = '/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/ver7_Segmod'
raster_files = [
    '21LYG_BAC_MERGED_20210505.tif', '21LYG_BAC_MERGED_20210517.tif',
    '21LYG_BAC_MERGED_20210529.tif', '21LYG_BAC_MERGED_20210610.tif',
    '21LYG_BAC_MERGED_20210622.tif', '21LYG_BAC_MERGED_20210704.tif',
    '21LYG_BAC_MERGED_20210716.tif', '21LYG_BAC_MERGED_20210728.tif'
]

# Video output settings
video_name = 'output_video2.avi'
fps = 1  # Frames per second

# Initialize video writer with H.264 codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = None
for file in raster_files:
    # Extract date from filename
    date = file.split('_')[3][:8]

    # Read the raster file
    with rasterio.open(os.path.join(raster_dir, file)) as src:
        raster = src.read(1)

    # Normalize and scale the raster data for visualization
    raster = raster.astype('float32')
    raster -= raster.min()
    raster /= raster.max()
    raster *= 255.0
    raster = raster.astype('uint8')

    # Convert to a format compatible with OpenCV
    raster_rgb = cv2.cvtColor(raster, cv2.COLOR_GRAY2RGB)

    # Create a matplotlib plot with adjusted layout
    fig, ax = plt.subplots()
    ax.imshow(raster_rgb)
    ax.set_title(date, fontsize=10)  # Adjust font size as needed
    ax.axis('off')

    # Adjust the layout to remove white space
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save the plot to a temporary file
    frame_filename = 'temp_frame.png'
    plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Read the image using OpenCV
    frame = cv2.imread(frame_filename)

    # Initialize video writer with the frame size from the first image
    if video is None:
        frame_size = (frame.shape[1], frame.shape[0])
        video = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

    video.write(frame)

    # Remove the temporary file
    os.remove(frame_filename)

# Release the video writer
video.release()