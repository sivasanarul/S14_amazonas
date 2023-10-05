from pathlib import Path
import os, requests
import subprocess
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte
import numpy as np
from datetime import datetime, timedelta, date
import math
from tondortools.tool import read_raster_info
import copy

tile = "21LYG"
year = [2019, 2020]
time_window = 120
acq_freq = 12
number_bands = 3
count_threshold = 10000
block_size = 256
min_labelpixel = 400
amazonas_root_folder = Path("/mnt/ssdarchive.nfs/amazonas_dir")
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
output_folder_multiband_mosaic = output_folder.joinpath('Multiband_mosaic')
output_folder_multiband_mosaic_tile = output_folder_multiband_mosaic.joinpath(tile)
output_folder_multiband_mosaic_tile_orbit = output_folder_multiband_mosaic_tile.joinpath('descending')
########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir")
support_data = amazonas_root_folder.joinpath("support_data")
gfw_folder_root = amazonas_root_folder.joinpath("Detections", "GFW")
gfw_data = work_dir.joinpath("gfw_data")
########################################################################################################################
gfw_folder_tile_folder = gfw_folder_root.joinpath(tile)
gfw_folder_tile_raster = gfw_folder_tile_folder.joinpath(f"{tile}.tif")
gfw_folder_tile_folder_label = gfw_folder_tile_folder.joinpath("label")

gfw_folder_tile_folder_renamed_label = gfw_folder_tile_folder.joinpath("label_renamed")
########################################################################################################################
########################################################################################################################
########################################################################################################################
training_folder = amazonas_root_folder.joinpath('training')
os.makedirs(training_folder, exist_ok=True)
training_folder_label = training_folder.joinpath('label')
os.makedirs(training_folder_label, exist_ok=True)
training_folder_data = training_folder.joinpath('data')
os.makedirs(training_folder_data, exist_ok=True)

mosaic_files = os.listdir(output_folder_multiband_mosaic_tile_orbit)
gfw_mosaic_pair = dict()
for year_item in year:
    print(f"-- {year_item} --")
    gfw_files = os.listdir(gfw_folder_tile_folder_renamed_label)
    for gfw_file in sorted(gfw_files):
        gfw_filepath = gfw_folder_tile_folder_renamed_label.joinpath(gfw_file)
        gfw_file_count = int(gfw_file.split('_')[-1].split('.')[0])
        if f"_{year_item}-" in gfw_file and gfw_file_count > count_threshold:
            mosaic_file_list = []
            gfw_file_date_str = gfw_file.split('_')[1]
            acq_date = datetime.strptime(gfw_file_date_str, "%Y-%m-%d")
            for mosaic_file in mosaic_files:
                mosaic_date_str = mosaic_file.split('_')[-1].split('.')[0]
                mosaic_date = datetime.strptime(mosaic_date_str, "%Y%m%d")
                if (mosaic_date <= acq_date) and (mosaic_date > acq_date - timedelta(days=time_window)):
                    mosaic_file_list.append(output_folder_multiband_mosaic_tile_orbit.joinpath(mosaic_file))
            print(f"{gfw_file} : {mosaic_file_list}")
            print("-----")
            gfw_mosaic_pair[gfw_filepath] = sorted(mosaic_file_list)

print("---------------------")

number_rasters_timewindow = math.floor(time_window/acq_freq)
total_bands = number_rasters_timewindow*number_bands

count = 1
for gfw_file, raster_list in gfw_mosaic_pair.items():
    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(raster_list[0])

    for x_block in range(0, RasterXSize+1, block_size):
        for y_block in range(0, RasterYSize+1, block_size):

            chunk_list = []

            array_bands = np.zeros((block_size, block_size))
            save_np = True

            if x_block + block_size < RasterXSize:
                cols = block_size
            else:
                cols = RasterXSize - x_block
                save_np = False

            if y_block + block_size < RasterYSize:
                rows = block_size
            else:
                rows = RasterYSize - y_block
                save_np = False

            for raster_list_item in raster_list:
                dataset = gdal.Open(str(raster_list_item))
                chunk = dataset.ReadAsArray(x_block, y_block, cols, rows)

                # Transpose the data
                transposed_chunk = chunk.transpose(1, 2, 0)
                chunk_list.append(transposed_chunk)

            label_dataset = gdal.Open(str(gfw_file))
            label_chunk = label_dataset.ReadAsArray(x_block, y_block, cols, rows)
            chunk_list_array = np.array(chunk_list)

            num_non_zero = np.count_nonzero(label_chunk)
            if not num_non_zero > min_labelpixel:
                save_np = False
            label_mask = copy.copy(label_chunk)
            label_mask[label_mask>0] = 1

            if save_np:
                training_folder_data_npy = training_folder_data.joinpath(f"data_{count}.npy")
                np.save(training_folder_data_npy, chunk_list_array)

                training_folder_label_npy = training_folder_label.joinpath(f"label_{count}.npy")
                np.save(training_folder_label_npy, label_mask)

                count += 1
                print(f" {count} with {num_non_zero} {Path(gfw_file).name}--")


