import shutil
from pathlib import Path
import os, requests
import subprocess
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte, GDT_Float32
import numpy as np
from datetime import datetime, timedelta, date
import math
from tondortools.tool import read_raster_info, save_raster, mosaic_tifs, save_raster_template

def create_merged_filepath(output_path, raster1, raster2):
    args = ["gdal_merge.py",
            "-separate",
            "-o", str(output_path),
            "-of", "GTiff",
            "-ot", "Float32",
            "-co", "compress=DEFLATE"]
    args = args + [str(raster1)] + [str(raster2)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")


def create_sieved(result_filepath_for_sieve, result_filepath_sieved, sieve=10):
    sieve_args = ["gdal_sieve.py", #"-mask", str(mask_tif),
                  "-4", "-nomask",
                  "-st", str(sieve),
                  str(result_filepath_for_sieve), str(result_filepath_sieved)]
    cmd_output = subprocess.run(sieve_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("exit code {} --> {}".format(cmd_output.returncode, sieve_args))

    intermediate_path = result_filepath_sieved.parent.joinpath(result_filepath_sieved.name.replace(".tif", "_inter.tif"))
    shutil.copy(result_filepath_sieved, intermediate_path)
    result_filepath_sieved.unlink()

    raster_ds = gdal.Open(str(intermediate_path))
    raster_array = raster_ds.GetRasterBand(1).ReadAsArray()
    save_raster_template(intermediate_path, result_filepath_sieved, raster_array, data_type=GDT_Byte,)
    intermediate_path.unlink()

tiles = ['18LVQ',  '18LVR',  '20LLQ',  '18LWR',  '18NYH',  '18NXH',  '20LLP',  '18NXG',  '20LMQ',  '20NQG']
tiles = ['18LVQ',  '18LVR',  '20LLQ',  '18LWR',  '18NYH',  '18NXH',  '20LLP',  '18NXG',  '20LMP',  '20LMQ',  '20NQG']
tiles = ['20LMP']

cutoff_prob = 0.2
model_version = 'ver7_Segmod'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir", "reclassification")
os.makedirs(work_dir, exist_ok=True)
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion_parent = detection_folder.joinpath(f'{model_version}')
########################################################################################################################
support_data = amazonas_root_folder.joinpath("support_data")
merged_sar_ref_folder = support_data.joinpath(f"ref_worldcover_sarmosaic")
predicted_baselc_folder = support_data.joinpath(f"base_worldcover_prediction")
########################################################################################################################

detection_folder_aiversion_reclassified = detection_folder_aiversion_parent.joinpath("reclassified")
detection_folder_aiversion_reclassified_sieved = detection_folder_aiversion_parent.joinpath("sieved")
os.makedirs(detection_folder_aiversion_reclassified_sieved, exist_ok=True)

for tile_item in tiles:
    detection_folder_aiversion = detection_folder_aiversion_parent.joinpath(tile_item, "reclassified")
    detection_folder_aiversion_reclassified_sieved_tile = detection_folder_aiversion_reclassified_sieved.joinpath(tile_item)
    os.makedirs(detection_folder_aiversion_reclassified_sieved_tile, exist_ok=True)

    raster_files = os.listdir(detection_folder_aiversion)
    for raster_file_item in sorted(raster_files):
        result_filepath_for_sieve = detection_folder_aiversion.joinpath(raster_file_item)
        result_filepath_sieved = detection_folder_aiversion_reclassified_sieved_tile.joinpath(raster_file_item)
        create_sieved(result_filepath_for_sieve, result_filepath_sieved)

