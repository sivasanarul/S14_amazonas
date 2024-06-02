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

def create_merged_filepath(output_path, raster1, raster2, bounds, pixel_width, width, height):
    output_inter_path = output_path.parent.joinpath(output_path.name.replace(".tif", "_inter.tif"))

    args = ["gdal_merge.py",
            "-separate",
            "-o", str(output_inter_path),
            "-ps", str(pixel_width), str(pixel_width),
            "-ul_lr", str(bounds[0]), str(bounds[3]), str(bounds[2]), str(bounds[1]),
            "-of", "GTiff",
            "-ot", "Float32",
            "-co", "compress=DEFLATE"]
    args = args + [str(raster1)] + [str(raster2)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")

    args = ["gdalwarp",
            "-ts", str(width), str(height),
            "-of", "GTiff",
            "-co", "compress=DEFLATE"]
    args = args + [str(output_inter_path)] + [str(output_path)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")




tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
print(len(tiles))

cutoff_prob = 0.2
model_version = 'ver7_Segmod'
use_mcd_version_base = True
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
baselc_folder = support_data.joinpath(f"base_worldcover")
########################################################################################################################

for tile_item in tiles:
    detection_folder_aiversion = detection_folder_aiversion_parent.joinpath(tile_item)
    detection_folder_aiversion_reclassified = detection_folder_aiversion.joinpath("reclassified")
    os.makedirs(detection_folder_aiversion_reclassified, exist_ok=True)
    # / mnt / hddarchive.nfs / amazonas_dir / support_data / base_worldcover_prediction / 18L
    # VQ_BASELC_2017_CLASS.tif

    if not use_mcd_version_base:
        base_lc_tile_filepath = predicted_baselc_folder.joinpath(f"{tile_item}_BASELC_2017_CLASS.tif")
        if not base_lc_tile_filepath.exists(): raise Exception(f"{base_lc_tile_filepath} does not exist")
    else:
        base_lc_tile_filepath = baselc_folder.joinpath(f"{tile_item}_MASK.tif")
        if not base_lc_tile_filepath.exists(): raise Exception(f"{base_lc_tile_filepath} does not exist")

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(base_lc_tile_filepath)

    dataset = gdal.Open(str(base_lc_tile_filepath))
    base_lc = dataset.GetRasterBand(1).ReadAsArray()
    base_deforestation = 1 - base_lc

    folder_files_list = os.listdir(detection_folder_aiversion)
    for folder_files_list_item in sorted(folder_files_list):
        detection_path = detection_folder_aiversion.joinpath(folder_files_list_item)

        if not "CLASS.tif" in folder_files_list_item: continue
        file_year = int(folder_files_list_item.split('_')[1][0:4])
        if not file_year > 2017: continue

        merged_detection_baselc_path = work_dir.joinpath(f"merged_{folder_files_list_item}")
        create_merged_filepath(merged_detection_baselc_path, base_lc_tile_filepath, detection_path, imagery_extent_box.bounds, pixel_width, RasterXSize, RasterYSize)


        dataset = gdal.Open(str(merged_detection_baselc_path))
        detected_deforestation = dataset.GetRasterBand(2).ReadAsArray()
        detected_change_baselc_removed = (detected_deforestation.astype(np.int32)) & (~base_deforestation.astype(np.int32))
        base_deforestation = base_deforestation + detected_change_baselc_removed

        detection_folder_aiversion_reclassified_path = detection_folder_aiversion_reclassified.joinpath(folder_files_list_item)
        save_raster_template(merged_detection_baselc_path, detection_folder_aiversion_reclassified_path, detected_change_baselc_removed, data_type=GDT_Byte)
        print(detection_folder_aiversion_reclassified_path)
        merged_detection_baselc_path.unlink(missing_ok=True)



