from pathlib import Path
import os, requests
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte
import numpy as np
import datetime
import subprocess

from tondortools.tool import reproject_multibandraster_toextent, read_raster_info, save_raster_template, mosaic_tifs

#######
def create_mosaic_filepath(raster_filepaths, output_path, NODATA_VALUE, aoi_epsg):
    # mosaic_rasters(gdalfile=tiles,
    #                dst_dataset=file.path(Out_folder, paste0(IPCC_class, '_Product_ISB'), "ISB_AOImasked.tif"),
    #                of="GTiff", gdalwarp_params=list(r="average", ot="Float32"),
    #                co=c("COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=9"), overwrite=TRUE, VERBOSE=TRUE)
    # GDAL Warp parameters
    raster_filepaths_str_list = [str(raster_item) for raster_item in raster_filepaths]
    gdalwarp_cmd = ['gdalwarp',
                    '-of', "GTiff",
                    "-r", "average",
                    "-ot", "Float32",
                    "-dstnodata", f"{NODATA_VALUE}",
                    "-t_srs", "EPSG:{}".format(aoi_epsg),
                    "-co", "BIGTIFF=IF_NEEDED",
                    "-co", "TILED=YES",
                    "-co", "COMPRESS=LZW",
                    "-co", "PREDICTOR=3",
                    "-co", "ZLEVEL=9",
                    "-overwrite"] + raster_filepaths_str_list + [str(output_path)]
    cmd_output = subprocess.run(gdalwarp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("exit code {} --> {}".format(cmd_output.returncode, gdalwarp_cmd))



def find_template_raster(root_folder):

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.tif'):
                return Path(foldername).joinpath(filename)


#######
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
gfw_extents_filename = "gftw_tileextents.gpkg"
s2_extents = "S2_tiles.gpkg"

ref_year = 2021
###
support_data = amazonas_root_folder.joinpath("support_data")
if not support_data.exists(): raise Exception(f"{support_data} doesnt exists.")

ref_data_folder = support_data.joinpath("ref_worldcover")
if not ref_data_folder.exists(): raise Exception(f"{ref_data_folder} doesnt exists.")

ref_data_mosaic_folder = support_data.joinpath("ref_worldcover_sarmosaic")
os.makedirs(ref_data_mosaic_folder, exist_ok=True)

work_dir = amazonas_root_folder.joinpath("work_dir", "tmp")
os.makedirs(work_dir, exist_ok=True)



archive_folder = amazonas_root_folder.joinpath("output")
archive_mosaic_folder = archive_folder.joinpath("Multiband_mosaic_from_2020")




tile_map_file = support_data.joinpath('tile_map.json')
tile_map_file_open = open(tile_map_file)
tile_map = json.load(tile_map_file_open)


for tile_item, big_tiles_list in tile_map.items():
    print(f"Tile: {tile_item} - {big_tiles_list}")
    archive_mosaic_tile_folder = archive_mosaic_folder.joinpath(tile_item)

    archive_mosaic_tile_orbits = os.listdir(archive_mosaic_tile_folder)

    tif_to_merge = []
    template_tif = None
    for archive_mosaic_tile_orbit_item in archive_mosaic_tile_orbits:
        archive_mosaic_tile_orbits_folder = archive_mosaic_tile_folder.joinpath(archive_mosaic_tile_orbit_item)

        backscatter_tifs = os.listdir(archive_mosaic_tile_orbits_folder)

        for backscatter_tif_item in sorted(backscatter_tifs):
            if backscatter_tif_item.endswith('.tif'):
                timestamp_tif =  backscatter_tif_item.split("_")[-1].split(".")[0]
                timestamp_year_tif =  int(timestamp_tif[0:4])
                if timestamp_year_tif == ref_year:
                    raster_path = archive_mosaic_tile_orbits_folder.joinpath(backscatter_tif_item)
                    tif_to_merge.append(raster_path)
                    template_tif = raster_path

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, bbox) = read_raster_info(
        raster_path)

    tile_merged_tif_path = ref_data_mosaic_folder.joinpath(f"{tile_item}_BAC_MERGED_{ref_year}.tif")
    ds_tif = gdal.Open(str(raster_path))
    ds_nodata = ds_tif.GetRasterBand(1).GetNoDataValue()
    print(f" tifs to merge: {tif_to_merge}")
    create_mosaic_filepath(tif_to_merge, tile_merged_tif_path, ds_nodata, epsg)







