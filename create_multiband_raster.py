import os
from osgeo import gdal
from collections import defaultdict
from tondortools.tool import save_raster_template
from gdalconst import GDT_Float32
from pathlib import Path
######################################################
amazonas_root_folder = Path("/mnt/ssdarchive.nfs/amazonas_dir")
tile = "21LYG"
######################################################
archive_folder = amazonas_root_folder.joinpath('output')
archive_mosaic_folder = archive_folder.joinpath("Mosaic")
archive_mosaic_tile_folder = archive_mosaic_folder.joinpath(tile)

archive_multiband_mosaic_folder = archive_folder.joinpath("Multiband_mosaic")
os.makedirs(archive_multiband_mosaic_folder, exist_ok=True)
archive_multiband_mosaic_tile_folder = archive_multiband_mosaic_folder.joinpath(tile)
os.makedirs(archive_multiband_mosaic_tile_folder,exist_ok=True)
######################################################

orbit_directions = os.listdir(archive_mosaic_tile_folder)
for orbit_direction in orbit_directions:

    archive_multiband_mosaic_tile_orbit_folder = archive_multiband_mosaic_tile_folder.joinpath(orbit_direction)
    os.makedirs(archive_multiband_mosaic_tile_orbit_folder, exist_ok=True)

    # Folder path
    folder_path = archive_mosaic_tile_folder.joinpath(orbit_direction)

    # Group files by date
    file_groups = defaultdict(dict)

    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.tif'):
            continue

        parts = file_name.split('_')
        date = parts[3]
        polarization = parts[2]

        file_groups[date][polarization] = os.path.join(folder_path, file_name)

    # Process each group
    for date, files in file_groups.items():
        if "VV" in files and "VH" in files:
            vh_file = files["VH"]
            vv_file = files["VV"]

            # Calculate difference
            vh_raster = gdal.Open(vh_file, gdal.GA_ReadOnly)
            vv_raster = gdal.Open(vv_file, gdal.GA_ReadOnly)

            vh_band = vh_raster.GetRasterBand(1).ReadAsArray()
            vv_band = vv_raster.GetRasterBand(1).ReadAsArray()

            difference = vh_band - vv_band

            diff_file = os.path.join(folder_path, f"21LYG_BAC_DIFF_{date}")
            save_raster_template(vv_file, diff_file, difference, GDT_Float32, vh_raster.GetRasterBand(1).GetNoDataValue())

            # Merge to a three-band raster
            merged_file = os.path.join(archive_multiband_mosaic_tile_orbit_folder, f"21LYG_BAC_MERGED_{date}")
            merge_command = f"gdal_merge.py -separate -o {merged_file} {vh_file} {vv_file} {diff_file}"
            os.system(merge_command)