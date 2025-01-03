import os

import pystac
from pystac.extensions.eo import Band, EOExtension
from shapely.geometry import Polygon, mapping
from datetime import datetime
from pathlib import Path
import rasterio
from shapely.geometry import shape

def file_datetime_list(folder_path):
    rasterfile_list = os.listdir(folder_path)
    datetime_list = []
    for rasterfile_list_item in rasterfile_list:
        if not rasterfile_list_item.endswith("tif"):
            continue
        datetime_str = rasterfile_list_item.split("_")[3]
        datetime_list.append(datetime_str)
    return datetime_list

# Function to Extract BBox and Footprint from S3 Asset
def get_bbox_and_footprint(s3_url_item):
    with rasterio.open(s3_url_item) as r:
        bounds = r.bounds
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        footprint = Polygon([
            [bounds.left, bounds.bottom],
            [bounds.left, bounds.top],
            [bounds.right, bounds.top],
            [bounds.right, bounds.bottom]
        ])
        return bbox, mapping(footprint)

