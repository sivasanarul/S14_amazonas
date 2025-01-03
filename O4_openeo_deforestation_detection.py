import openeo
import json
from pathlib import Path
import openeo.processes as eop
from openeo.processes import array_element, subtract, array_create, rename_labels


# Connect to the openEO backend (e.g., VITO backend)
# You can replace this URL with the specific backend you're using
# Setup the connection
connection = openeo.connect("openeo.dataspace.copernicus.eu")
# connection.authenticate_oidc().authenticate_basic("grasslandwatch@gmail.com", "Productionenvironment1234!")
connection.authenticate_oidc_device()
# Authenticate (if needed, this could be skipped if not required)
# connection.authenticate_oidc()

# Define the area of interest (AOI) using a bounding box (or GeoJSON if available)
# Example bounding box for a region (replace with your desired area)
aoi = {
    "west": -55.1653,
    "east": -54.1585,
    "north": -11.7537,
    "south": -12.7371,
    "crs": "EPSG:4326"
}
# -55.059959007,-54.881706667,-12.654733931,-12.447807500
# aoi = {
#     "west": -55.059959007,
#     "east": -54.881706667,
#     "north": -12.447807500,
#     "south": -12.654733931,
#     "crs": "EPSG:4326"
# }

# -55.16530653244335,
# -12.737140243568525,
# -54.15849189244335,
# -11.753681403568525

# Load the DEM collection from the STAC catalog (e.g., "DEM_aspec_30m")
# You can browse collections at https://radiantearth.github.io/stac-browser/#/external/stac.openeo.vito.be/
# collection_id = "DEM_slope_30m"  # Change this to the appropriate collection if needed
# datacube = connection.load_stac(
#     url="https://stac.openeo.vito.be/collections/DEM_slope_10m",
#     spatial_extent=aoi,
#     temporal_extent=None,  # Set this if you want data for a specific time range
#     bands=["SLP10"]  # Specify the band(s) if necessary
# )

url = "https://s3.waw3-1.cloudferro.com/swift/v1/gisat-archive/SAR/21LYG/21LYG_catalog.json"
bands = ["VV", "VH"]
resample_method = "cubicspline"

temporal_extent = [f"2021-01-04", f"2021-04-24"]
# temporal_extent = [f"2021-05-04", f"2021-08-22"]
# temporal_extent = [f"2021-01-04", f"2021-01-06"]

stac_item = connection.load_stac(
    url=url,
    spatial_extent=aoi,
    temporal_extent=temporal_extent,
    bands=bands
)

# pheno_datacube = pheno.reduce_temporal('last')
# Apply nearest-neighbor resampling to avoid resolution changes
datacube = stac_item.resample_spatial(resolution=20, method=resample_method)


# Reduce to a single array by collapsing the time dimension into a single stack
datacube_time_as_bands = datacube.apply_dimension(
    dimension='t',
    target_dimension='bands',
    process=lambda d: eop.array_create(data=d)
)
band_names = [band + "_t" + str(i+1).zfill(2) for band in ["VV", "VH"] for i in range(10)]
print(f"band names {band_names}")
datacube_time_as_bands = datacube_time_as_bands.rename_labels('bands', band_names)



# job = datacube_time_as_bands.create_job(out_format="netCDF") #, job_options=job_options)
# job.start_and_wait()
# results = job.get_results()
# results.download_file("result_udf.nc")

# Load the UDF from a file.
udf = openeo.UDF.from_file(Path(__file__).parent.resolve() / "O5_udf_deforestation_detection.py")

# Apply the UDF to the data cube.
datacube_udf = datacube_time_as_bands.apply_neighborhood(
    process=udf,
    size=[
        {"dimension": "x", "value": 400, "unit": "px"},
        {"dimension": "y", "value": 400, "unit": "px"},
    ],
    overlap=[
        {"dimension": "x", "value": 0, "unit": "px"},
        {"dimension": "y", "value": 0, "unit": "px"},
    ])


# Optionally, define additional operations on the datacube, like rescaling or masking
# For now, we just download the DEM data as is


job_options = {
    "executor-memory": "2500m",
    "executor-memoryOverhead": "2G",
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "soft-errors": True,
    "max_executors": 15
}

# Save the result as a GeoTIFF file
job = datacube_udf.create_job(out_format="GTiff") #, job_options=job_options)
job.start_and_wait()
results = job.get_results()
results.download_file(f"openeo_detection_21LYG_20210306_sieved12_count3_{resample_method}.tiff")
