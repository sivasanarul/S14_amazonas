import openeo
from pathlib import Path
from datetime import datetime, timedelta

import openeo.processes as eop
from openeo.processes import array_element, subtract, array_create, rename_labels


def compute_differences(data):
    # Compute differences between corresponding bands (1-7, 2-8, etc.)
    diff_1_7 = subtract(array_element(data, index=0), array_element(data, index=5))  # VV_VH_1
    diff_2_8 = subtract(array_element(data, index=1), array_element(data, index=6))  # VV_VH_2
    diff_3_9 = subtract(array_element(data, index=2), array_element(data, index=7))  # VV_VH_3
    diff_4_10 = subtract(array_element(data, index=3), array_element(data, index=8))  # VV_VH_4
    diff_5_11 = subtract(array_element(data, index=4), array_element(data, index=9))  # VV_VH_5

    # Create array of differences
    differences = array_create(data=[diff_1_7, diff_2_8, diff_3_9, diff_4_10, diff_5_11])

    # Rename the bands in the differences
    renamed_differences = rename_labels(
        data=differences,
        dimension="bands",
        target=["VV_VH_1", "VV_VH_2", "VV_VH_3", "VV_VH_4", "VV_VH_5"]
    )

    return renamed_differences


def timesteps_as_bands(base_features):
    # Dynamically generate band names for each timestep
    band_names = [band + "_m" + str(i + 1) for band in base_features.metadata.band_names for i in range(5)]

    # Apply dimension transformation: move time dimension to bands
    result = base_features.apply_dimension(
        dimension='t',
        target_dimension='bands',
        process=lambda d: eop.array_create(data=d)
    )

    #return result  # Optionally, you can rename the labels as shown below
    return result.rename_labels('bands', band_names)


# Test parameters
north = -12.640966421
south = -12.675937110
west = -55.094750623
east = -55.061511236
bbox = {"west": round(west, 2), "south": round(south, 2), "east": round(east, 2), "north": round(north, 2), 'crs': 4326}
base_output_path = Path("s1datacube")
base_output_path.mkdir(exist_ok=True)

# Connect and authenticate to openEO back-end
c = openeo.connect("openeo.dataspace.copernicus.eu")
c.authenticate_oidc()

def connection():
    return c

# Temporal extent
temporal_extent = ["2022-06-04", "2022-08-04"]

# Create a list of 5 intervals of 12 days each, going backwards from the second date
intervals = []
second_date = datetime.strptime(temporal_extent[1], "%Y-%m-%d")
current_date = second_date

for _ in range(5):
    start_date = current_date - timedelta(days=12)
    intervals.append([start_date.strftime("%Y-%m-%d"), current_date.strftime("%Y-%m-%d")])
    current_date = start_date

# Reverse the intervals to ensure correct chronological order
intervals_reversed = intervals[::-1]

# Adjust the first date of the temporal extent
first_date_adjusted = second_date - timedelta(days=60)
temporal_extent[0] = first_date_adjusted.strftime("%Y-%m-%d")

# Load Sentinel-1 GRD collection
start_date = first_date_adjusted
end_date = second_date

# create Sentinel-1 12 day composite
def sentinel1_composite(
    start_date,
    end_date,
    connection_provider=connection,
    provider="Terrascope",
    processing_opts={},
    relativeOrbit=None,
    orbitDirection=None,
    sampling=False,
    stepsize=12,
    overlap=6,
    reducer="mean",
):
    """
    Compute a Sentinel-1 datacube, composited at 12-day intervals.
    """

    c = connection_provider()
    # define the temporal extent for the Sentinel-1 data
    temp_ext_s1 = [start_date, end_date]

    s1 = c.load_collection(
        "SENTINEL1_GRD", temporal_extent=temp_ext_s1, bands=["VH", "VV"]
    )

    # apply SAR backscatter processing to the collection
    s1 = s1.sar_backscatter(coefficient="sigma0-ellipsoid")

    # apply band-wise processing to create a ratio and log-transformed bands
    s1 = s1.apply_dimension(
        dimension="bands",
        process=lambda x: array_create(
            [
                30.0 * x[0] / x[1],  # Ratio of VH to VV
                30.0 + 10.0 * x[0].log(base=10),  # Log-transformed VH
                30.0 + 10.0 * x[1].log(base=10),  # Log-transformed VV
            ]
        ),
    )

    s1 = s1.rename_labels("bands", ["ratio"] + s1.metadata.band_names)

    # scale to int16
    s1 = s1.linear_scale_range(0, 30, 0, 30000)

    # aggregate the collection to 12-day intervals using the specified reducer
    s1_dekad = s1.aggregate_temporal_period(period="dekad", reducer="median")

    # apply linear interpolation to fill any temporal gaps
    s1_dekad = s1_dekad.apply_dimension(
        dimension="t", process="array_interpolate_linear"
    )
    return s1_dekad

s1_dekad = sentinel1_composite(
    start_date,
    end_date,
    connection,
    "cdse",
    processing_opts=dict(tile_size=256),
    orbitDirection="ASCENDING",
    sampling=True,
    stepsize=12,
)

def timesteps_as_bands(base_features):
    band_names = [band + "_m" + str(i+1)  for band in base_features.metadata.band_names for i in range(5) ]
    result =  base_features.apply_dimension(
        dimension='t',
        target_dimension='bands',
        process=lambda d: eop.array_create(data=d)
    )

    return result.rename_labels('bands', band_names)

s1_features = timesteps_as_bands(s1_dekad)
datacube = s1_features.save_result(format="GTiff")
job = datacube.filter_bbox(bbox).execute_batch(out_format="GTiff")
