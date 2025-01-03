import os
from pathlib import Path
import subprocess

######################################################



def copy_to_s3(bucket_name, s3_path, local_path, config_path):
    cmd = ['rclone',
           '--config', config_path,
        'copy',
        '--ignore-existing',
        '-v',
        f'{local_path}',
        f'gisat:{bucket_name}/{s3_path}']

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing rclone: {result.stderr}")
    else:
        print("Download completed!")


# Example usage:
bucket_name = "gisat-archive"
s3_path = f""

local_path = "/mnt/hddarchive.nfs/amazonas_dir/openEO"
config_path = "/home/eouser/userdoc/rclone.conf"

copy_to_s3(bucket_name, s3_path, local_path, config_path)