import psutil
import time
import zarr
import numpy as np
from pathlib import Path
import os


# Function to monitor CPU and IO usage
def monitor_resources():
    # Get the current CPU and IO stats
    cpu_percent = psutil.cpu_percent(interval=1)
    io_counters = psutil.disk_io_counters()

    # Print current CPU and IO stats
    print(f"CPU usage: {cpu_percent}%")
    print(f"Read IO: {io_counters.read_bytes / 1e6} MB")
    print(f"Write IO: {io_counters.write_bytes / 1e6} MB")


# Create a sample Zarr file
DIR = Path(__file__).parent.resolve()
zarr_file = os.path.join(DIR, "example.zarr")
shape = (1000, 1000)
chunks = (100, 100)

# Open or create Zarr store
store = zarr.open(zarr_file, mode="w", shape=shape, chunks=chunks, dtype="f4")

# Writing data to the Zarr file in a loop
for i in range(10):
    store[i * 100 : (i + 1) * 100, :] = np.random.random((100, 1000))

    # Monitor resources during writing
    monitor_resources()

    # Optional: pause to observe changes in stats
    time.sleep(1)  # Adjust as needed to monitor over time
