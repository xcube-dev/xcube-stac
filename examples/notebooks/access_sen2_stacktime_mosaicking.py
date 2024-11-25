from xcube.core.store import new_data_store

# from dask.distributed import Client
# import psutil
# import time
# import os
# import threading
# import matplotlib.pyplot as plt

# attrs = {
#     "site_id": 1280,
#     "path": "cubes/training/0.0.1/1280.zarr",
#     "center_wgs84": [53.16298452696103, 29.38809076583505],
#     "center_utm": [5893064.93625913, 659648.5183325501],
#     "bbox_wgs84": [
#         29.386939788333212,
#         53.16233384987463,
#         29.38925624089212,
#         53.163636088441415,
#     ],
#     "bbox_utm": [659574, 5892990, 659724, 5893140],
#     "utm_zone": "35U",
#     "version": "0.0.1",
#     "creation_datetime": "2024-11-19T15:26:33.004419",
#     "last_modified_datetime": "2024-11-19T15:26:33.004423",
#     "landcover_first": None,
#     "landcover_first_percentage": None,
#     "landcover_second": None,
#     "landcover_second_percentage": None,
#     "protection_mask": None,
#     "acknowledgment": "DeepFeatures project",
#     "contributor_name": "Brockmann Consult GmbH",
#     "contributor_url": "www.brockmann-consult.de",
#     "creator_email": "info@brockmann-consult.de",
#     "creator_name": "Brockmann Consult GmbH",
#     "creator_url": "www.brockmann-consult.de",
#     "institution": "Brockmann Consult GmbH",
#     "project": "DeepExtreme",
#     "publisher_email": "info@brockmann-consult.de",
#     "publisher_name": "Brockmann Consult GmbH",
#     "ground_measurement": None,
#     "protection_status": None,
#     "flux_tower_elevation": None,
#     "time_range_start": "2023-11-02",
#     "time_range_end": "2024-04-30",
# }

# client = Client()
# time.sleep(3)


# def print_memory():
#     while True:
#         print(psutil.Process(os.getpid()).memory_info().rss / 1024**3)
#         time.sleep(1)
#
#
# daemon = threading.Thread(target=print_memory, daemon=True, name="Monitor")
# daemon.start()


credentials = {
    "key": "O0M0CUQIDQO9TDZ4D8NR",
    "secret": "qPUyXs9G6j8on6MY5KPhQNHuA5uZTqxEscrbBCGx",
}
store = new_data_store("stac-cdse", stack_mode=True, **credentials)
bbox = [
    29.386939788333212,
    53.16233384987463,
    29.38925624089212,
    53.163636088441415,
]
bbox_utm = [659574, 5892990, 659724, 5893140]


ds = store.open_data(
    data_id="SENTINEL-2",
    bbox=bbox_utm,
    time_range=["2023-11-01", "2023-12-01"],
    processing_level="L2A",
    spatial_res=10,
    crs="EPSG:32635",
    asset_names=[
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "SCL",
    ],
)
print(ds)
store_file = new_data_store("file")
store_file.write_data(ds, "minicube_test2.zarr", replace=True)

# _DIR = Path(__file__).parent.resolve()
# ds.to_zarr(
#     os.path.join(_DIR, "minicube_test2.zarr"),
#     mode="w",
#     synchronizer=zarr.sync.ThreadSynchronizer,
# )
