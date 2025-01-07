import datetime

import matplotlib.pyplot as plt
from xcube.core.store import new_data_store
import logging
import pyproj


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

credentials = {
    "key": "OX94JKCFEF1PLNVNAU2J",
    "secret": "ceucxUr3s8yNhaP8k3g0FnQERKgm8SUfd9TdSFrF",
}
store = new_data_store("stac-cdse", stack_mode=True, **credentials)
file_store = new_data_store("file", root="data")
time_range = ["2023-07-01", "2023-07-10"]
lon = 9.99
lat = 53.55
target_crs = "EPSG:3035"
t = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
x, y = t.transform(lon, lat)
half_size = 2000
bbox = [x - half_size, y - half_size, x + half_size, y + half_size]

print(f"{datetime.datetime.now()}: start open_data")
ds = store.open_data(
    data_id="SENTINEL-2",
    bbox=bbox,
    time_range=time_range,
    processing_level="L2A",
    spatial_res=10,
    crs="EPSG:3035",
    tile_size=(201, 201),
    apply_scaling=True,
    asset_names=[
        "B01",
    ],
)
print(f"{datetime.datetime.now()}: start write_data")
file_store.write_data(ds, "test.zarr", replace=True)
print(f"{datetime.datetime.now()}: end write_data")

ds = file_store.open_data("test.zarr")
ds.B01.isel(time=-1).plot()
plt.show()
