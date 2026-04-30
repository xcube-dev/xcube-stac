from xcube.core.store import new_data_store
from matplotlib import pyplot as plt
import time

store = new_data_store("stac-pc-ardc")
start = time.time()
ds = store.open_data(
    data_id="sentinel-3-synergy-syn-l2-netcdf",
    time_range=("2025-04-15", "2025-04-16"),
    bbox=[731912.0, 8615835.0, 735912.0, 8619835.0],
    spatial_res=10,
    crs=" EPSG:32752",
    asset_names=["syn-s1n-reflectance"],
)

print(ds)
end = time.time()
print(end - start, "seconds")
print(ds.compute())
end2 = time.time()
print(end2 - start, "seconds")


fig, ax = plt.subplots(1, ds.sizes["time"], figsize=(4 * ds.sizes["time"], 4))
for i in range(ds.sizes["time"]):
    ds.SDR_Oa01.isel(time=0).plot(ax=ax[i])
plt.show()
