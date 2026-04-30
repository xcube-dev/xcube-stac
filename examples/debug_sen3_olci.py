from xcube.core.store import new_data_store
from matplotlib import pyplot as plt
import time

credentials = {
    "key": "O0M0CUQIDQO9TDZ4D8NR",
    "secret": "qPUyXs9G6j8on6MY5KPhQNHuA5uZTqxEscrbBCGx",
}
store = new_data_store("stac-cdse-ardc", **credentials)
start = time.time()
ds = store.open_data(
    data_id="sentinel-3-syn-2-syn-ntc",
    time_range=("2025-04-15", "2025-04-16"),
    bbox=[731912.0, 8615835.0, 735912.0, 8619835.0],
    spatial_res=10,
    crs=" EPSG:32752",
    add_error_bands=True,
    asset_names=["syn_S1N_reflectance"],
)

print(ds)
end = time.time()
print(end - start, "seconds")
print(ds.compute())
end2 = time.time()
print(end2 - start, "seconds")


fig, ax = plt.subplots(1, ds.sizes["time"], figsize=(4 * ds.sizes["time"], 4))
for i in range(ds.sizes["time"]):
    ds.SDR_S1N.isel(time=0).plot(ax=ax[i])
plt.show()
