from xcube.core.store import new_data_store
import matplotlib.pyplot as plt

credentials = dict(
    key="O0M0CUQIDQO9TDZ4D8NR",
    secret="qPUyXs9G6j8on6MY5KPhQNHuA5uZTqxEscrbBCGx",
)
store = new_data_store("stac-cdse-ardc", **credentials)
print(store.list_data_ids())
ds = store.open_data(
    data_id="sentinel-3-sl-2-lst-ntc",
    bbox=[8, 52, 12, 55],
    time_range=["2020-07-31", "2020-08-01"],
    spatial_res=300 / 111320,  # meter in degree
    crs="EPSG:4326",
)
print(ds)

ds.LST.isel(time=0).plot()
plt.show()
