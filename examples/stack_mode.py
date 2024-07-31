from xcube.core.store import new_data_store
import matplotlib.pyplot as plt


store = new_data_store(
    "stac", url="https://earth-search.aws.element84.com/v1", stack_mode=True
)
print(store.list_data_ids())
ds = store.open_data(
    data_id="sentinel-2-l2a",
    bbox=[9, 47, 10, 48],
    time_range=["2020-07-01", "2020-07-05"],
    bands=["red", "green", "blue"],
    groupby="id",
    crs="EPSG:4326",
    chunks={"time": 1, "x": 2048, "y": 2048},
)
print(ds.num_levels)
ds = ds.get_dataset(2)
print(ds)
print(ds.time.values)
ds.red.isel(time=1).plot()
plt.show()
