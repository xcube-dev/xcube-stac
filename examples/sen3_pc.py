import matplotlib.pyplot as plt
from xcube.core.store import new_data_store

# store = new_data_store("stac-pc")
# data_id = "collections/sentinel-3-slstr-lst-l2-netcdf/items/S3A_SL_2_LST_20200705T094658_20200705T094958_0179_060_136_2160"
# ds = store.open_data(data_id)
# print(ds)

store = new_data_store("stac-pc-ardc")
print(store.list_data_ids())
ds = store.open_data(
    data_id="sentinel-3-slstr-lst-l2-netcdf",
    bbox=[8, 52, 12, 55],
    time_range=["2020-08-01", "2020-08-01"],
    spatial_res=700 / 111320,  # meter in degree
    crs="EPSG:4326",
)
print(ds)

ds.LST.isel(time=0).plot()
plt.show()
ds.cloud_mask.isel(time=0).plot()
plt.show()
