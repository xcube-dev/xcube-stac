import matplotlib.pyplot as plt
from xcube.core.store import new_data_store

credentials = dict(
    key="O0M0CUQIDQO9TDZ4D8NR",
    secret="qPUyXs9G6j8on6MY5KPhQNHuA5uZTqxEscrbBCGx",
)
# store = new_data_store("stac-cdse-ardc", **credentials)
# print(store.list_data_ids())
# ds = store.open_data(
#     data_id="sentinel-3-sl-2-lst-ntc",
#     bbox=[8, 52, 12, 55],
#     time_range=["2020-07-31", "2020-08-01"],
#     spatial_res=300 / 111320,  # meter in degree
#     crs="EPSG:4326",
# )
# print(ds)
#
# ds.LST.isel(time=0).plot()
# plt.show()


store = new_data_store("stac-cdse", **credentials)
data_id = (
    "collections/sentinel-3-syn-2-syn-ntc/items/S3B_SY_2_SYN____20250706T233058_"
    "20250706T233358_20250708T043306_0179_108_258_3420_ESA_O_NT_002"
)
ds = store.open_data(data_id, apply_rectification=False, add_error_bands=False)
print(ds)
ds.SDR_Oa11[::4, ::4].plot()
plt.show()
