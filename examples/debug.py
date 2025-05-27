import xarray as xr

storage_options = {
    "anon": False,
    "client_kwargs": dict(endpoint_url="https://eodata.dataspace.copernicus.eu"),
    "key": "OX94JKCFEF1PLNVNAU2J",
    "secret": "ceucxUr3s8yNhaP8k3g0FnQERKgm8SUfd9TdSFrF",
}

path = "s3://eodata/Sentinel-3/SYNERGY/SY_2_SYN___/2025/06/29/S3B_SY_2_SYN____20250629T174217_20250629T174517_20250701T015713_0179_108_155_1800_ESA_O_NT_002.SEN3/geolocation.nc"
# path = "s3://eodata/Sentinel-3/SYNERGY/SY_2_SYN___/2020/07/05/S3A_SY_2_SYN____20200705T094658_20200705T094958_20200706T235401_0180_060_136_2160_LN2_O_NT_002.SEN3/time.nc"

ds = xr.open_dataset(
    path,
    engine="h5netcdf",  # or "scipy"
    chunks={},
    backend_kwargs={},
    storage_options=storage_options,
)
print(ds)

# from xcube.core.store import new_data_store
# store = new_data_store("stac-cdse", stack_mode=False, **credentials)
# # descriptors = list(
# #     store.search_data(
# #         collections=["sentinel-3-syn-2-syn-ntc"],
# #         bbox=[9, 47, 10, 48],
# #         time_range=["2020-07-01", "2020-07-05"],
# #     )
# # )
# # for descriptor in descriptors:
# #     print([d.to_dict() for d in descriptors])
#
# ds = store.open_data(
#     "collections/sentinel-3-syn-2-syn-ntc/items/S3A_SY_2_SYN____20200705T094658_20200705T094958_20200706T235401_0180_060_136_2160_LN2_O_NT_002"
# )
# print(ds)
