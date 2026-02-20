from xcube.core.store import new_data_store

store = new_data_store("stac-pc-ardc")
print(store.list_data_ids())
ds = store.open_data(
    data_id="sentinel-3-slstr-lst-l2-netcdf",
    bbox=[8, 52, 12, 55],
    time_range=["2020-07-31", "2020-08-01"],
    spatial_res=300 / 111320,  # meter in degree
    crs="EPSG:4326",
)
print(ds)

print(ds.LST.isel(time=0).values)
