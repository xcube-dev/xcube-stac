from xcube.core.store import new_data_store

store = new_data_store("stac-pc-ardc")

ds = store.open_data(
    data_id="landsat-c2-l2",
    bbox=[9.7, 53.3, 10.3, 53.8],
    time_range=["2026-06-24", "2026-07-08"],
    spatial_res=30 / 111320,
    crs="EPSG:4326",
    asset_names=["red", "green", "blue", "lwir11", "qa_pixel"],
)
print(ds)
