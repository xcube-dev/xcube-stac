from xcube.core.store import new_data_store

credentials = dict(
    key="O0M0CUQIDQO9TDZ4D8NR",
    secret="qPUyXs9G6j8on6MY5KPhQNHuA5uZTqxEscrbBCGx",
)

store = new_data_store("stac-cdse-ardc", **credentials)


ds = store.open_data(
    data_id="sentinel-2-l1c",
    bbox=[9.7, 53.3, 10.3, 53.8],
    time_range=["2020-07-15", "2020-08-01"],
    spatial_res=10 / 111320,  # meter in degree
    crs="EPSG:4326",
    asset_names=["B02", "B03", "B04"],
    add_angles=True,
)
print(ds)
