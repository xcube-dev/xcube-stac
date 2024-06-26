from xcube.core.store import new_data_store


store = new_data_store("stac", url="http://127.0.0.1:8080/ogc")
ds = store.open_data("collections/datacubes/items/zarr_file")
print(ds)
