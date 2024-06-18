from xcube.core.store import new_data_store

url = "http://127.0.0.1:8080/ogc"
store = new_data_store("stac", url=url)
ds = store.open_data("collections/datacubes/items/local_ts")
print(ds)
