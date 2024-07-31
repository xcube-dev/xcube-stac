from xcube.core.store import new_data_store, get_data_store_params_schema

url = "http://127.0.0.1:8080/ogc"
store = new_data_store("stac", url=url)
data_ids = store.list_data_ids()
print(data_ids)
