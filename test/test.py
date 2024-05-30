from xcube.core.store import new_data_store, DataStoreError


url_nonsearchable = (
    "https://raw.githubusercontent.com/stac-extensions/"
    "label/main/examples/multidataset/catalog.json"
)
store = new_data_store("stac", url=url_nonsearchable)
try:
    item = store._access_item("bla")
except DataStoreError:
    

        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(DataStoreError) as cm:
            store._access_item(self.data_id_nonsearchable.replace("z", "s"))
        self.assertIn(
            "requests.exceptions.HTTPError: 404 Client Error: Not Found for url",
            f"{cm.exception}",
        )