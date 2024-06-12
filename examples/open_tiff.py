from xcube.core.store.store import new_data_store


store = new_data_store("stac", url="https://earth-search.aws.element84.com/v1")

# open data without open_params
ds = store.open_data(
    "collections/sentinel-2-l2a/items/S2A_32UNU_20200305_0_L2A",
    asset_names=["blue", "green", "nir"],
)
print(ds)
