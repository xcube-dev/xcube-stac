from xcube.core.store.store import new_data_store


url = "https://s3.eu-central-1.wasabisys.com/stac/odse/catalog.json"
store = new_data_store("stac", url=url)

# open data without open_params
ds = store.open_data(
    "lcv_blue_landsat.glad.ard/lcv_blue_landsat.glad.ard_1999.12.02..2000.03.20/lcv_blue_landsat.glad.ard_1999.12.02..2000.03.20.json",
    asset_names=["blue_p50"],
)
print(ds)
