from xcube.core.store.store import new_data_store


store = new_data_store(
    "stac", url="https://planetarycomputer.microsoft.com/api/stac/v1"
)

# open data without open_params
ds = store.open_data("collections/nasa-nex-gddp-cmip6/items/CESM2-WACCM.ssp245.2100")
print(ds)
