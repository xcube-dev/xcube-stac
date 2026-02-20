from xcube.core.store import new_data_store
import matplotlib.pyplot as plt

store = new_data_store("stac-pc")
descriptors = list(
    store.search_data(
        collections=["sentinel-3-slstr-lst-l2-netcdf"],
        bbox=[8, 52, 8.1, 52.1],
        time_range=["2020-07-31", "2020-07-31"],
    )
)

fig, _ax = plt.subplots(2, 3, figsize=(12, 8))
ax = _ax.flatten()

for i, descriptor in enumerate(descriptors):
    ds = store.open_data(descriptor.data_id)
    print(ds)

plt.tight_layout()
