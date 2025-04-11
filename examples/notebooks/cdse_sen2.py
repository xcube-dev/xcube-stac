from xcube.core.store import new_data_store
from xcube_stac.utils import reproject_bbox


credentials = {
    "key": "O0M0CUQIDQO9TDZ4D8NR",
    "secret": "qPUyXs9G6j8on6MY5KPhQNHuA5uZTqxEscrbBCGx",
}
store = new_data_store("stac-cdse", stack_mode=True, **credentials)

bbox = [11.7, 53.0, 12.7, 53.6]
crs_target = "EPSG:32632"
bbox_utm = reproject_bbox(bbox, "EPSG:4326", crs_target)
time_range = ["2020-07-15", "2020-07-20"]
ds = store.open_data(
    data_id="sentinel-2-l2a",
    bbox=bbox_utm,
    time_range=time_range,
    spatial_res=40,
    crs=crs_target,
    asset_names=["B02", "B03", "B04", "SCL"],
    apply_scaling=True,
)
print(ds)
print(ds.chunksizes)
# ds.B02.isel(time=0).compute()
