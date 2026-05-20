from xcube.core.store import new_data_store
from xcube_resampling.utils import reproject_bbox

store = new_data_store("stac-pc-ardc")

bbox_wgs84 = [9.9, 53.1, 10.7, 53.5]
crs_target = "EPSG:32632"
bbox_utm = reproject_bbox(bbox_wgs84, "EPSG:4326", crs_target)
ds = store.open_data(
    data_id="hls2-l30",
    asset_names=["B04", "B03", "B02"],
    bbox=bbox_wgs84,
    time_range=["2020-08-20", "2020-09-01"],
    spatial_res=0.00054,
    crs="EPSG:4326",
)
print(ds)
