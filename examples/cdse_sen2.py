from xcube.core.store import new_data_store
import matplotlib.pyplot as plt

from xcube_stac.utils import reproject_bbox


credentials = {
    "key": "O0M0CUQIDQO9TDZ4D8NR",
    "secret": "qPUyXs9G6j8on6MY5KPhQNHuA5uZTqxEscrbBCGx",
}
store = new_data_store("stac-cdse", stack_mode=True, **credentials)

ds = store.open_data(
    data_id="sentinel-2-l2a",
    bbox=[9.1, 53.1, 10.7, 54],
    time_range=["2020-07-15", "2020-08-01"],
    spatial_res=10 / 111320,  # meter in degree
    crs="EPSG:4326",
    asset_names=["B02", "B03", "B04", "SCL"],
    apply_scaling=True,
    angles_sentinel2=True,
)
print(ds)

# bbox = [11.7, 53.0, 12.7, 53.6]
# # bbox = [9.7, 53.0, 10.7, 53.6]
# crs_target = "EPSG:32632"
# bbox_utm = reproject_bbox(bbox, "EPSG:4326", crs_target)
# time_range = ["2025-05-11", "2025-05-14"]
# ds = store.open_data(
#     data_id="sentinel-2-l2a",
#     bbox=bbox_utm,
#     time_range=time_range,
#     spatial_res=40,
#     crs=crs_target,
#     asset_names=["B02", "B03", "B04", "SCL"],
#     apply_scaling=True,
#     angles_sentinel2=True,
# )
# print(ds)
# # ds.B02.isel(time=1)[::5, ::5].plot(vmin=0., vmax=0.3)
# # plt.show()


# bbox = [9.5, 53.3, 10.0, 53.8]
# crs_target = "EPSG:32632"
# bbox_utm = reproject_bbox(bbox, "EPSG:4326", crs_target)
# ds = store.open_data(
#     data_id="sentinel-2-l2a",
#     bbox=bbox_utm,
#     time_range=["2020-07-26", "2020-08-01"],
#     spatial_res=10,
#     crs="EPSG:32632",
#     asset_names=["B04"],
#     apply_scaling=True,
#     angles_sentinel2=True,
# )
# print(ds)
# fig, ax = plt.subplots(1, 3, figsize=(20, 6))
# ds.B04.isel(time=-1)[::10, ::10].plot(ax=ax[0], vmin=0, vmax=0.2)
# ds.solar_angle.isel(angle=0, time=-1)[::10, ::10].plot(ax=ax[1])
# ds.viewing_angle.isel(band=0, angle=0, time=-1)[::10, ::10].plot(ax=ax[2])
# plt.show()
# bbox = [9.1, 53.1, 10.7, 54]
# crs_target = "EPSG:32632"
# bbox_utm = reproject_bbox(bbox, "EPSG:4326", crs_target)
# print(bbox_utm)
# ds = store.open_data(
#     data_id="sentinel-2-l2a",
#     bbox=bbox_utm,
#     time_range=["2020-07-25", "2020-08-01"],
#     spatial_res=60,
#     crs=crs_target,
#     asset_names=["B04", "SCL"],
#     apply_scaling=True,
#     angles_sentinel2=True,
# )
# print(ds)
# time_idx = -2
# fig, ax = plt.subplots(1, 3, figsize=(20, 6))
# ds.B04.isel(time=time_idx)[::10, ::10].plot(ax=ax[0], vmin=0, vmax=0.2)
# ds.solar_angle.isel(angle=0, time=time_idx).plot(ax=ax[1])
# ds.viewing_angle.isel(band=0, angle=0, time=time_idx).plot(ax=ax[2])
# plt.show()

# bbox_wgs84 = [9.9, 53.1, 10.7, 53.5]
# crs_target = "EPSG:32632"
# bbox_utm = reproject_bbox(bbox_wgs84, "EPSG:4326", crs_target)
# ds = store.open_data(
#     data_id="sentinel-2-l2a",
#     bbox=bbox_utm,
#     time_range=["2020-08-29", "2020-09-03"],
#     spatial_res=60,
#     crs=crs_target,
#     asset_names=["B04"],
#     apply_scaling=True,
#     angles_sentinel2=True,
# )
# print(ds)
#
# print(
#     [
#         ds.chunksizes["time"][0],
#         ds.chunksizes["y"][0],
#         ds.chunksizes["x"][0],
#         ds.chunksizes["angle_y"][0],
#         ds.chunksizes["angle_x"][0],
#         ds.chunksizes["angle"][0],
#         ds.chunksizes["band"][0],
#     ]
# )


# bbox_utm = [620000, 5800000, 630000, 5810000]
# ds = store.open_data(
#     data_id="sentinel-2-l2a",
#     bbox=bbox_utm,
#     time_range=["2023-11-01", "2023-11-10"],
#     spatial_res=20,
#     crs="EPSG:32635",
#     apply_scaling=True,
#     angles_sentinel2=True,
# )
# print(ds)

# store = new_data_store("stac-cdse", **credentials)
#
# data_id = (
#     "collections/sentinel-2-l2a/items/S2A_MSIL2A_20200301T090901"
#     "_N0500_R050_T35UPU_20230630T033416"
# )
#
# # open data as dataset
# ds = store.open_data(
#     data_id=data_id,
#     apply_scaling=True,
#     angles_sentinel2=True,
# )
# print(ds)
