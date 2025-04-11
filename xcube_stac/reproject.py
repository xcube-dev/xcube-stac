import math

import numpy as np
import dask.array as da
import xarray as xr
import pandas as pd
import pyproj
from xcube.core.gridmapping import GridMapping

from xcube_stac.utils import get_gridmapping


def generate_test_dataset():
    res = 0.01
    lat_start, lat_end = 75.0, 30.0
    lon_start, lon_end = -25.0, 45.0

    lats = np.arange(lat_start - res / 2, lat_end, -res)
    lons = np.arange(lon_start + res / 2, lon_end, res)
    times = pd.date_range("2022-01-01", "2022-01-15", freq="D")

    data0 = da.zeros(
        (len(times), len(lats), len(lons)), chunks=(1, 500, 1000), dtype="int16"
    )
    data1 = da.ones(
        (len(times), len(lats), len(lons)), chunks=(1, 1000, 500), dtype="float32"
    )

    ds = xr.Dataset(
        {
            "variable0": (("time", "lat", "lon"), data0),
            "variable1": (("time", "lat", "lon"), data1),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds = ds.assign_coords(
        spatial_ref=xr.DataArray(0, attrs=pyproj.CRS.from_string("EPSG:4326").to_cf())
    )
    return ds


def reproject_dataset(ds: xr.Dataset, target_gm: GridMapping):
    # get ij_bbox in source dataset
    source_gm = GridMapping.from_dataset(ds)
    trans_backward = pyproj.Transformer.from_crs(
        target_gm.crs, source_gm.crs, always_xy=True
    )
    num_tiles_x = math.ceil(target_gm.width / target_gm.tile_width)
    num_tiles_y = math.ceil(target_gm.height / target_gm.tile_height)
    scr_ij_bboxes = np.full((4, num_tiles_y, num_tiles_x), -1, dtype=np.int16)
    for idx, xy_bbox in enumerate(target_gm.xy_bboxes):
        idx_y, idx_x = np.unravel_index(idx, (num_tiles_y, num_tiles_x))
        source_xy_bbox = trans_backward.transform_bounds(*xy_bbox)
        j_min = np.argmin(abs(source_gm.x_coords.values - source_xy_bbox[0]))
        j_max = np.argmin(abs(source_gm.x_coords.values - source_xy_bbox[2]))
        if j_min != 0:
            j_min -= 1
        if j_max != source_gm.width:
            j_max += 1
        i_min = np.argmin(abs(source_gm.y_coords.values - source_xy_bbox[1]))
        i_max = np.argmin(abs(source_gm.y_coords.values - source_xy_bbox[3]))
        if i_min > i_max:
            tmp0, tmp1 = i_min, i_max
            i_min, i_max = tmp1, tmp0
        if i_min != 0:
            i_min -= 1
        if i_max != source_gm.height:
            i_max += 1
        scr_ij_bboxes[:, idx_y, idx_x] = [j_min, i_min, j_max, i_max]

    # get meshed coordinates
    target_x = da.from_array(target_gm.x_coords.values, chunks=target_gm.tile_width)
    target_y = da.from_array(target_gm.y_coords.values, chunks=target_gm.tile_height)
    target_xx, target_yy = da.meshgrid(target_x, target_y)

    def transform_block(target_xx: np.ndarray, target_yy: np.ndarray):
        trans_xx, trans_yy = trans_backward.transform(target_xx, target_yy)
        return np.stack([trans_xx, trans_yy])

    source_xx_yy = da.map_blocks(
        transform_block,
        target_xx,
        target_yy,
        dtype=np.float32,
        chunks=(2, target_yy.chunks[0][0], target_yy.chunks[1][0]),
    )
    source_xx = source_xx_yy[0]
    source_yy = source_xx_yy[1]

    # reproject dataset
    x_name, y_name = target_gm.xy_dim_names
    ds_out = xr.Dataset(
        coords={
            "time": ds.time,
            x_name: target_gm.x_coords,
            y_name: target_gm.y_coords,
            "spatial_ref": xr.DataArray(0, attrs=target_gm.crs.to_cf()),
        },
        attrs=ds.attrs,
    )
    for var_name, data_array in ds.items():
        slices_reprojected = []
        for idx, chunk_size in enumerate(data_array.chunks[0]):
            dim0_start = idx * chunk_size
            dim0_end = (idx + 1) * chunk_size
            da_slice = data_array[dim0_start:dim0_end, ...]

            def reproject_block(
                source_xx: np.ndarray,
                source_yy: np.ndarray,
                block_id=None,
            ):
                interpolation = 0
                scr_ij_bbox = scr_ij_bboxes[:, block_id[1], block_id[2]]

                da_clip = da_slice[
                    :, scr_ij_bbox[1] : scr_ij_bbox[3], scr_ij_bbox[0] : scr_ij_bbox[2]
                ]
                x_name, y_name = source_gm.xy_dim_names
                y = da_clip[y_name].values
                x = da_clip[x_name].values
                data = da_clip.values

                ix = (source_xx - x[0]) / source_gm.x_res
                iy = (source_yy - y[0]) / -source_gm.y_res

                if interpolation == 0:
                    ix = np.rint(ix).astype(np.int16)
                    iy = np.rint(iy).astype(np.int16)
                    data_reprojected = data[:, iy, ix]
                else:
                    raise NotImplementedError()

                return data_reprojected

            data_reprojected = da.map_blocks(
                reproject_block,
                source_xx,
                source_yy,
                dtype=data_array.dtype,
                chunks=(
                    da_slice.chunks[0][0],
                    source_yy.chunks[0][0],
                    source_yy.chunks[1][0],
                ),
            )
            data_reprojected = data_reprojected[
                :, : target_gm.height, : target_gm.width
            ]
            slices_reprojected.append(data_reprojected)
        ds_out[var_name] = (
            ("time", y_name, x_name),
            da.concatenate(slices_reprojected, axis=0),
        )
    return ds_out


ds = generate_test_dataset()
print(ds)


target_crs = "EPSG:3035"
target_spatial_res = 500

bbox = [-10, 35, 20, 65]
t = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
target_bbox = t.transform_bounds(*bbox)
target_gm = get_gridmapping(
    list(target_bbox), target_spatial_res, target_crs, tile_size=500
)

ds_reproject = reproject_dataset(ds, target_gm)
print(ds_reproject)
print(ds_reproject.variable0[0].compute())
