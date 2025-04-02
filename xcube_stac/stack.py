# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import dask.array as da
import numpy as np
import pystac
import pyproj
import xarray as xr
from xcube.core.resampling import affine_transform_dataset

from .utils import (
    add_nominal_datetime,
    get_spatial_dims,
    get_gridmapping,
    clip_dataset_by_geometry,
    wrapper_clip_dataset_by_geometry,
    wrapper_resample_in_space,
)
from .constants import LOG, TILE_SIZE


def groupby_solar_day(items: list[pystac.Item]) -> xr.DataArray:
    items = add_nominal_datetime(items)

    # get dates and tile IDs of the items
    dates = []
    tile_ids = []
    proc_versions = []
    for item in items:
        dates.append(item.properties["datetime_nominal"].date())
        tile_ids.append(item.properties["grid:code"])
        proc_versions.append(get_processing_version(item))
    dates = np.unique(dates)
    tile_ids = np.unique(tile_ids)
    proc_versions = np.unique(proc_versions)[::-1]

    # sort items by date and tile ID into a data array
    grouped = xr.DataArray(
        np.empty((len(dates), len(tile_ids), 2, len(proc_versions)), dtype=object),
        dims=("time", "tile_id", "idx", "proc_version"),
        coords=dict(
            time=dates, tile_id=tile_ids, idx=[0, 1], proc_version=proc_versions
        ),
    )
    for idx, item in enumerate(items):
        date = item.properties["datetime_nominal"].date()
        tile_id = item.properties["grid:code"]
        proc_version = get_processing_version(item)
        if not grouped.sel(
            time=date, tile_id=tile_id, idx=0, proc_version=proc_version
        ).values:
            grouped.loc[date, tile_id, 0, proc_version] = item
        elif not grouped.sel(
            time=date, tile_id=tile_id, idx=1, proc_version=proc_version
        ).values:
            grouped.loc[date, tile_id, 1, proc_version] = item
        else:
            item0 = grouped.sel(
                time=date, tile_id=tile_id, idx=0, proc_version=proc_version
            ).item()
            item1 = grouped.sel(
                time=date, tile_id=tile_id, idx=1, proc_version=proc_version
            ).item()
            LOG.warn(
                "More that two items found for datetime and tile ID: "
                f"[{item0.id}, {item1.id}, {item.id}]. Only the first tow items "
                "are considered."
            )

    # take the latest processing version
    da_arr = grouped.data
    mask = da_arr != None
    proc_version_idx = np.argmax(mask, axis=-1)
    da_arr_select = np.take_along_axis(
        da_arr, proc_version_idx[..., np.newaxis], axis=-1
    )[..., 0]
    grouped = xr.DataArray(
        da_arr_select,
        dims=("time", "tile_id", "idx"),
        coords=dict(time=dates, tile_id=tile_ids, idx=[0, 1]),
    )

    # replace date by datetime form first item
    dts = []
    for date in grouped.time.values:
        next_item = next(
            value for value in grouped.sel(time=date, idx=0).values if value is not None
        )
        dts.append(
            np.datetime64(
                next_item.properties["datetime_nominal"].replace(tzinfo=None)
            ).astype("datetime64[ns]")
        )
    grouped = grouped.assign_coords(time=dts)

    return grouped


def get_bounding_box(
    access_params: xr.DataArray,
) -> tuple[list[float | int], list[list[float | int]]]:
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    bboxes = []
    for tile_id in access_params.tile_id.values:
        params = next(
            value
            for value in access_params.sel(tile_id=tile_id).values.flatten()
            if value is not None
        )
        bbox = params["item"].assets[params["name"]].extra_fields["proj:bbox"]
        bboxes.append(bbox)
        if xmin > bbox[0]:
            xmin = bbox[0]
        if ymin > bbox[1]:
            ymin = bbox[1]
        if xmax < bbox[2]:
            xmax = bbox[2]
        if ymax < bbox[3]:
            ymax = bbox[3]
    return [xmin, ymin, xmax, ymax], bboxes


def create_empty_dataset(
    access_params: xr.DataArray, asset_name: str, bbox: list[float | int]
) -> xr.Dataset:
    params = next(
        value
        for value in access_params.sel(asset_name=asset_name).values.flatten()
        if value is not None
    )
    spatial_res = params["item"].assets[params["name"]].extra_fields["gsd"]
    crs = params["item"].assets[params["name"]].extra_fields["proj:code"]
    half_res = spatial_res / 2
    y = np.arange(bbox[3] - half_res, bbox[1], -spatial_res)
    x = np.arange(bbox[0] + half_res, bbox[2], spatial_res)
    empty_data = da.full(
        (access_params.sizes["time"], len(y), len(x)),
        np.nan,
        chunks=(1, TILE_SIZE, TILE_SIZE),
    )
    ds = xr.Dataset(
        {params["name_origin"]: (("time", "y", "x"), empty_data)},
        coords={"x": x, "y": y, "time": access_params.time},
    )
    ds = ds.assign_coords(
        spatial_ref=xr.DataArray(0, attrs=pyproj.CRS.from_string(crs).to_cf())
    )
    return ds


def merge_utm_zones(list_ds: list[xr.Dataset], **open_params) -> xr.Dataset:
    resampled_list_ds = []
    for ds in list_ds:
        resampled_list_ds.append(_resample_dataset_soft(ds, **open_params))
    return mosaic_spatial_along_time_take_first(resampled_list_ds)


def _resample_dataset_soft(ds: xr.Dataset, **open_params) -> xr.Dataset:
    crs_final = pyproj.CRS.from_string(open_params["crs"])
    crs_data = pyproj.CRS.from_cf(ds.spatial_ref.attrs)
    if crs_final == crs_data:
        ds = clip_dataset_by_geometry(ds, geometry=open_params["bbox"])
        if (
            ds.x[1] - ds.x[0] != open_params["spatial_res"]
            or ds.y[1] - ds.y[0] != -open_params["spatial_res"]
        ):
            target_gm = get_gridmapping(
                [ds.x[0], ds.y[0], ds.x[-1], ds.y[-1]],
                open_params["spatial_ref"],
                crs_data,
            )
            ds = affine_transform_dataset(ds, target_gm=target_gm)
    else:
        ds = wrapper_clip_dataset_by_geometry(ds, **open_params)
        target_gm = get_gridmapping(
            open_params["bbox"],
            open_params["spatial_ref"],
            crs_final,
        )
        ds = wrapper_resample_in_space(ds, target_gm)
    return ds


def get_processing_version(item: pystac.Item) -> float:
    return float(
        item.properties.get(
            "processing:version",
            item.properties.get("s2:processing_baseline", "1.0"),
        )
    )


def mosaic_spatial_take_first(list_ds: list[xr.Dataset]) -> xr.Dataset:
    if len(list_ds) == 1:
        return list_ds[0]
    dim = "dummy"
    ds = xr.concat(list_ds, dim=dim)
    y_coord, x_coord = get_spatial_dims(list_ds[0])

    ds_mosaic = xr.Dataset()
    for key in ds:
        if ds[key].dims[-2:] == (y_coord, x_coord):
            axis = ds[key].dims.index(dim)
            da_arr = ds[key].data
            nan_mask = da.isnan(da_arr)
            first_non_nan_index = (~nan_mask).argmax(axis=axis)
            da_arr_select = da.choose(first_non_nan_index, da_arr)
            ds_mosaic[key] = xr.DataArray(
                da_arr_select,
                dims=ds[key].dims[1:],
                coords=ds[key].coords,
            )
        else:
            ds_mosaic[key] = ds[key]

    return ds_mosaic


def mosaic_spatial_along_time_take_first(list_ds: list[xr.Dataset]) -> xr.Dataset:
    if len(list_ds) == 1:
        return list_ds[0]

    dts = np.sort(np.unique(np.concatenate([ds.time.values for ds in list_ds])))
    final_slices = []
    for dt in dts:
        slice_ds = [ds.sel(time=dt) for ds in list_ds if dt in ds.coords["time"].values]
        slice_ds = [ds.drop_vars("time") for ds in slice_ds]
        if len(slice_ds) == 1:
            ds_mosaic = slice_ds[0]
        else:
            ds_mosaic = mosaic_spatial_take_first(slice_ds)
        final_slices.append(ds_mosaic)
    final_ds = xr.concat(final_slices, dim="time", join="exact")
    final_ds = final_ds.assign_coords(coords=dict(time=dts))
    return final_ds
