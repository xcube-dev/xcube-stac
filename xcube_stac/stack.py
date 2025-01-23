# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
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

import datetime

import dask.array as da
import numpy as np
import pystac
import xarray as xr

from ._utils import add_nominal_datetime
from ._utils import get_spatial_dims
from .constants import LOG


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
        if ds[key].dims == (dim, y_coord, x_coord):
            axis = ds[key].dims.index(dim)
            da_arr = ds[key].data
            nan_mask = da.isnan(da_arr)
            first_non_nan_index = (~nan_mask).argmax(axis=axis)
            da_arr_select = da.choose(first_non_nan_index, da_arr)
            ds_mosaic[key] = xr.DataArray(
                da_arr_select,
                dims=(y_coord, x_coord),
                coords={y_coord: ds[y_coord], x_coord: ds[x_coord]},
            )
        else:
            ds_mosaic[key] = ds[key]

    return ds_mosaic


def mosaic_spatial_along_time_take_first(
    list_ds: list[xr.Dataset], dts: list[datetime.datetime] = None
) -> xr.Dataset:
    if len(list_ds) == 1:
        return list_ds[0]

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
    if "crs" in final_ds:
        final_ds = final_ds.drop_vars("crs")
        final_ds["crs"] = list_ds[0].crs
    return final_ds
