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
import collections
from typing import Union

import dask.array as da
import numpy as np
import pyproj
import pystac
import rioxarray
import xarray as xr
from xcube.core.resampling import resample_in_space
from xcube.core.gridmapping import GridMapping


from ._utils import add_nominal_datetime
from .constants import TILE_SIZE


def stack_items(parsed_items: dict[list[pystac.Item]], **open_params) -> xr.Dataset:
    target_gm = _get_gridmapping(
        open_params["bbox"],
        open_params["spatial_res"],
        open_params["crs"],
        open_params.get("tile_size", TILE_SIZE),
    )
    ds_dates = []
    np_datetimes = []
    for datetime, items_for_date in parsed_items.items():
        print(datetime)
        np_datetimes.append(np.datetime64(datetime).astype("datetime64[ns]"))
        list_ds_items = []
        for item in items_for_date:
            list_ds_asset = []
            for band in open_params["bands"]:
                ds = rioxarray.open_rasterio(
                    item.assets[band].href,
                    chunks=dict(x=TILE_SIZE, y=TILE_SIZE),
                    band_as_variable=True,
                )
                ds = ds.rename(dict(band_1=band))
                ds = ds.where(ds != 0)
                ds = resample_in_space(ds, target_gm)
                list_ds_asset.append(ds)
            ds = list_ds_asset[0].copy()
            for ds_asset in list_ds_asset[1:]:
                ds.update(ds_asset)
            list_ds_items.append(ds)
        ds_mosaic = mosaic_take_first(list_ds_items)
        ds_dates.append(ds_mosaic)
    ds = xr.concat(ds_dates, dim="time")
    ds = ds.assign_coords(coords=dict(time=np_datetimes))
    ds = ds.drop_vars("crs")
    ds["crs"] = ds_dates[0].crs

    return ds


def groupby_solar_day(items: list[pystac.Item]) -> dict:
    items = add_nominal_datetime(items)
    nested_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    # group by date and processing baseline
    for idx, item in enumerate(items):
        date = item.properties["datetime_nominal"].date()
        proc_version = float(item.properties["processorVersion"])
        nested_dict[date][proc_version].append(idx)
    # if two processing baselines are available, take most recent one
    grouped = {}
    for date, proc_version_dic in nested_dict.items():
        proc_version = max(list(proc_version_dic.keys()))
        grouped[date] = nested_dict[date][proc_version]
    return grouped


def resample_in_space(ds: xr.Dataset, target_gm: GridMapping) -> xr.Dataset:
    ds = resample_in_space(ds, target_gm=target_gm, encode_cf=True)
    if "spatial_ref" in ds.coords:
        ds = ds.drop_vars("spatial_ref")
    if "x_bnds" in ds.coords:
        ds = ds.drop_vars("x_bnds")
    if "y_bnds" in ds.coords:
        ds = ds.drop_vars("y_bnds")
    if "transformed_x" in ds.data_vars:
        ds = ds.drop_vars("transformed_x")
    if "transformed_y" in ds.data_vars:
        ds = ds.drop_vars("transformed_y")
    return ds


def mosaic_take_first(list_ds):
    dim = "dummy"
    ds = xr.concat(list_ds, dim=dim)
    ds = ds.drop_vars("crs")
    ds_mosaic = xr.Dataset()
    for key in ds:
        axis = ds[key].dims.index(dim)
        da_arr = ds[key].data
        nan_mask = da.isnan(da_arr)
        first_non_nan_index = (~nan_mask).argmax(axis=axis)
        da_arr_select = da.choose(first_non_nan_index, da_arr)
        ds_mosaic[key] = xr.DataArray(
            da_arr_select,
            dims=("y", "x"),
            coords={"y": ds.y, "x": ds.x},
        )
    ds_mosaic["crs"] = list_ds[0].crs
    return ds_mosaic


def _get_timestamps(grouped: dict, items: list[pystac.Item]) -> dict:
    grouped_new = {}
    for old_key, value in grouped.items():
        new_key = items[value[0]].properties["datetime_nominal"].replace(tzinfo=None)
        grouped_new[new_key] = value
    return grouped


def _get_gridmapping(
    bbox: list[float],
    spatial_res: float,
    crs: Union[str, pyproj.crs.CRS],
    tile_size: Union[int, tuple[int, int]] = TILE_SIZE,
) -> GridMapping:
    x_size = int((bbox[2] - bbox[0]) / spatial_res) + 1
    y_size = int(abs(bbox[3] - bbox[1]) / spatial_res) + 1
    return GridMapping.regular(
        size=(x_size, y_size),
        xy_min=(bbox[0] - spatial_res / 2, bbox[1] - spatial_res / 2),
        xy_res=spatial_res,
        crs=crs,
        tile_size=tile_size,
    )
