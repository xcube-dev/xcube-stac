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

import dask.array as da
import pystac
import xarray as xr

from ._utils import add_nominal_datetime
from ._utils import get_spatial_dims


def groupby_solar_day(items: list[pystac.Item]) -> dict:
    items = add_nominal_datetime(items)
    nested_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    # group by date and processing baseline if given
    for idx, item in enumerate(items):
        date = item.properties["datetime_nominal"].date()
        processing_baseline = float(item.properties.get("processing:version", "1.0"))
        nested_dict[date][processing_baseline].append(item)

    # if two processing baselines are available, take most recent one
    grouped = {}
    for date, proc_version_dic in nested_dict.items():
        proc_version = max(list(proc_version_dic.keys()))
        grouped[date] = nested_dict[date][proc_version]

    # get timestamp
    grouped_new = {}
    for date, items in grouped.items():
        dt = items[0].properties["datetime_nominal"].replace(tzinfo=None)
        grouped_new[dt] = sorted(items, key=lambda item: item.id)
    return grouped_new


def mosaic_take_first(list_ds: list[xr.Dataset]) -> xr.Dataset:
    if len(list_ds) == 1:
        return list_ds[0]
    dim = "dummy"
    ds = xr.concat(list_ds, dim=dim)
    if "crs" in ds:
        ds = ds.drop_vars("crs")
    y_coord, x_coord = get_spatial_dims(ds)
    ds_mosaic = xr.Dataset()
    for key in ds:
        axis = ds[key].dims.index(dim)
        da_arr = ds[key].data
        nan_mask = da.isnan(da_arr)
        first_non_nan_index = (~nan_mask).argmax(axis=axis)
        da_arr_select = da.choose(first_non_nan_index, da_arr)
        ds_mosaic[key] = xr.DataArray(
            da_arr_select,
            dims=("time", y_coord, x_coord),
            coords={"time": ds["time"], y_coord: ds[y_coord], x_coord: ds[x_coord]},
        )
    if "crs" in list_ds[0]:
        ds_mosaic["crs"] = list_ds[0].crs
    return ds_mosaic
