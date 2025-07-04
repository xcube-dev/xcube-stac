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

from collections import defaultdict

import dask.array as da
import numpy as np
import pyproj
import pystac
import xarray as xr
from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.resampling import (
    affine_transform_dataset,
    rectify_dataset,
    reproject_dataset,
)
from xcube.core.store import DataTypeLike

from xcube_stac.constants import LOG, TILE_SIZE
from xcube_stac.stac_extension.raster import apply_offset_scaling_stack_mode
from xcube_stac.utils import (
    add_nominal_datetime,
    clip_dataset_by_bbox,
    get_format_id,
    get_gridmapping,
    get_spatial_dims,
    is_valid_ml_data_type,
    merge_datasets,
    mosaic_spatial_take_first,
    reproject_bbox,
)

from .sen2 import SENTINEL2_FILL_VALUE


class S3Sentinel3DataAccessor:
    """Implementation of the data accessor supporting
    the nc format of Sentinel-3 data via the AWS S3 protocol.
    """

    def __init__(self, root: str, storage_options: dict):
        self._root = root
        self._storage_options = storage_options

    @property
    def root(self) -> str:
        return self._root

    def open_data(
        self,
        access_params: dict,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        if opener_id is None:
            opener_id = ""
        if is_valid_ml_data_type(data_type) or opener_id.split(":")[0] == "mldataset":
            return NotImplementedError(
                "Multi-level dataset not implemented for Sen3 product stored as netcdf."
            )
        ds = xr.open_dataset(
            (
                f"{access_params['protocol']}://{access_params['root']}/"
                f"{access_params['fs_path']}"
            ),
            chunks={},
            engine="h5netcdf",
            backend_kwargs={"file_obj": {"s3": self._storage_options}},
        )
        if open_params.get("rectify_dataset", None):
            ds = rectify_dataset(ds)
        return ds

    def groupby_solar_day(self, items: list[pystac.Item]) -> xr.DataArray:
        items = add_nominal_datetime(items)

        # get dates and tile IDs of the items
        dates = []
        proc_versions = []
        for item in items:
            dates.append(item.properties["datetime_nominal"].date())
            proc_versions.append(self._get_processing_version(item))
        dates = np.unique(dates)
        proc_versions = np.unique(proc_versions)[::-1]

        # sort items by date and tile ID into a data array
        grouped = xr.DataArray(
            np.empty((len(dates), len(proc_versions)), dtype=object),
            dims=("time", "proc_version"),
            coords=dict(time=dates, proc_version=proc_versions),
        )
        for idx, item in enumerate(items):
            date = item.properties["datetime_nominal"].date()
            proc_version = self._get_processing_version(item)
            if not grouped.sel(time=date, proc_version=proc_version).values:
                grouped.loc[date, proc_version] = [item]
            else:
                grouped.loc[date, proc_version].append(item)

        # take the latest processing version
        da_arr = grouped.data
        mask = da_arr != None
        proc_version_idx = np.argmax(mask, axis=-1)
        da_arr_select = np.take_along_axis(
            da_arr, proc_version_idx[..., np.newaxis], axis=-1
        )[..., 0]
        grouped = xr.DataArray(
            da_arr_select,
            dims=("time"),
            coords=dict(time=dates),
        )

        # replace date by datetime form first item
        dts = []
        for date in grouped.time.values:
            next_item = next(
                value
                for value in grouped.sel(time=date, idx=0).values
                if value is not None
            )
            dts.append(
                np.datetime64(
                    next_item.properties["datetime_nominal"].replace(tzinfo=None)
                ).astype("datetime64[ns]")
            )
        grouped = grouped.assign_coords(time=dts)

        return grouped

    def _get_processing_version(self, item: pystac.Item) -> float:
        return float(
            item.properties.get(
                "processing:version",
                item.properties.get("s2:processing_baseline", "1.0"),
            )
        )

    def generate_cube(self, access_params: xr.DataArray, **open_params) -> xr.Dataset:
        utm_tile_id = defaultdict(list)
        for tile_id in access_params.tile_id.values:
            item = next(
                value["item"]
                for value in access_params.sel(tile_id=tile_id, idx=0).values.ravel()
                if value is not None
            )
            crs = item.assets["AOT_10m"].extra_fields["proj:code"]
            utm_tile_id[crs].append(tile_id)

        list_ds_utm = []
        for crs, tile_ids in utm_tile_id.items():
            access_params_sel = access_params.sel(tile_id=tile_ids)
            list_ds_utm.append(self._sort_in_utm(access_params_sel, crs, **open_params))

        ds_final = _merge_utm_zones(list_ds_utm, **open_params)
        if open_params.get("apply_scaling", True):
            ds_final = apply_offset_scaling_stack_mode(ds_final, access_params)

        return ds_final

    def _sort_in_utm(
        self,
        access_params_sel: xr.DataArray,
        crs_utm: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset:
        items_bbox = _get_bounding_box(access_params_sel)
        final_bbox = reproject_bbox(open_params["bbox"], open_params["crs"], crs_utm)

        list_ds_asset = []
        for asset_name in access_params_sel.asset_name.values:
            asset_ds = _create_empty_dataset(
                access_params_sel, asset_name, items_bbox, final_bbox
            )
            for dt_idx, dt in enumerate(access_params_sel.time.values):
                for tile_id in access_params_sel.tile_id.values:
                    list_ds_idx = []
                    for idx in access_params_sel.idx.values:
                        params = access_params_sel.sel(
                            tile_id=tile_id, asset_name=asset_name, time=dt, idx=idx
                        ).item()
                        if not params:
                            continue
                        try:
                            ds = self.open_data(
                                params, opener_id, data_type, **open_params
                            )
                        except Exception as e:
                            LOG.error(
                                f"An error occurred: {e}. Data could not be opened "
                                f"with parameters {params}"
                            )
                            continue
                        ds = clip_dataset_by_bbox(ds, final_bbox)
                        if any(size == 0 for size in ds.sizes.values()):
                            continue
                        list_ds_idx.append(ds)
                    if not list_ds_idx:
                        continue
                    ds = mosaic_spatial_take_first(list_ds_idx)
                    asset_ds = self._sort_in(asset_ds, asset_name, ds, dt_idx)
            list_ds_asset.append(asset_ds)

        return merge_datasets(list_ds_asset)

    def _sort_in(self, asset_ds, asset_name, ds, dt_idx):
        xmin = asset_ds.indexes["x"].get_loc(ds.x[0].item())
        xmax = asset_ds.indexes["x"].get_loc(ds.x[-1].item())
        ymin = asset_ds.indexes["y"].get_loc(ds.y[0].item())
        ymax = asset_ds.indexes["y"].get_loc(ds.y[-1].item())
        asset_ds[asset_name][dt_idx, ymin : ymax + 1, xmin : xmax + 1] = ds["band_1"]
        return asset_ds


def _merge_utm_zones(list_ds_utm: list[xr.Dataset], **open_params) -> xr.Dataset:
    resampled_list_ds = []
    ds = list_ds_utm[0]
    target_gm = GridMapping.from_dataset(ds)
    spatial_res = open_params["spatial_res"]
    if not isinstance(spatial_res, tuple):
        spatial_res = (spatial_res, spatial_res)
    diff_crs = pyproj.CRS.from_string(open_params["crs"]) != target_gm.crs
    diff_res = (
        ds.x[1] - ds.x[0] != spatial_res[0] or abs(ds.y[1] - ds.y[0]) != spatial_res[1]
    )
    if diff_crs or diff_res:
        target_gm = get_gridmapping(
            open_params["bbox"],
            open_params["spatial_res"],
            open_params["crs"],
            open_params.get("tile_size", TILE_SIZE),
        )
    for ds in list_ds_utm:
        resampled_list_ds.append(_resample_dataset_soft(ds, target_gm))
    return mosaic_spatial_take_first(resampled_list_ds)


def _resample_dataset_soft(
    ds: xr.Dataset,
    target_gm: GridMapping,
    fill_value: float | int = None,
    interpolation: str = "nearest",
) -> xr.Dataset:
    source_gm = GridMapping.from_dataset(ds)
    if source_gm.is_close(target_gm):
        return ds
    if target_gm.crs == source_gm.crs:
        var_configs = {}
        for var in ds.data_vars:
            var_configs[var] = dict(recover_nan=True)
        ds = affine_transform_dataset(
            ds, target_gm=target_gm, gm_name="spatial_ref", var_configs=var_configs
        )
    else:
        if fill_value is None:
            fill_value = SENTINEL2_FILL_VALUE
        ds = reproject_dataset(
            ds, target_gm=target_gm, fill_value=fill_value, interpolation=interpolation
        )
    return ds


def _get_bounding_box(access_params: xr.DataArray) -> list[float | int]:
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for tile_id in access_params.tile_id.values:
        params = next(
            value
            for value in access_params.sel(tile_id=tile_id).values.flatten()
            if value is not None
        )
        bbox = params["item"].assets[params["name"]].extra_fields["proj:bbox"]
        if xmin > bbox[0]:
            xmin = bbox[0]
        if ymin > bbox[1]:
            ymin = bbox[1]
        if xmax < bbox[2]:
            xmax = bbox[2]
        if ymax < bbox[3]:
            ymax = bbox[3]
    return [xmin, ymin, xmax, ymax]


def _create_empty_dataset(
    access_params: xr.DataArray,
    asset_name: str,
    items_bbox: list[float | int],
    final_bbox: list[float | int],
    spatial_res: int | float = None,
) -> xr.Dataset:
    params = next(
        value
        for value in access_params.sel(asset_name=asset_name).values.flatten()
        if value is not None
    )
    if spatial_res is None:
        spatial_res = params["item"].assets[params["name"]].extra_fields["gsd"]
    crs = params["item"].assets[params["name"]].extra_fields["proj:code"]
    half_res = spatial_res / 2

    y_start = items_bbox[3] - spatial_res * (
        (items_bbox[3] - final_bbox[3]) // spatial_res
    )
    y_end = items_bbox[1] + spatial_res * (
        (final_bbox[1] - items_bbox[1]) // spatial_res
    )
    y = np.arange(y_start - half_res, y_end, -spatial_res)
    x_end = items_bbox[2] - spatial_res * (
        (items_bbox[2] - final_bbox[2]) // spatial_res
    )
    x_start = items_bbox[0] + spatial_res * (
        (final_bbox[0] - items_bbox[0]) // spatial_res
    )
    x = np.arange(x_start + half_res, x_end, spatial_res)

    empty_data = da.zeros(
        (access_params.sizes["time"], len(y), len(x)),
        dtype=np.uint16,
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
