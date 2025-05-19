# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
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

import re

import numpy as np
import pystac
import xarray as xr
from xcube.core.store import DataStoreError

from ._href_parse import decode_href
from .accessor.https import HttpsDataAccessor
from .accessor.s3 import S3DataAccessor
from .accessor.sen2 import (
    SENITNEL2_L2A_BANDS,
    SENTINEL2_BAND_RESOLUTIONS,
    SENTINEL2_REGEX_ASSET_NAME,
    S3Sentinel2DataAccessor,
)
from .constants import MLDATASET_FORMATS, CONVERSION_FACTOR_DEG_METER
from .utils import (
    get_format_from_path,
    get_format_id,
    is_valid_ml_data_type,
    normalize_crs,
)

Accessor = S3Sentinel2DataAccessor | HttpsDataAccessor


class Helper:

    def __init__(self):
        self.s3_accessor = S3DataAccessor

    def list_assets_from_item(
        self, item: pystac.Item, **open_params
    ) -> list[pystac.Asset]:
        asset_names = open_params.get("asset_names")
        assets = []
        for key, asset in item.assets.items():
            format_id = get_format_id(asset)
            if (asset_names is None or key in asset_names) and format_id is not None:
                asset.extra_fields["id"] = key
                asset.extra_fields["id_origin"] = key
                asset.extra_fields["format_id"] = format_id
                assets.append(asset)
        return assets

    def list_format_ids(self, item: pystac.Item, **open_params) -> list[str]:
        assets = self.list_assets_from_item(item, **open_params)
        return list(np.unique([asset.extra_fields["format_id"] for asset in assets]))

    def is_mldataset_available(self, item: pystac.Item, **open_params) -> bool:
        format_ids = self.list_format_ids(item, **open_params)
        return all(format_id in MLDATASET_FORMATS for format_id in format_ids)

    def get_protocols(self, item: pystac.Item, **open_params) -> list[str]:
        params = self.get_data_access_params(item, **open_params)
        return list(np.unique([params[key]["protocol"] for key in params]))

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        assets = self.list_assets_from_item(item, **open_params)
        data_access_params = {}
        for asset in assets:
            protocol, root, fs_path, storage_options = decode_href(asset.href)
            format_id = get_format_id(asset)
            data_access_params[asset.extra_fields["id"]] = dict(
                name=asset.extra_fields["id"],
                name_origin=asset.extra_fields["id_origin"],
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options=storage_options,
                format_id=format_id,
                href=asset.href,
                item=item,
            )
        return data_access_params

    def stack_items(
        self, items: list[pystac.Item], accessor: Accessor, **open_params
    ) -> xr.Dataset:
        raise NotImplementedError("No stacking mode implemented.")


class HelperXcube(Helper):

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        asset_names = open_params.get("asset_names")
        assets = self.list_assets_from_item(item, **open_params)
        opener_id_data_type = open_params.get("opener_id")
        if opener_id_data_type is not None:
            opener_id_data_type = opener_id_data_type.split(":")[0]
        if asset_names is None:
            if is_valid_ml_data_type(open_params.get("data_type")):
                assets = [assets[1]]
            elif is_valid_ml_data_type(opener_id_data_type):
                assets = [assets[1]]
            else:
                assets = [assets[0]]
        elif "analytic_multires" in asset_names and "analytic" in asset_names:
            raise DataStoreError(
                "Xcube server publishes data resources as 'dataset' and "
                "'mldataset' under the asset names 'analytic' and "
                "'analytic_multires'. Please select only one asset in "
                "<asset_names> when opening the data."
            )
        data_access_params = {}
        for asset in assets:
            protocol = asset.extra_fields["xcube:data_store_id"]
            data_store_params = asset.extra_fields["xcube:data_store_params"]
            root = data_store_params["root"]
            storage_options = data_store_params["storage_options"]
            fs_path = asset.extra_fields["xcube:open_data_params"]["data_id"]
            format_id = get_format_from_path(fs_path)
            data_access_params[asset.extra_fields["id"]] = dict(
                name=asset.extra_fields["id"],
                name_origin=asset.extra_fields["id_origin"],
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options=storage_options,
                format_id=format_id,
                href=asset.href,
                item=item,
            )
        return data_access_params

    def is_mldataset_available(self, item: pystac.Item, **open_params) -> bool:
        return True

    def list_format_ids(self, item: pystac.Item, **open_params) -> list[str]:
        return ["zarr", "levels"]

    def get_protocols(self, item: pystac.Item, **open_params) -> list[str]:
        return ["s3"]


class HelperCdse(Helper):

    def __init__(self):
        super().__init__()
        self.s3_accessor = S3Sentinel2DataAccessor

    def list_assets_from_item(
        self, item: pystac.Item, **open_params
    ) -> list[pystac.Asset]:
        asset_names = open_params.get("asset_names")
        if not asset_names:
            asset_names = SENITNEL2_L2A_BANDS

        if "crs" in open_params:
            crs = normalize_crs(open_params["crs"])
            if crs.is_geographic:
                spatial_res = open_params["spatial_res"] * CONVERSION_FACTOR_DEG_METER
            else:
                spatial_res = open_params["spatial_res"]
        else:
            spatial_res = open_params.get("spatial_res", 10)

        assets_sel = []
        for i, asset_name in enumerate(asset_names):
            if not re.fullmatch(SENTINEL2_REGEX_ASSET_NAME, asset_name):
                res_diff = abs(spatial_res - SENTINEL2_BAND_RESOLUTIONS)
                for spatial_res in SENTINEL2_BAND_RESOLUTIONS[np.argsort(res_diff)]:
                    asset_name_res = f"{asset_name}_{spatial_res}m"
                    if asset_name_res in item.assets:
                        break
            asset = item.assets[asset_name_res]
            asset.extra_fields["id"] = asset_name_res
            asset.extra_fields["id_origin"] = asset_names[i]
            asset.extra_fields["format_id"] = get_format_id(asset)
            assets_sel.append(asset)
        return assets_sel

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        assets = self.list_assets_from_item(item, **open_params)
        data_access_params = {}
        for asset in assets:
            protocol, remain = asset.href.split("://")
            # some STAC items show hrefs with s3://DIAS/..., which does not exist;
            # error has been reported.
            root = "eodata"
            fs_path = "/".join(remain.split("/")[1:])
            format_id = get_format_id(asset)
            data_access_params[asset.extra_fields["id"]] = dict(
                name=asset.extra_fields["id"],
                name_origin=asset.extra_fields["id_origin"],
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options={},
                format_id=format_id,
                href=asset.href,
                item=item,
            )
        return data_access_params

    def stack_items(
        self, items: list[pystac.Item], accessor: Accessor, **open_params
    ) -> xr.Dataset:
        # get STAC assets grouped by solar day
        grouped_items = accessor.groupby_solar_day(items)

        # extract access parameters from STAC assets
        access_params = self._group_assets(grouped_items, **open_params)

        # apply mosaicking and stacking
        ds = accessor.generate_cube(access_params, **open_params)

        # add attributes
        # Gather all used STAC item IDs used  in the data cube for each time step
        # and organize them in a dictionary. The dictionary keys are datetime
        # strings, and the values are lists of corresponding item IDs.
        ds.attrs["stac_item_ids"] = dict(
            {
                dt.astype("datetime64[ms]")
                .astype("O")
                .isoformat(): [
                    item.id
                    for item in grouped_items.sel(time=dt).values.flatten()
                    if item is not None
                ]
                for dt in access_params.time.values
            }
        )

        return ds

    def _group_assets(self, grouped_items: xr.DataArray, **open_params) -> xr.DataArray:
        asset_names = open_params.get("asset_names")
        if not asset_names:
            item = next(val for val in grouped_items.values.ravel() if val is not None)
            assets = self.list_assets_from_item(item)
            asset_names = [asset.extra_fields["id_origin"] for asset in assets]

        access_params = xr.DataArray(
            np.empty(
                (
                    grouped_items.sizes["tile_id"],
                    len(asset_names),
                    grouped_items.sizes["time"],
                    grouped_items.sizes["idx"],
                ),
                dtype=object,
            ),
            dims=("tile_id", "asset_name", "time", "idx"),
            coords=dict(
                tile_id=grouped_items["tile_id"],
                asset_name=asset_names,
                time=grouped_items["time"],
                idx=grouped_items["idx"],
            ),
        )
        for dt in grouped_items.time.values:
            for tile_id in grouped_items.tile_id.values:
                for idx in grouped_items.idx.values:
                    item = grouped_items.sel(time=dt, tile_id=tile_id, idx=idx).item()
                    if item is None:
                        continue

                    item_access_params = self.get_data_access_params(
                        item, **open_params
                    )
                    for key, val in item_access_params.items():
                        access_params.loc[tile_id, val["name_origin"], dt, idx] = val

        return access_params
