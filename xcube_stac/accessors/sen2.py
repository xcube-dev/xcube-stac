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

import re
from typing import Sequence

import boto3
import dask
import numpy as np
import pystac
import rasterio.session
import rioxarray
import xarray as xr

from xcube_stac.constants import LOG, TILE_SIZE, CONVERSION_FACTOR_DEG_METER
from xcube_stac.accessor import StacItemAccessor
from xcube_stac.stac_extension.raster import apply_offset_scaling, get_stac_extension
from xcube_atac.utils import (
    update_dict,
    list_assets_from_item,
    normalize_grid_mapping,
    rename_dataset,
    normalize_crs,
)

SENITNEL2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]
SENITNEL2_L2A_BANDS = SENITNEL2_BANDS + ["AOT", "SCL", "WVP"]
SENTINEL2_FILL_VALUE = 0
SENITNEL2_L2A_BANDS.remove("B10")
SENTINEL2_BAND_RESOLUTIONS = np.array([10, 20, 60])
SENTINEL2_REGEX_ASSET_NAME = "^[A-Z]{3}_[0-9]{2}m$"


class Sen2CdseStacItemAccessor(StacItemAccessor):
    """Provides methods for accessing the data of one general STAC Item"""

    def __init__(self, **storage_options_s3):
        self.session = rasterio.session.AWSSession(
            aws_unsigned=storage_options_s3["anon"],
            endpoint_url=storage_options_s3["client_kwargs"]["endpoint_url"].split(
                "//"
            )[1],
            aws_access_key_id=storage_options_s3["key"],
            aws_secret_access_key=storage_options_s3["secret"],
        )
        self.env = rasterio.env.Env(session=self.session, AWS_VIRTUAL_HOSTING=False)
        # keep the rasterio environment open so that the data can be accessed
        # when plotting or writing the data
        self.env = self.env.__enter__()
        # dask multi-threading needs to be turned off, otherwise the GDAL
        # reader for JP2 raises error.
        dask.config.set(scheduler="single-threaded")
        # need boto2 client to read xml meta data remotely
        self.s3_boto = boto3.client(
            "s3",
            endpoint_url=storage_options_s3["client_kwargs"]["endpoint_url"],
            aws_access_key_id=storage_options_s3["key"],
            aws_secret_access_key=storage_options_s3["secret"],
            region_name="default",
        )

    @staticmethod
    def open_asset(asset: pystac.Asset, **open_params) -> xr.Dataset:
        return rioxarray.open_rasterio(
            asset.href,
            chunks=dict(),
            band_as_variable=True,
        )

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        apply_scaling = open_params.pop("apply_scaling", False)
        assets = self._list_assets_from_item(item, **open_params)
        dss = [self.open_asset(asset) for asset in assets]
        return self._combiner_function(
            dss,
            item=item,
            assets=assets,
            apply_scaling=apply_scaling,
        )

    @staticmethod
    def _list_assets_from_item(item: pystac.Item, **open_params) -> list[pystac.Asset]:
        """Select and return a list of assets from a STAC item based on specified
        asset names and spatial resolution.

        If no asset names are provided, a default set is used. The method attempts to
        match asset names exactly; if an exact match is not found, it tries to append
        a spatial resolution suffix (e.g., "_10m") based on the closest available
        resolution to the requested spatial resolution.

        Args:
            item: The STAC item containing the assets to filter.
            **open_params: Optional parameters to control asset selection:
                - asset_names (list[str], optional): List of desired asset keys.
                    Defaults to SENITNEL2_L2A_BANDS.
                - crs (str or CRS-like, optional): Coordinate reference system of
                  the query area.
                - spatial_res (float, optional): Desired spatial resolution in meters
                  or degrees, depending on the request CRS. Defaults to 10 if not
                  provided.

        Returns:
            Filtered list of assets matching the requested names and spatial resolution.
            Each asset's extra_fields is augmented with:
                - 'id': the selected asset name (including spatial resolution
                    suffix if applied).
                - 'id_origin': the original requested asset name.
                - 'format_id': the asset's format identifier extracted from the
                    asset metadata.
        """
        asset_names = open_params.get("asset_names")
        if not asset_names:
            asset_names = SENITNEL2_L2A_BANDS

        if "crs" in open_params:
            crs = normalize_crs(open_params["crs"])
            if crs.is_geographic:
                spatial_res_final = (
                    open_params["spatial_res"] * CONVERSION_FACTOR_DEG_METER
                )
            else:
                spatial_res_final = open_params["spatial_res"]
        else:
            spatial_res_final = open_params.get("spatial_res", 10)

        assets_sel = []
        for i, asset_name in enumerate(asset_names):
            asset_name_res = asset_name
            if not re.fullmatch(SENTINEL2_REGEX_ASSET_NAME, asset_name):
                res_diff = abs(spatial_res_final - SENTINEL2_BAND_RESOLUTIONS)
                for spatial_res in SENTINEL2_BAND_RESOLUTIONS[np.argsort(res_diff)]:
                    asset_name_res = f"{asset_name}_{spatial_res}m"
                    if asset_name_res in item.assets:
                        break
            asset = item.assets[asset_name_res]
            asset.extra_fields["xcube:asset_id"] = asset_name_res
            asset.extra_fields["xcube:asset_id_origin"] = asset_names[i]
            assets_sel.append(asset)
        return assets_sel

    def _combiner_function(
        self,
        dss: Sequence[xr.Dataset],
        item: pystac.Item = None,
        assets: Sequence[pystac.Asset] = None,
        apply_scaling: bool = False,
    ) -> xr.Dataset:
        dss = [
            rename_dataset(ds, asset.extra_fields["xcube_stac:asset_id"])
            for (ds, asset) in zip(dss, assets)
        ]
        if apply_scaling:
            raster_version = get_stac_extension(item)
            dss = [
                apply_offset_scaling(ds, asset, raster_version)
                for (ds, asset) in zip(dss, assets)
            ]
        combined_ds = dss[0].copy()
        for ds in dss[1:]:
            combined_ds.update(ds)
        return normalize_grid_mapping(combined_ds)
