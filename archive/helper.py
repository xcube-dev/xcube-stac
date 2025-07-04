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

import numpy as np
import pystac
import xarray as xr
from xcube.core.store import DataStoreError

from ._href_parse import decode_href
from .accessor.https import HttpsDataAccessor
from .accessor.s3 import S3DataAccessor
from .accessor.sen2 import S3Sentinel2DataAccessor, list_assets_from_sen2_item
from .accessor.sen3 import S3Sentinel3DataAccessor
from .constants import LOG, MLDATASET_FORMATS
from .utils import (
    _list_assets_from_item,
    get_format_from_path,
    get_format_id,
    is_valid_ml_data_type,
)

Accessor = S3Sentinel2DataAccessor | S3Sentinel3DataAccessor | HttpsDataAccessor


class Helper:

    def __init__(self):
        self.s3_accessor = S3DataAccessor

    def list_assets_from_item(
        self, item: pystac.Item, **open_params
    ) -> list[pystac.Asset]:
        return _list_assets_from_item(item, **open_params)

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
        """Generate a dictionary of data access parameters for all relevant assets in
        a STAC item.

        Args:
            item: The STAC item to process.
            **open_params: Optional parameters passed to filter or modify asset
                selection.

        Returns:
            dict: A mapping from asset IDs to dictionaries containing access
                parameters, including:
                - name: Asset ID used internally
                - name_origin: Original asset ID (may differ in subclasses)
                - protocol: Protocol derived from the asset's href (e.g. 'https', 's3')
                - root: Root path or URL for the storage location
                - fs_path: Filesystem path within the storage
                - storage_options: Additional storage options (e.g.,
                    credentials or config)
                - format_id: Identifier of the asset format
                - href: The asset's full href URL or path
                - item: The original pystac.Item the asset belongs to
        """

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
        """Stack multiple STAC items into a single xarray.Dataset.

        This method is intended to be overridden in subclasses, such as HelperCDSE,
        where the actual stacking logic is implemented for the CDSE STAC catalog.

        Args:
            items: A list of STAC items objects to be stacked.
            accessor: An Accessor instance used to access the underlying data.
            **open_params: Additional opening parameters.

        Raises:
            NotImplementedError: Always raised in this base implementation.

        Returns:
            A single stacked dataset combining all input items
            (in subclass implementations).
        """
        raise NotImplementedError("No stacking mode implemented.")


class HelperXcube(Helper):

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        """Generate a dictionary of data access parameters for selected assets
        in a STAC item.

        This method filters and processes the assets of the provided STAC item
        according to specified opening parameters, extracting the necessary
        connection and format details to enable data access.

        Args:
            item: The STAC item containing assets to be accessed.
            **open_params: Optional parameters influencing asset selection
                and processing:
                - asset_names (list[str], optional): List of asset keys to include.
                    If None, default asset selection rules apply.
                - opener_id (str, optional): Identifier influencing asset
                    filtering by data type.
                - data_type (DataTypeLike, optional): Data type used to filter assets.

        Raises:
            DataStoreError: If mutually exclusive asset names 'analytic' and
            'analytic_multires' are both selected, which is not supported.

        Returns:
            A mapping from asset IDs to dictionaries containing data access
            parameters, including protocol, storage root, file system path,
            storage options, format ID, the asset href, and the original STAC item.
        """
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
        self.s3_accessor = {
            "sentinel-2-l2a": S3Sentinel2DataAccessor,
            "sentinel-3-syn-2-syn-ntc": S3Sentinel3DataAccessor,
        }

    def list_assets_from_item(
        self, item: pystac.Item, **open_params
    ) -> list[pystac.Asset]:
        if "_MSIL2A_" in item.id:
            assets_sel = list_assets_from_sen2_item(item, **open_params)
        elif "_SY_2_SYN_" in item.id:
            assets_sel = _list_assets_from_item(item, **open_params)
        else:
            LOG.warning(
                f"Collection could not be derived from item ID {item.id!r}. "
                f"Item is disregarded."
            )
            assets_sel = []
        return assets_sel

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        """Extract data access parameters for assets within a given STAC item.

        This method collects relevant information needed to access the asset data,
        including protocol, storage root, file system path, format identifier, and
        other metadata. It processes the asset href to determine the protocol and
        path components. Note that some STAC items may contain hrefs with prefixes
        like 's3://DIAS/' which are known issues and handled by setting a default root.

        Args:
            item: The STAC item from which to extract assets and access parameters.
            **open_params: Additional optional opening parameters that control
                asset filtering.

        Returns:
            A dictionary mapping asset IDs to their respective data access
            parameters. Each value is a dict containing:
                - 'name' (str): Asset ID.
                - 'name_origin' (str): Original requested asset name.
                - 'protocol' (str): Access protocol (e.g., 's3', 'https').
                - 'root' (str): Storage root or bucket name; here hardcoded as 'eodata'
                  due to known href issues.
                - 'fs_path' (str): Path to the data within the storage root.
                - 'storage_options' (dict): Additional options for storage access,
                  empty by default.
                - 'format_id' (str | None): Format identifier extracted from the asset.
                - 'href' (str): Original asset href.
                - 'item' (pystac.Item): Reference to the original STAC item.
        """
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
        """Create a stacked xarray Dataset (data cube) from a list of STAC items.

        This method groups STAC items by their solar day, extracts the necessary
        access parameters, and then applies mosaicking and stacking to produce
        a spatiotemporal data cube. It also annotates the resulting dataset with
        metadata about the STAC items used for each time step.

        Args:
            items: List of STAC items to be stacked and mosaicked.
            accessor: An accessor object responsible for grouping,
                mosaicking, and generating the data cube.
            **open_params: Additional opening parameters controlling data processing
                passed down to the accessor methods.

        Returns:
            A dataset representing the stacked data cube.
        """
        # get STAC assets grouped by solar day
        grouped_items = accessor.groupby_solar_day(items)

        # extract access parameters from STAC assets
        access_params = self._group_assets(grouped_items, **open_params)

        # apply mosaicking and stacking
        ds = accessor.generate_cube(access_params, **open_params)

        # add attributes
        # Gather all used STAC item IDs used in the data cube for each time step
        # and organize them in a dictionary. The dictionary keys are datetime
        # strings, and the values are lists of corresponding item IDs.
        ds.attrs["stac_item_ids"] = dict(
            {
                dt.astype("datetime64[ms]")
                .astype("O")
                .isoformat(): [
                    item.id
                    for item in grouped_items.sel(time=dt).values.ravel()
                    if item is not None
                ]
                for dt in access_params.time.values
            }
        )

        return ds

    def _group_assets(self, grouped_items: xr.DataArray, **open_params) -> xr.DataArray:
        """Organize STAC item assets into a structured array of access parameters.

        This method iterates over an array of STAC items grouped by dimensions such
        as time, tile_id, and idx. It extracts data access parameters from each
        item’s assets and organizes them into a new array. The idx dimension is
        necessary because the Sentinel-2 L2A collection can contain multiple STAC
        items (observations) for the same tile_id and time.

        Args:
            grouped_items: An array containing STAC items grouped by spatial and
                temporal dimensions (e.g., time, tile_id, idx).
            **open_params: Additional opening parameters passed to
                `get_data_access_params`, used for asset filtering.

        Returns:
            A 4-dimensional DataArray indexed by ("tile_id", "asset_name", "time",
            "idx"), where each element is a dictionary of data access parameters for
            the corresponding asset.
        """

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
