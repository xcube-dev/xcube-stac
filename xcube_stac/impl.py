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

import json
from typing import Any, Iterator, Union

import numpy as np
import odc.stac
import odc.geo
import pystac
import requests
import xarray as xr
from xcube.core.mldataset import (
    MultiLevelDataset,
    CombinedMultiLevelDataset,
    LazyMultiLevelDataset,
)
from xcube.core.store import DataStoreError, DataTypeLike, new_data_store
from xcube.util.jsonschema import JsonObjectSchema


from .constants import (
    FloatInt,
    LOG,
    MLDATASET_FORMATS,
    STAC_OPEN_PARAMETERS,
    STAC_OPEN_PARAMETERS_STACK_MODE,
    STAC_SEARCH_PARAMETERS,
    STAC_SEARCH_PARAMETERS_STACK_MODE,
)
from .href_parse import _decode_href
from .accessor import HttpsDataAccessor, S3DataAccessor
from .utils import (
    _apply_scaling_nodata,
    _convert_datetime2str,
    _extract_params_xcube_server_asset,
    _get_data_id_from_pystac_object,
    _get_format_from_asset,
    _get_formats_from_item,
    _get_resolutions_cog,
    _get_url_from_pystac_object,
    _is_xcube_server_asset,
    _is_valid_ml_data_type,
    _list_assets_from_item,
    _search_collections,
    _search_nonsearchable_catalog,
    _select_xcube_server_asset,
    _update_dict,
    _xarray_rename_vars,
)

_HTTPS_STORE = new_data_store("https")
_OPEN_DATA_PARAMETERS = {
    "open_params_dataset_netcdf": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:netcdf:https"
    ),
    "open_params_dataset_zarr": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:zarr:https"
    ),
    "open_params_dataset_geotiff": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:geotiff:https"
    ),
    "open_params_mldataset_geotiff": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="mldataset:geotiff:https"
    ),
    "open_params_dataset_levels": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="dataset:levels:https"
    ),
    "open_params_mldataset_levels": _HTTPS_STORE.get_open_data_params_schema(
        opener_id="mldataset:levels:https"
    ),
}


class SingleImplementation:
    """Implementations to access single STAC items"""

    def __init__(
        self,
        catalog: pystac.Catalog,
        url_mod: str,
        searchable: bool,
        storage_options_s3: dict,
    ):
        self._catalog = catalog
        self._url_mod = url_mod
        self._searchable = searchable
        self._storage_options_s3 = storage_options_s3
        self._https_accessor = None
        self._s3_accessor = None

    def access_item(self, data_id: str) -> pystac.Item:
        """Access item for a given data ID.

        Args:
            data_id: An identifier of data that is provided by this store.

        Returns:
            item object

        Raises:
            DataStoreError: Error, if the item json cannot be accessed.
        """
        response = requests.request(method="GET", url=f"{self._url_mod}{data_id}")
        if response.ok:
            return pystac.Item.from_dict(
                json.loads(response.text),
                href=f"{self._url_mod}{data_id}",
                root=self._catalog,
                preserve_dict=False,
            )
        else:
            raise DataStoreError(response.raise_for_status())

    def get_data_ids(self, is_mldataset: bool) -> Iterator[tuple[str, pystac.Item]]:
        for item in self._catalog.get_items(recursive=True):
            if is_mldataset:
                formats = _get_formats_from_item(item)
                if not (len(formats) == 1 and formats[0] in MLDATASET_FORMATS):
                    continue
            data_id = _get_data_id_from_pystac_object(item, catalog_url=self._url_mod)
            yield data_id, item

    def get_open_data_params_schema(
        self,
        data_id: str = None,
        opener_id: str = None,
    ):
        properties = {}
        if opener_id is not None:
            key = "_".join(opener_id.split(":")[:2])
            key = f"open_params_{key}"
            properties[key] = _OPEN_DATA_PARAMETERS[key]
        if data_id is not None:
            item = self.access_item(data_id)
            formats = _get_formats_from_item(item)
            for form in formats:
                for key in _OPEN_DATA_PARAMETERS.keys():
                    if form == key.split("_")[-1]:
                        properties[key] = _OPEN_DATA_PARAMETERS[key]
        if not properties:
            properties = _OPEN_DATA_PARAMETERS
        return JsonObjectSchema(
            properties=_update_dict(properties, STAC_OPEN_PARAMETERS, inplace=False),
            required=[],
            additional_properties=False,
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)
        item = self.access_item(data_id)
        ds = self.build_dataset(
            item, opener_id=opener_id, data_type=data_type, **open_params
        )
        ds = _apply_scaling_nodata(ds, item)
        return ds

    def get_extent(self, data_id: str) -> dict:
        item = self.access_item(data_id)

        # prepare metadata
        time_range = (None, None)
        if "start_datetime" in item.properties and "end_datetime" in item.properties:
            time_range = (
                item.properties["start_datetime"],
                item.properties["end_datetime"],
            )
        elif "datetime" in item.properties:
            time_range = (item.properties["datetime"], None)

        return dict(bbox=item.bbox, time_range=time_range)

    def search_data(self, **search_params) -> Iterator[pystac.Item]:
        if self._searchable:
            # rewrite to "datetime"
            time_range = search_params.pop("time_range", None)
            if time_range:
                search_params["datetime"] = "/".join(time_range)
            items = self._catalog.search(**search_params).items()
        else:
            items = _search_nonsearchable_catalog(self._catalog, **search_params)
        return items

    @classmethod
    def get_search_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_PARAMETERS),
            required=[],
            additional_properties=False,
        )

    def build_dataset(
        self,
        item: pystac.Item,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        """Builds a dataset where the data variable names correspond
        to the asset keys. If the loaded data consists of multiple
        data variables, the variable name follows the structure
        '<asset_key>_<data_variable_name>'

        Args:
            item: item/feature
            opener_id: Data opener identifier. Defaults to None.
            data_type: Data type assigning the return value data type.
                May be given as type alias name, as a type, or as a
                :class:`xcube.core.store.DataType` instance.

        Returns:
            Dataset representation of the data resources identified
            by *data_id* and *open_params*.
        """
        asset_names = open_params.pop("asset_names", None)
        assets = _list_assets_from_item(item, asset_names=asset_names)
        if _is_xcube_server_asset(assets):
            assets = _select_xcube_server_asset(
                assets, asset_names=asset_names, data_type=data_type
            )

        list_ds_asset = []
        for asset in assets:
            if _is_xcube_server_asset(asset):
                protocol, root, fs_path, storage_options = (
                    _extract_params_xcube_server_asset(asset)
                )
            else:
                protocol, root, fs_path, storage_options = _decode_href(asset.href)

            if protocol == "s3":
                self._storage_options_s3 = _update_dict(
                    self._storage_options_s3, storage_options
                )

            format_id = _get_format_from_asset(asset)
            if opener_id is not None:
                key = "_".join(opener_id.split(":")[:2])
                open_params_asset = open_params.get(f"open_params_{key}", {})
            elif data_type is not None:
                open_params_asset = open_params.get(
                    f"open_params_{data_type}_{format_id}", {}
                )
            else:
                open_params_asset = open_params.get(
                    f"open_params_dataset_{format_id}", {}
                )

            if protocol == "https":
                opener = self._get_https_accessor(root)
                ds_asset = opener.open_data(
                    fs_path,
                    format_id,
                    opener_id=opener_id,
                    data_type=data_type,
                    **open_params_asset,
                )
            elif protocol == "s3":
                opener = self._get_s3_accessor(
                    root, storage_options=self._storage_options_s3
                )
                ds_asset = opener.open_data(
                    fs_path,
                    opener_id=opener_id,
                    data_type=data_type,
                    **open_params_asset,
                )
            else:
                url = _get_url_from_pystac_object(item)
                raise DataStoreError(
                    f"Only 's3' and 'https' protocols are supported, not {protocol!r}. "
                    f"The asset {asset.extra_fields['id']!r} has a href {asset.href!r}."
                    f" The item's url is given by {url!r}."
                )

            if isinstance(ds_asset, MultiLevelDataset):
                var_names = list(ds_asset.base_dataset.keys())
            else:
                var_names = list(ds_asset.keys())
            if len(var_names) == 1:
                name_dict = {var_names[0]: asset.extra_fields["id"]}
            else:
                name_dict = {
                    var_name: f"{asset.extra_fields['id']}_{var_name}"
                    for var_name in var_names
                }
            if isinstance(ds_asset, MultiLevelDataset):
                ds_asset = ds_asset.apply(
                    _xarray_rename_vars, dict(name_dict=name_dict)
                )
            else:
                ds_asset = ds_asset.rename_vars(name_dict=name_dict)
            list_ds_asset.append(ds_asset)

        if len(list_ds_asset) == 1:
            ds = list_ds_asset[0]
        else:
            if all(isinstance(ds, MultiLevelDataset) for ds in list_ds_asset):
                ds = CombinedMultiLevelDataset(list_ds_asset)
            else:
                ds = list_ds_asset[0].copy()
                for ds_asset in list_ds_asset[1:]:
                    ds.update(ds_asset)
        return ds

    ##########################################################################
    # Implementation helpers

    def _get_s3_accessor(self, root: str, storage_options: dict) -> S3DataAccessor:
        """This function returns the S3 data accessor associated with the
        bucket *root*. It creates the S3 data accessor only if it is not
        created yet or if *root* changes.

        Args:
            root: bucket of AWS S3 object storage
            storage_options: storage option of the S3 data store

        Returns:
            S3 data opener
        """
        if self._s3_accessor is None:
            self._s3_accessor = S3DataAccessor(root, storage_options=storage_options)
        else:
            if not self._s3_accessor.root == root:
                LOG.debug(
                    f"The bucket {self._s3_accessor.root!r} of the "
                    f"S3 object storage changed to {root!r}. "
                    "A new s3 data opener will be initialized."
                )
                self._s3_accessor = S3DataAccessor(
                    root, storage_options=storage_options
                )
        return self._s3_accessor

    def _get_https_accessor(self, root: str) -> HttpsDataAccessor:
        """This function returns the HTTPS data opener associated with the
        *root* url and the *opener_id*. It creates the HTTPS data opener
        only if it is not created yet or if *root* or *opener_id* changes.

        Args:
            root: root of URL

        Returns:
            HTTPS data opener
        """
        if self._https_accessor is None:
            self._https_accessor = HttpsDataAccessor(root)
        else:
            if not self._https_accessor.root == root:
                LOG.debug(
                    f"The root {self._https_accessor.root!r} of the "
                    f"https data opener changed to {root!r}. "
                    "A new https data opener will be initialized."
                )
                self._https_accessor = HttpsDataAccessor(root)
        return self._https_accessor


class StackImplementation:
    """Implementations to access stacked STAC items within one collection"""

    def __init__(
        self,
        catalog: pystac.Catalog,
        url_mod: str,
        searchable: bool,
        storage_options_s3: dict,
    ):
        self._catalog = catalog
        self._url_mod = url_mod
        self._searchable = searchable
        self._storage_options_s3 = storage_options_s3

    def access_collection(self, data_id: str) -> pystac.Collection:
        """Access collection for a given data ID.

        Args:
            data_id: An identifier of data that is provided by this store.

        Returns:
            collection object.

        Raises:
            DataStoreError: Error, if the item json cannot be accessed.
        """
        prefix = "collections/"
        if prefix in data_id:
            data_id = data_id.replace(prefix, "")
        return self._catalog.get_child(data_id)

    def access_item(self, data_id: str) -> pystac.Item:
        """Access the first item of a collection for a given data ID.

        Args:
            data_id: An identifier of data that is provided by this store.

        Returns:
            item object.

        Raises:
            DataStoreError: Error, if the item json cannot be accessed.
        """
        collection = self.access_collection(data_id)
        return next(collection.get_items())

    def get_data_ids(
        self, is_mldataset: bool
    ) -> Iterator[tuple[str, pystac.Collection]]:
        for collection in self._catalog.get_collections():
            if is_mldataset:
                item = next(collection.get_items())
                formats = _get_formats_from_item(item)
                if not (len(formats) == 1 and formats[0] in MLDATASET_FORMATS):
                    continue
            data_id = _get_data_id_from_pystac_object(
                collection, catalog_url=self._url_mod
            )
            yield data_id, collection

    def get_open_data_params_schema(
        self,
        data_id: str = None,
        opener_id: str = None,
    ):
        return JsonObjectSchema(
            properties=dict(**STAC_OPEN_PARAMETERS_STACK_MODE),
            required=[],
            additional_properties=False,
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        bbox: [FloatInt, FloatInt, FloatInt, FloatInt] = None,
        time_range: [str, str] = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        if time_range is not None:
            "/".join(time_range)
        items = list(
            self._catalog.search(
                collections=[data_id], bbox=bbox, datetime=time_range
            ).items()
        )
        get_mldd = False
        if data_type is None and opener_id is None:
            formats = _get_formats_from_item(
                items[0], asset_names=open_params.get("bands")
            )
            if len(formats) == 1 and formats[0] in MLDATASET_FORMATS:
                get_mldd = True
        if opener_id is None:
            opener_id = ""
        if (
            get_mldd
            or _is_valid_ml_data_type(data_type)
            or opener_id.split(":")[0] == "mldataset"
        ):
            ds = StackModeMultiLevelDataset(data_id, items, **open_params)
        else:
            resolutions = _get_resolutions_cog(
                items[0],
                asset_names=open_params.get("asset_names", None),
                crs=open_params.get("crs", None),
            )
            ds = odc.stac.load(
                items,
                resolution=resolutions[0],
                **open_params,
            )
            ds = _apply_scaling_nodata(ds, items)
        return ds

    def get_extent(self, data_id: str) -> dict:
        collection = self.access_collection(data_id)
        temp_extent = collection.extent.temporal.intervals[0]
        temp_extent_str = [None, None]
        if temp_extent[0] is not None:
            temp_extent_str[0] = _convert_datetime2str(temp_extent[0])
        if temp_extent[1] is not None:
            temp_extent_str[1] = _convert_datetime2str(temp_extent[1])
        return dict(
            bbox=collection.extent.spatial.bboxes[0],
            time_range=temp_extent_str,
        )

    def search_data(self, **search_params) -> Iterator[pystac.Item]:
        return _search_collections(self._catalog, **search_params)

    @classmethod
    def get_search_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_PARAMETERS_STACK_MODE),
            required=[],
            additional_properties=False,
        )


class StackModeMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset for stack-mode.

    Args:
        data_id: data identifier
        items: list of items to be stacked
        open_params: opening parameters of odc.stack.load
    """

    def __init__(
        self,
        data_id: str,
        items: list[pystac.Item],
        **open_params: dict[str, Any],
    ):
        super().__init__(ds_id=data_id)
        self._data_id = data_id
        self._open_params = open_params
        self._items = sorted(items, key=lambda item: item.properties.get("datetime"))

        self._resolutions = _get_resolutions_cog(
            self._items[0],
            asset_names=self._open_params.get("asset_names", None),
            crs=self._open_params.get("crs", None),
        )

    def _get_num_levels_lazily(self):
        return len(self._resolutions)

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        ds = odc.stac.load(
            self._items,
            resolution=self._resolutions[index],
            **self._open_params,
        )
        ds = _apply_scaling_nodata(ds, self._items)
        return ds
