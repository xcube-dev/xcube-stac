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
from typing import Any, Container, Dict, Iterator, Tuple, Union

import pystac
import pystac_client
import requests
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset, CombinedMultiLevelDataset
from xcube.core.store import (
    DATASET_TYPE,
    MULTI_LEVEL_DATASET_TYPE,
    DatasetDescriptor,
    MultiLevelDatasetDescriptor,
    DataStore,
    DataStoreError,
    DataType,
    DataTypeLike,
    new_data_store,
)
from xcube.core.store.fs.impl.fs import S3FsAccessor
from xcube.util.jsonschema import JsonObjectSchema, JsonStringSchema

from .constants import (
    DATA_OPENER_IDS,
    LOG,
    STAC_OPEN_PARAMETERS,
    STAC_SEARCH_PARAMETERS,
)
from .href_parse import _decode_href
from .accessor import HttpsDataAccessor, S3DataAccessor
from .utils import (
    _get_attrs_from_item,
    _list_assets_from_item,
    _get_format_from_asset,
    _get_formats_from_item,
    _get_url_from_item,
    _search_nonsearchable_catalog,
    _update_dict,
    _xarray_rename_vars,
)


class StacDataStore(DataStore):
    """STAC implementation of the data store.

    Args:
        url: URL to STAC catalog
        storage_options_s3: storage option for 's3' data store
    """

    def __init__(self, url: str, **storage_options_s3):
        self._url = url
        url_mod = url
        if url_mod[-12:] == "catalog.json":
            url_mod = url_mod[:-12]
        if url_mod[-1] != "/":
            url_mod += "/"
        self._url_mod = url_mod

        # if STAC catalog is not searchable, pystac_client
        # falls back to pystac; to prevent warnings from pystac_client
        # use catalog from pystac instead. For more discussion refer to
        # https://github.com/xcube-dev/xcube-stac/issues/5
        catalog = pystac_client.Client.open(url)
        self._searchable = True
        if not catalog.conforms_to("ITEM_SEARCH"):
            catalog = pystac.Catalog.from_file(url)
            self._searchable = False
        self._catalog = catalog
        self._https_accessor = None
        self._s3_accessor = None
        self._storage_options_s3 = storage_options_s3

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        stac_params = dict(url=JsonStringSchema(title="URL to STAC catalog"))
        stac_params["storage_options_s3"] = S3FsAccessor.get_storage_options_schema()
        return JsonObjectSchema(
            description="Describes the parameters of the xcube data store 'stac'.",
            properties=stac_params,
            required=["url"],
            additional_properties=False,
        )

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return DATASET_TYPE.alias, MULTI_LEVEL_DATASET_TYPE.alias

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        if self._are_all_assets_geotiffs(data_id):
            return MULTI_LEVEL_DATASET_TYPE.alias, DATASET_TYPE.alias
        elif self._is_xcube_server_item(data_id):
            return MULTI_LEVEL_DATASET_TYPE.alias, DATASET_TYPE.alias
        else:
            return (DATASET_TYPE.alias,)

    def get_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        self._assert_valid_data_type(data_type)
        for item in self._catalog.get_items(recursive=True):
            if self._is_valid_ml_data_type(data_type):
                formats = _get_formats_from_item(item)
                if not (len(formats) == 1 and formats[0] == "geotiff"):
                    continue
            data_id = self._get_data_id_from_item(item)
            if include_attrs is None:
                yield data_id
            else:
                attrs = _get_attrs_from_item(item, include_attrs)
                yield data_id, attrs

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if self._is_valid_data_type(data_type):
            in_store = True
            try:
                _ = self._access_item(data_id)
            except requests.exceptions.HTTPError:
                in_store = False
            if not in_store:
                return False
            else:
                if self._is_valid_ml_data_type(data_type):
                    return self._are_all_assets_geotiffs(data_id)
                return True
        return False

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        self._assert_valid_data_type(data_type)
        if data_type is None:
            return DATA_OPENER_IDS
        else:
            if not self.has_data(data_id, data_type=data_type):
                raise DataStoreError(f"Data resource {data_id!r} is not available.")
            data_type = DataType.normalize(data_type)
            return tuple(
                opener_id
                for opener_id in DATA_OPENER_IDS
                if opener_id.split(":")[0] == data_type.alias
            )

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        self._assert_valid_opener_id(opener_id)
        store = new_data_store("https")
        properties = store.get_open_data_params_schema(opener_id=opener_id).properties
        return JsonObjectSchema(
            properties=_update_dict(properties, STAC_OPEN_PARAMETERS),
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
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        self._assert_valid_data_type(data_type)
        self._assert_valid_opener_id(opener_id)
        item = self._access_item(data_id)
        return self._build_dataset(
            item, opener_id=opener_id, data_type=data_type, **open_params
        )

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> Union[DatasetDescriptor, MultiLevelDatasetDescriptor]:
        self._assert_valid_data_type(data_type)
        item = self._access_item(data_id)

        # prepare metadata
        time_range = (None, None)
        if "start_datetime" in item.properties and "end_datetime" in item.properties:
            time_range = (
                item.properties["start_datetime"],
                item.properties["end_datetime"],
            )
        elif "datetime" in item.properties:
            time_range = (item.properties["datetime"], None)
        metadata = dict(bbox=item.bbox, time_range=time_range)
        if self._are_all_assets_geotiffs(data_id) and self._is_valid_ml_data_type(
            data_type
        ):
            mlds = self.open_data(data_id)
            return MultiLevelDatasetDescriptor(data_id, mlds.num_levels, **metadata)
        else:
            if self._is_valid_ml_data_type(data_type):
                LOG.info(
                    f"The data ID {data_id!r} also not only assets in geotiff "
                    f"format. Therefore, data_type is set to {DATASET_TYPE.alias!r}"
                )
            return DatasetDescriptor(data_id, **metadata)

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[Union[DatasetDescriptor, MultiLevelDatasetDescriptor]]:
        self._assert_valid_data_type(data_type)
        if self._searchable:
            # rewrite to "datetime"
            time_range = search_params.pop("time_range", None)
            if time_range:
                search_params["datetime"] = "/".join(time_range)
            items = self._catalog.search(**search_params).items()
        else:
            items = _search_nonsearchable_catalog(self._catalog, **search_params)
        for item in items:
            data_id = self._get_data_id_from_item(item)
            yield self.describe_data(data_id, data_type=data_type)

    @classmethod
    def get_search_params_schema(
        cls, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_PARAMETERS),
            required=[],
            additional_properties=False,
        )

    ##########################################################################
    # Implementation helpers

    @classmethod
    def _is_valid_data_type(cls, data_type: DataTypeLike) -> bool:
        """Auxiliary function to check if data type is supported
        by the store.

        Args:
            data_type: Data type that is to be checked.

        Returns:
            bool: True if *data_type* is supported by the store, otherwise False
        """
        return (
            data_type is None
            or DATASET_TYPE.is_super_type_of(data_type)
            or MULTI_LEVEL_DATASET_TYPE.is_super_type_of(data_type)
        )

    @classmethod
    def _assert_valid_data_type(cls, data_type: DataTypeLike):
        """Auxiliary function to assert if data type is supported
        by the store.

        Args:
            data_type: Data type that is to be checked.

        Raises:
            DataStoreError: Error, if *data_type* is not
                supported by the store.
        """
        if not cls._is_valid_data_type(data_type):
            raise DataStoreError(
                f"Data type must be {DATASET_TYPE.alias!r} or "
                f"{MULTI_LEVEL_DATASET_TYPE.alias!r}, but got {data_type!r}."
            )

    @classmethod
    def _is_valid_ml_data_type(cls, data_type: DataTypeLike) -> bool:
        """Auxiliary function to check if data type is a multi-level
        dataset type.

        Args:
            data_type: Data type that is to be checked.

        Returns:
            bool: True if *data_type* is a multi-level dataset type, otherwise False
        """
        return MULTI_LEVEL_DATASET_TYPE.is_super_type_of(data_type)

    @classmethod
    def _assert_valid_opener_id(cls, opener_id: str):
        """Auxiliary function to assert if data opener identified by
        *opener_id* is supported by the store.

        Args:
            opener_id: Data opener identifier

        Raises:
            DataStoreError: Error, if *opener_id* is not
                supported by the store.
        """
        if opener_id is not None and opener_id not in DATA_OPENER_IDS:
            raise DataStoreError(
                f"Data opener identifier must be one of "
                f"{DATA_OPENER_IDS}, but got {opener_id!r}."
            )

    def _are_all_assets_geotiffs(self, data_id: str) -> bool:
        """Auxiliary function to check if all assets are tifs, tiffs, or geotiffs.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Returns: True, if all assets are tifs, tiffs, or geotiffs.

        """
        item = self._access_item(data_id)
        formats = _get_formats_from_item(item)
        return len(formats) == 1 and formats[0] == "geotiff"

    def _is_xcube_server_item(self, data_id: str) -> bool:
        """Auxiliary function to check if the item is published by xcube server.

        Args:
            data_id: An identifier of data that is provided by this store.

        Returns: True, if item is published by xcube server.

        """
        item = self._access_item(data_id)
        assets = _list_assets_from_item(item)
        return self._is_xcube_server_asset(assets)

    def _is_xcube_server_asset(
        self, asset: Union[pystac.Asset, list[pystac.Asset]]
    ) -> bool:
        """Auxiliary function to check if the asset(s) is/are published by xcube server.

        Args:
            asset: a list or single pystac.Asset object

        Returns: True, if the asset(s) is/are published by xcube server.

        """
        if isinstance(asset, list):
            asset = asset[0]
        return "xcube:store_kwargs" in asset.extra_fields

    def _select_xcube_server_asset(
        self,
        assets: list[pystac.Asset],
        asset_names: list[str] = None,
        data_type: DataTypeLike = None,
    ):
        if asset_names is None:
            if self._is_valid_ml_data_type(data_type):
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
        return assets

    def _extract_params_xcube_server_asset(self, asset: pystac.Asset):
        store_kwargs = asset.extra_fields["xcube:store_kwargs"]
        protocol = store_kwargs["data_store_id"]
        root = store_kwargs["root"]
        storage_options = store_kwargs["storage_options"]
        fs_path = asset.extra_fields["xcube:open_kwargs"]["data_id"]
        return protocol, root, fs_path, storage_options

    def _access_item(self, data_id: str) -> Union[pystac.Item, str]:
        """Access item for a given data ID.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Returns:
            item object

        Raises:
            DataStoreError: Error, if the item json cannot be accessed.
        """
        response = requests.request(method="GET", url=f"{self._url_mod}{data_id}")
        if response.status_code == 200:
            return pystac.Item.from_dict(
                json.loads(response.text),
                href=f"{self._url_mod}{data_id}",
                root=self._catalog,
                preserve_dict=False,
            )
        else:
            raise DataStoreError(response.raise_for_status())

    def _get_data_id_from_item(self, item: pystac.Item) -> str:
        """Extracts the data ID from an item object.

        Args:
            item: Item object

        Returns:
            data ID consisting the URL section of an item
            following the catalog URL.
        """
        return _get_url_from_item(item).replace(self._url_mod, "")

    def _build_dataset(
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
            assets: iterator over assets stored in an item
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
        if self._is_xcube_server_asset(assets):
            assets = self._select_xcube_server_asset(
                assets, asset_names=asset_names, data_type=data_type
            )

        list_ds_asset = []
        for asset in assets:
            if self._is_xcube_server_asset(asset):
                protocol, root, fs_path, storage_options = (
                    self._extract_params_xcube_server_asset(asset)
                )
            else:
                protocol, root, fs_path, storage_options = _decode_href(asset.href)
            if protocol == "s3":
                self._storage_options_s3 = _update_dict(
                    self._storage_options_s3, storage_options
                )

            if protocol == "https":
                opener = self._get_https_accessor(root)
            elif protocol == "s3":
                opener = self._get_s3_accessor(
                    root, storage_options=self._storage_options_s3
                )
            else:
                url = _get_url_from_item(item)
                raise DataStoreError(
                    f"Only 's3' and 'https' protocols are supported, not {protocol!r}. "
                    f"The asset {asset.extra_fields['id']!r} has a href {asset.href!r}."
                    f" The item's url is given by {url!r}."
                )
            format_id = _get_format_from_asset(asset)
            ds_asset = opener.open_data(
                fs_path,
                format_id,
                opener_id=opener_id,
                data_type=data_type,
                **open_params,
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

    def _get_https_accessor(self, root: str, opener_id: str) -> HttpsDataAccessor:
        """This function returns the HTTPS data opener associated with the
        *root* url and the *opener_id*. It creates the HTTPS data opener
        only if it is not created yet or if *root* or *opener_id* changes.

        Args:
            root: root of URL
            opener_id: data opener identifier

        Returns:
            HTTPS data opener
        """
        if self.https_accessor is None:
            self.https_accessor = HttpsDataAccessor(root)
        else:
            if not self.https_accessor.root == root:
                LOG.debug(
                    f"The root {self.https_accessor.root!r} of the "
                    f"https data opener changed to {root!r}. "
                    "A new https data opener will be initialized."
                )
                self.https_accessor = HttpsDataAccessor(root)
        return self.https_accessor
