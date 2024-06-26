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
import warnings

import planetary_computer as pc
import pystac
import pystac_client
import requests
import xarray as xr
from xcube.core.store import (
    DATASET_TYPE,
    MULTI_LEVEL_DATASET_TYPE,
    DatasetDescriptor,
    DataStore,
    DataStoreError,
    DataTypeLike,
)
from xcube.core.store.fs.impl.fs import S3FsAccessor
from xcube.util.jsonschema import JsonObjectSchema, JsonStringSchema

from .constants import (
    DATA_OPENER_ID,
    STAC_OPEN_PARAMETERS,
    STAC_SEARCH_PARAMETERS,
)
from .href_parse import _decode_href
from .opener import HttpsDataOpener, S3DataOpener
from .utils import (
    _convert_datetime2str,
    _convert_str2datetime,
    _get_attrs_from_item,
    _list_assets_from_item,
    _get_formats_from_assets,
    _get_formats_from_item,
    _get_opener_id,
    _search_nonsearchable_catalog,
    _update_nested_dict,
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

        # if Microsoft Planetary Computer STAC API is used, href needs
        # to be signed with DAD token
        # (https://planetarycomputer.microsoft.com/docs/concepts/sas/)
        if self._url_mod == "https://planetarycomputer.microsoft.com/api/stac/v1/":
            self._pc = True
        else:
            self._pc = False

        self._https_opener = None
        self._s3_opener = None
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
        return "dataset", "mldataset"

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        item = self._access_item(data_id)
        formats = _get_formats_from_item(item)
        if len(formats) == 1 and formats[0] == "geotiff":
            return "mldataset", "dataset"
        else:
            return ("dataset",)

    def get_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        self._assert_valid_data_type(data_type)
        for item in self._catalog.get_items(recursive=True):
            data_id = self._get_data_id_from_item(item)
            if include_attrs is None:
                yield data_id
            else:
                attrs = _get_attrs_from_item(item, include_attrs)
                yield data_id, attrs

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if self._is_valid_data_type(data_type):
            return data_id in self.list_data_ids()
        return False

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        self._assert_valid_data_type(data_type)
        if data_id is not None and not self.has_data(data_id, data_type=data_type):
            raise DataStoreError(f"Data resource {data_id!r} is not available.")
        return DATA_OPENER_ID

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        self._assert_valid_opener_id(opener_id)
        if data_id is not None:
            item = self._access_item(data_id)
            formats = _get_formats_from_item(item)
            if len(formats) != 1:
                warnings.warn(
                    f"The data ID '{data_id}' contains the formats {list(formats)}. "
                    "Please, do not specify 'opener_id' as multiple openers "
                    "will be used."
                )
            elif opener_id is not None:
                opener_id_format = opener_id.split(":")[1]
                if formats[0] != opener_id_format:
                    warnings.warn(
                        f"The data ID '{data_id}' contains the format '{formats[0]}', "
                        f"but 'opener_id' is set to '{opener_id}'. The 'opener_id' "
                        "will be changed in the open_data method."
                    )

        return JsonObjectSchema(
            properties=dict(**STAC_OPEN_PARAMETERS),
            required=[],
            additional_properties=False,
        )

    def open_data(
        self, data_id: str, opener_id: str = None, **open_params
    ) -> xr.Dataset:
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        self._assert_valid_opener_id(opener_id)
        item = self._access_item(data_id)
        if self._pc:
            item = pc.sign(item)
        return self._build_dataset(item, opener_id=opener_id, **open_params)

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DatasetDescriptor:
        self._assert_valid_data_type(data_type)
        item = self._access_item(data_id)

        # prepare metadata
        time_range = (None, None)
        if "start_datetime" in item.properties and "end_datetime" in item.properties:
            time_range = (
                _convert_datetime2str(
                    _convert_str2datetime(item.properties["start_datetime"]).date()
                ),
                _convert_datetime2str(
                    _convert_str2datetime(item.properties["end_datetime"]).date()
                ),
            )
        elif "datetime" in item.properties:
            time_range = (
                _convert_datetime2str(
                    _convert_str2datetime(item.properties["datetime"]).date()
                ),
                None,
            )
        metadata = dict(bbox=item.bbox, time_range=time_range)
        return DatasetDescriptor(data_id, **metadata)

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DatasetDescriptor]:
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
                f"Data type must be {DATASET_TYPE!r} or {MULTI_LEVEL_DATASET_TYPE!r}, "
                f"but got {data_type!r}."
            )

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
        if opener_id is not None and opener_id not in DATA_OPENER_ID:
            raise DataStoreError(
                f"Data opener identifier must be one of "
                f"{DATA_OPENER_ID!r}, but got {opener_id!r}."
            )

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

    def _get_url_from_item(self, item: pystac.Item) -> str:
        """Extracts the URL an item object.

        Args:
            item: Item object

        Returns:
            the URL of an item.
        """
        links = [link for link in item.links if link.rel == "self"]
        assert len(links) == 1
        return links[0].href

    def _get_data_id_from_item(self, item: pystac.Item) -> str:
        """Extracts the data ID from an item object.

        Args:
            item: Item object

        Returns:
            data ID consisting the URL section of an item
            following the catalog URL.
        """
        return self._get_url_from_item(item).replace(self._url_mod, "")

    def _build_dataset(
        self, item: pystac.Item, opener_id: str = None, **open_params
    ) -> Union[xr.Dataset,]:
        """Builds a dataset where the data variable names correspond
        to the asset keys. If the loaded data consists of multiple
        data variables, the variable name follows the structure
        '<asset_key>_<data_variable_name>'

        Args:
            assets: iterator over assets stored in an item
            opener_id: Data opener identifier. Defaults to None.

        Returns:
            Dataset representation of the data resources identified
            by *data_id* and *open_params*.
        """
        asset_names = open_params.pop("asset_names", None)
        assets = _list_assets_from_item(item, asset_names=asset_names)
        formats = _get_formats_from_assets(assets)

        ds = xr.Dataset()
        for asset in assets:
            protocol, root, fs_path, storage_options = _decode_href(asset.href)
            self._storage_options_s3 = _update_nested_dict(
                self._storage_options_s3, storage_options
            )
            opener_id_asset = _get_opener_id(
                asset, formats, protocol, opener_id=opener_id
            )
            if protocol == "https":
                opener = self._get_https_opener(root, opener_id_asset)
                ds_asset = opener.open_data(data_id=fs_path, **open_params)
            elif protocol == "s3":
                opener = self._get_s3_opener(
                    root, opener_id_asset, storage_options=self._storage_options_s3
                )
                ds_asset = opener.open_data(data_id=fs_path, **open_params)
            else:
                url = self._get_url_from_item(item)
                raise DataStoreError(
                    f"Only 's3' and 'https' protocols are supported, not '{protocol}'. "
                    f"The asset '{asset.extra_fields['id']}' has a href '{asset.href}'."
                    f" The item's url is given by '{url}'."
                )

            for varname, da in ds_asset.data_vars.items():
                if len(ds_asset) == 1:
                    key = asset.extra_fields["id"]
                else:
                    key = f"{asset.extra_fields['id']}_{varname}"
                ds[key] = da
        return ds

    def _get_s3_opener(
        self, root: str, opener_id: str, storage_options: dict
    ) -> S3DataOpener:
        if self._s3_opener is None:
            self._s3_opener = S3DataOpener(
                root, opener_id, storage_options=storage_options
            )
        else:
            if not self._s3_opener.root == root:
                warnings.warn(
                    f"The bucket '{self._s3_opener.root}' of the "
                    f"S3 object storage changed to '{root}'. "
                    "A new s3 data opener will be initialized."
                )
                self._s3_opener = S3DataOpener(
                    root, opener_id, storage_options=storage_options
                )
        return self._s3_opener

    def _get_https_opener(self, root: str, opener_id: str) -> HttpsDataOpener:
        if self._https_opener is None:
            self._https_opener = HttpsDataOpener(root, opener_id)
        else:
            if not self._https_opener.root == root:
                warnings.warn(
                    f"The root '{self._https_opener.root}' of the "
                    f"https data opener changed to '{root}'. "
                    "A new https data opener will be initialized."
                )
                self._https_opener = HttpsDataOpener(root, opener_id)
        return self._https_opener
