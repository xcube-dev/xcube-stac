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

from typing import Any, Container, Dict, Iterator, Tuple, Union

import numpy as np
import pystac
import pystac_client
import requests
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import (
    DATASET_TYPE,
    MULTI_LEVEL_DATASET_TYPE,
    DatasetDescriptor,
    MultiLevelDatasetDescriptor,
    DataStore,
    DataStoreError,
    DataType,
    DataTypeLike,
)
from xcube.core.store.fs.impl.fs import S3FsAccessor
from xcube.util.jsonschema import JsonObjectSchema

from .constants import (
    DATA_OPENER_IDS,
    STAC_STORE_PARAMETERS,
)
from ._href_parse import decode_href
from .impl import SingleStoreMode
from .impl import StackStoreMode
from ._utils import (
    are_all_assets_geotiffs,
    assert_valid_data_type,
    assert_valid_opener_id,
    get_attrs_from_pystac_object,
    get_data_id_from_pystac_object,
    get_formats_from_assets,
    is_valid_data_type,
    is_valid_ml_data_type,
    is_xcube_server_item,
    list_assets_from_item,
)


_CATALOG_JSON = "catalog.json"


class StacDataStore(DataStore):
    """STAC implementation of the data store.

    Args:
        url: URL to STAC catalog
        storage_options_s3: storage option for 's3' data store
    """

    def __init__(
        self, url: str, stack_mode: Union[bool, str] = False, **storage_options_s3
    ):
        self._url = url
        url_mod = url
        if url_mod[-len(_CATALOG_JSON) :] == "catalog.json":
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

        self._storage_options_s3 = storage_options_s3

        self._stack_mode = stack_mode
        if stack_mode is False:
            self._impl = SingleStoreMode(
                self._catalog, self._url_mod, self._searchable, self._storage_options_s3
            )
        elif stack_mode is True or stack_mode == "odc-stac":
            self._impl = StackStoreMode(
                self._catalog, self._url_mod, self._searchable, self._storage_options_s3
            )
        else:
            raise DataStoreError(
                "Invalid parameterization detected: a boolean or"
                " 'odc-stac', was expected"
            )

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        stac_params = STAC_STORE_PARAMETERS
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
        item = self._impl.access_item(data_id)
        if are_all_assets_geotiffs(item):
            return MULTI_LEVEL_DATASET_TYPE.alias, DATASET_TYPE.alias
        elif is_xcube_server_item(item):
            return MULTI_LEVEL_DATASET_TYPE.alias, DATASET_TYPE.alias
        else:
            return (DATASET_TYPE.alias,)

    def get_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        assert_valid_data_type(data_type)
        is_mldataset = is_valid_ml_data_type(data_type)
        data_ids_obj = self._impl.get_data_ids(is_mldataset)
        for data_id, pystac_obj in data_ids_obj:
            if include_attrs is None:
                yield data_id
            else:
                attrs = get_attrs_from_pystac_object(pystac_obj, include_attrs)
                yield data_id, attrs

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if is_valid_data_type(data_type):
            try:
                item = self._impl.access_item(data_id)
            except requests.exceptions.HTTPError:
                return False
            if is_valid_ml_data_type(data_type):
                return are_all_assets_geotiffs(item) or is_xcube_server_item(item)
            return True
        return False

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        assert_valid_data_type(data_type)

        if data_id is not None:
            if not self.has_data(data_id, data_type=data_type):
                raise DataStoreError(f"Data resource {data_id!r} is not available.")
            item = self._impl.access_item(data_id)
            assets = list_assets_from_item(item)
            if is_xcube_server_item(item):
                protocols = np.array(["s3"])
                formats = ["zarr", "levels"]
            else:
                formats = get_formats_from_assets(assets)
                protocols = []
                for asset in assets:
                    protocol, _, _, _ = decode_href(asset.href)
                    protocols.append(protocol)
                protocols = np.unique(protocols)

        if data_type is None and data_id is None:
            return DATA_OPENER_IDS
        elif data_type is None and data_id is not None:
            return tuple(
                opener_id
                for opener_id in DATA_OPENER_IDS
                if opener_id.split(":")[1] in formats
                and opener_id.split(":")[2] in protocols
            )
        elif data_type is not None and data_id is None:
            data_type = DataType.normalize(data_type)
            return tuple(
                opener_id
                for opener_id in DATA_OPENER_IDS
                if opener_id.split(":")[0] == data_type.alias
            )
        elif data_type is not None and data_id is not None:
            data_type = DataType.normalize(data_type)
            return tuple(
                opener_id
                for opener_id in DATA_OPENER_IDS
                if opener_id.split(":")[0] == data_type.alias
                and opener_id.split(":")[1] in formats
                and opener_id.split(":")[2] in protocols
            )

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        assert_valid_opener_id(opener_id)
        return self._impl.get_open_data_params_schema(
            data_id=data_id, opener_id=opener_id
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        assert_valid_data_type(data_type)
        assert_valid_opener_id(opener_id)
        return self._impl.open_data(
            data_id,
            opener_id=opener_id,
            data_type=data_type,
            **open_params,
        )

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> Union[DatasetDescriptor, MultiLevelDatasetDescriptor]:
        assert_valid_data_type(data_type)
        metadata = self._impl.get_extent(data_id)

        if is_valid_ml_data_type(data_type):
            mlds = self.open_data(data_id, data_type="mldataset")
            return MultiLevelDatasetDescriptor(data_id, mlds.num_levels, **metadata)
        else:
            return DatasetDescriptor(data_id, **metadata)

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[Union[DatasetDescriptor, MultiLevelDatasetDescriptor]]:
        assert_valid_data_type(data_type)
        pystac_objs = self._impl.search_data(**search_params)

        for pystac_obj in pystac_objs:
            data_id = get_data_id_from_pystac_object(
                pystac_obj, catalog_url=self._url_mod
            )
            yield self.describe_data(data_id, data_type=data_type)

    def get_search_params_schema(
        self, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        return self._impl.get_search_params_schema()
