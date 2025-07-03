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

from collections.abc import Container, Iterator
from typing import Any

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
    DataStore,
    DataStoreError,
    DataType,
    DataTypeLike,
    MultiLevelDatasetDescriptor,
    new_data_store,
)
from xcube.util.jsonschema import JsonObjectSchema, JsonBooleanSchema

from .constants import (
    CDSE_S3_ENDPOINT,
    CDSE_STAC_URL,
    DATA_OPENER_IDS,
    MAP_FILE_EXTENSION_FORMAT,
    PROTOCOLS,
    SCHEMA_URL,
    SCHEMA_S3_STORE,
    SCHEMA_ASSET_NAMES,
    SCHEMA_TIME_RANGE,
    SCHEMA_BBOX,
    SCHEMA_COLLECTIONS,
    SCHEMA_ADDITIONAL_QUERY,
    SCHEMA_APPLY_SCALING,
    SCHEMA_SPATIAL_RES,
    LOG,
)
from .utils import (
    assert_valid_data_type,
    assert_valid_opener_id,
    get_attrs_from_pystac_object,
    get_data_id_from_pystac_object,
    is_valid_data_type,
    is_valid_ml_data_type,
    modify_catalog_url,
    update_dict,
    access_item,
    is_mldataset_available,
    list_format_ids,
    list_protocols,
    search_items,
)
from .accessors.base import BaseStacItemAccessor
from .accessors.sen2 import Sen2CdseStacItemAccessor
from modes import StackMode


class StacDataStore(DataStore):
    """STAC implementation of the data store.

    Args:
        url: URL to STAC catalog
        storage_options_s3: storage option for 's3' data store
    """

    def __init__(self, url: str, **storage_options_s3):
        self._url = modify_catalog_url(url)
        self._storage_options_s3 = storage_options_s3
        if not hasattr(self, "_accessor"):
            self._accessor = BaseStacItemAccessor(**storage_options_s3)

        # if STAC catalog is not searchable, pystac_client
        # falls back to pystac; to prevent warnings from pystac_client
        # use catalog from pystac instead. For more discussion refer to
        # https://github.com/xcube-dev/xcube-stac/issues/5
        catalog = pystac_client.Client.open(url, timeout=3600)
        self._searchable = True
        if not catalog.conforms_to("ITEM_SEARCH"):
            catalog = pystac.Catalog.from_file(url)
            self._searchable = False
        self._catalog = catalog

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            description="Describes the parameters of the xcube data store 'stac'.",
            properties=dict(url=SCHEMA_URL, **SCHEMA_S3_STORE),
            required=["url"],
            additional_properties=False,
        )

    @classmethod
    def get_data_types(cls) -> tuple[str, ...]:
        return DATASET_TYPE.alias, MULTI_LEVEL_DATASET_TYPE.alias

    def get_data_types_for_data(self, data_id: str) -> tuple[str, ...]:
        url = f"{self._url}/{data_id}"
        item = access_item(url, self._catalog)
        if self._is_mldataset_available(item):
            return DATASET_TYPE.alias, MULTI_LEVEL_DATASET_TYPE.alias
        else:
            return (DATASET_TYPE.alias,)

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        include_attrs: Container[str] | bool = False,
    ) -> Iterator[str | tuple[str, dict[str, Any]], None]:
        assert_valid_data_type(data_type)
        for item in self._catalog.get_items(recursive=True):
            if is_valid_ml_data_type(data_type):
                if not self._is_mldataset_available(item):
                    continue
            data_id = get_data_id_from_pystac_object(item, catalog_url=self._url)
            if not include_attrs:
                yield data_id
            else:
                attrs = get_attrs_from_pystac_object(item, include_attrs)
                yield data_id, attrs

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if is_valid_data_type(data_type):
            try:
                url = f"{self._url}/{data_id}"
                item = access_item(url, self._catalog)
            except requests.exceptions.HTTPError:
                return False
            if is_valid_ml_data_type(data_type):
                return self._is_mldataset_available(item)
            return True
        return False

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> tuple[str, ...]:
        assert_valid_data_type(data_type)

        if data_id is not None:
            if not self.has_data(data_id, data_type=data_type):
                raise DataStoreError(f"Data resource {data_id!r} is not available.")
            url = f"{self._url}/{data_id}"
            item = access_item(url, self._catalog)
            protocols = list_protocols(item)
            format_ids = list_format_ids(item)
        else:
            protocols = PROTOCOLS
            format_ids = list(np.unique(list(MAP_FILE_EXTENSION_FORMAT.values())))

        return self._select_opener_id(protocols, format_ids, data_type=data_type)

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        assert_valid_opener_id(opener_id)
        if data_id is not None and opener_id is None:
            opener_id = self.get_data_opener_ids(data_id=data_id)[0]

        if opener_id is not None:
            store = new_data_store("https")
            params_schema = store.get_open_data_params_schema(opener_id=opener_id)
            params_properties = params_schema.properties
            params_required = params_schema.required
        else:
            params_properties = {}
            params_required = []

        return JsonObjectSchema(
            properties=dict(
                asset_names=SCHEMA_ASSET_NAMES,
                apply_scaling=SCHEMA_APPLY_SCALING,
                **params_properties,
            ),
            required=[] + params_required,
            additional_properties=False,
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        # check input parameter
        assert_valid_data_type(data_type)
        assert_valid_opener_id(opener_id)
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)

        # access item and open with accessor
        url = f"{self._url}/{data_id}"
        item = access_item(url, self._catalog)
        return self._accessor.open_item(
            item,
            opener_id=opener_id,
            data_type=data_type,
            **open_params,
        )

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DatasetDescriptor | MultiLevelDatasetDescriptor:
        assert_valid_data_type(data_type)

        # get extent from item
        url = f"{self._url}/{data_id}"
        item = access_item(url, self._catalog)
        time_range = (None, None)
        if "start_datetime" in item.properties and "end_datetime" in item.properties:
            time_range = (
                item.properties["start_datetime"],
                item.properties["end_datetime"],
            )
        elif "datetime" in item.properties:
            time_range = (item.properties["datetime"], None)

        if is_valid_ml_data_type(data_type):
            mlds = self.open_data(data_id, data_type="mldataset")
            return MultiLevelDatasetDescriptor(
                data_id, mlds.num_levels, bbox=item.bbox, time_range=time_range
            )
        else:
            return DatasetDescriptor(data_id, bbox=item.bbox, time_range=time_range)

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DatasetDescriptor | MultiLevelDatasetDescriptor]:
        assert_valid_data_type(data_type)
        schema = self.get_search_params_schema()
        schema.validate_instance(search_params)
        items = search_items(self._catalog, self._searchable, **search_params)

        for item in items:
            data_id = get_data_id_from_pystac_object(item, catalog_url=self._url)
            yield self.describe_data(data_id, data_type=data_type)

    def get_search_params_schema(
        self, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                time_range=SCHEMA_TIME_RANGE,
                bbox=SCHEMA_BBOX,
                collections=SCHEMA_COLLECTIONS,
                query=SCHEMA_ADDITIONAL_QUERY,
            ),
            required=[],
            additional_properties=False,
        )

    @staticmethod
    def _select_opener_id(
        protocols: list[str],
        format_ids: list[str],
        data_type: DataTypeLike = None,
    ):
        if data_type is None:
            return tuple(
                opener_id
                for opener_id in DATA_OPENER_IDS
                if opener_id.split(":")[1] in format_ids
                and opener_id.split(":")[2] in protocols
            )
        else:
            data_type = DataType.normalize(data_type)
            return tuple(
                opener_id
                for opener_id in DATA_OPENER_IDS
                if opener_id.split(":")[0] == data_type.alias
                and opener_id.split(":")[1] in format_ids
                and opener_id.split(":")[2] in protocols
            )

    @staticmethod
    def _is_mldataset_available(item: pystac.Item) -> bool:
        return is_mldataset_available(item)


class StacXcubeDataStore(StacDataStore):
    """STAC implementation of the data store for xcube STAC API."""

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            description="Describes the parameters of the xcube data store 'stac-xcube'.",
            properties=dict(url=SCHEMA_URL, **SCHEMA_S3_STORE),
            required=["url"],
            additional_properties=False,
        )

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> tuple[str, ...]:
        assert_valid_data_type(data_type)
        protocols = ["s3"]
        format_ids = ["zarr", "levels"]

        return self._select_opener_id(protocols, format_ids, data_type=data_type)

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        # check input parameter
        assert_valid_data_type(data_type)
        assert_valid_opener_id(opener_id)
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)

        # access item and open with accessor
        url = f"{self._url}/{data_id}"
        item = access_item(url, self._catalog)

        # decide between levels and zarr
        asset_names = open_params.pop("asset_names", None)
        opener_id_data_type = open_params.get("opener_id")
        if opener_id_data_type is not None:
            opener_id_data_type = opener_id_data_type.split(":")[0]
        if asset_names is None:
            if is_valid_ml_data_type(data_type):
                asset_names = ["analytic_multires"]
            elif is_valid_ml_data_type(opener_id_data_type):
                asset_names = ["analytic_multires"]
            else:
                asset_names = ["analytic"]
        elif "analytic_multires" in asset_names and "analytic" in asset_names:
            raise DataStoreError(
                "Xcube server publishes data resources as 'dataset' and "
                "'mldataset' under the asset names 'analytic' and "
                "'analytic_multires'. Please select only one asset in "
                "<asset_names> when opening the data."
            )
        return self._accessor.open_item(
            item,
            opener_id=opener_id,
            data_type=data_type,
            assert_names=asset_names,
            **open_params,
        )

    @staticmethod
    def _is_mldataset_available(item: pystac.Item) -> bool:
        return True


class StacCdseDataStore(StacDataStore):
    """STAC implementation of the data store for CDSE STAC API."""

    def __init__(self, **storage_options_s3):
        storage_options_s3 = update_dict(
            storage_options_s3,
            dict(anon=False, client_kwargs=dict(endpoint_url=CDSE_S3_ENDPOINT)),
        )
        self._accessor = Sen2CdseStacItemAccessor(**storage_options_s3)
        super().__init__(url=CDSE_STAC_URL, **storage_options_s3)

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            description="Describes the parameters of the xcube data store 'stac-cdse'.",
            properties=dict(**SCHEMA_S3_STORE),
            required=[],
            additional_properties=False,
        )

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> tuple[str, ...]:
        LOG.info(
            "In the 'stac-cdse' data store, data openers are specific to each "
            "Sentinel mission. Returning a generic opener ID."
        )
        # We return a generic opener ID here. In this data store, specific data
        # accessors are delegated based on the data ID to handle different
        # Sentinel missions transparently.
        return ("dataset:format:stac-cdse",)

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                asset_names=SCHEMA_ASSET_NAMES,
                spatial_res=SCHEMA_SPATIAL_RES,
                apply_scaling=SCHEMA_APPLY_SCALING,
            ),
            required=[],
            additional_properties=False,
        )

    @staticmethod
    def _is_mldataset_available(item: pystac.Item) -> bool:
        return False
