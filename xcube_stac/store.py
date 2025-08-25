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
)
from xcube.util.jsonschema import JsonObjectSchema, JsonStringSchema

from .accessors import list_ardc_data_ids, guess_ardc_accessor, guess_item_accessor
from .accessors.base import XcubeStacItemAccessor
from .constants import (
    CDSE_S3_ENDPOINT,
    CDSE_STAC_URL,
    DATA_OPENER_IDS,
    DATA_STORE_ID_CDSE,
    DATA_STORE_ID_CDSE_ARDC,
    DATA_STORE_ID_PC_ARDC,
    DATA_STORE_ID_PC,
    DATA_STORE_ID,
    DATA_STORE_ID_XCUBE,
    PC_STAC_URL,
    LOG,
    MAP_FILE_EXTENSION_FORMAT,
    PROTOCOLS,
    SCHEMA_ADDITIONAL_QUERY,
    SCHEMA_BBOX,
    SCHEMA_COLLECTIONS,
    SCHEMA_S3_STORE,
    SCHEMA_TIME_RANGE,
    SCHEMA_URL,
)
from .utils import (
    access_collection,
    access_item,
    convert_datetime2str,
    get_attrs_from_pystac_object,
    get_data_id_from_pystac_object,
    is_mldataset_available,
    is_valid_ml_data_type,
    list_format_ids,
    list_protocols,
    modify_catalog_url,
    reproject_bbox,
    search_collections,
    search_items,
    update_dict,
)
from .version import version


class StacDataStore(DataStore):
    """STAC implementation of the data store.

    Args:
        url: URL to STAC catalog
        storage_options_s3: storage option for 's3' data store
    """

    def __init__(self, url: str, **storage_options_s3):
        self._url = modify_catalog_url(url)
        self._storage_options_s3 = storage_options_s3
        self._store_id = DATA_STORE_ID

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
        self._assert_valid_data_type(data_type)
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
        if self._is_valid_data_type(data_type):
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
        self._assert_valid_data_type(data_type)

        if data_id is not None:
            url = f"{self._url}/{data_id}"
            item = access_item(url, self._catalog)
            protocols = list_protocols(item)
            format_ids = list_format_ids(item)
        else:
            protocols = PROTOCOLS
            format_ids = list(np.unique(list(MAP_FILE_EXTENSION_FORMAT.values())))

        return self._filter_opener_ids(protocols, format_ids, data_type=data_type)

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        self._assert_valid_opener_id(opener_id)
        accessor = guess_item_accessor(self._store_id, data_id)(
            self._catalog, **self._storage_options_s3
        )
        if data_id is not None and opener_id is None:
            opener_ids = self.get_data_opener_ids(data_id=data_id)
            if not opener_ids:
                opener_id = None
            else:
                opener_id = opener_ids[0]
        return accessor.get_open_data_params_schema(
            data_id=data_id, opener_id=opener_id
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        # check input parameter
        self._assert_valid_data_type(data_type)
        self._assert_valid_opener_id(opener_id)
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)

        # access item and open with accessor
        url = f"{self._url}/{data_id}"
        item = access_item(url, self._catalog)
        accessor = guess_item_accessor(self._store_id, data_id)(
            self._catalog, **self._storage_options_s3
        )
        return accessor.open_item(
            item,
            opener_id=opener_id,
            data_type=data_type,
            **open_params,
        )

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DatasetDescriptor | MultiLevelDatasetDescriptor:
        self._assert_valid_data_type(data_type)

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
        self._assert_valid_data_type(data_type)
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
    def _filter_opener_ids(
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

    def _is_valid_data_type(self, data_type: DataTypeLike) -> bool:
        """Auxiliary function to check if data type is supported
        by the store.

        Args:
            data_type: Data type that is to be checked.

        Returns:
            True if *data_type* is supported by the store, otherwise False
        """
        return data_type is None or any(
            DataType.normalize(data_type_str).is_super_type_of(data_type)
            for data_type_str in self.get_data_types()
        )

    def _assert_valid_data_type(self, data_type: DataTypeLike) -> None:
        """Auxiliary function to assert if data type is supported
        by the store.

        Args:
            data_type: Data type that is to be checked.

        Raises:
            DataStoreError: Error, if *data_type* is not
                supported by the store.
        """
        if not self._is_valid_data_type(data_type):
            raise DataStoreError(
                f"Data type must be one of {self.get_data_types()!r}, "
                f"but got {data_type!r}."
            )

    def _assert_valid_opener_id(self, opener_id: str) -> None:
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
                f"{self.get_data_opener_ids()}, but got {opener_id!r}."
            )


class StacXcubeDataStore(StacDataStore):
    """STAC implementation of the data store for xcube STAC API."""

    def __init__(self, url: str, **storage_options_s3):
        super().__init__(url, **storage_options_s3)
        self._store_id = DATA_STORE_ID_XCUBE

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
        self._assert_valid_data_type(data_type)
        protocols = ["s3"]
        format_ids = ["zarr", "levels"]

        return self._filter_opener_ids(protocols, format_ids, data_type=data_type)

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        # check input parameter
        self._assert_valid_data_type(data_type)
        self._assert_valid_opener_id(opener_id)
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)

        # access item and open with accessor
        url = f"{self._url}/{data_id}"
        item = access_item(url, self._catalog)

        # decide between levels and zarr
        asset_names = open_params.pop("asset_names", None)
        if opener_id is not None:
            opener_id_data_type = opener_id.split(":")[0]
        else:
            opener_id_data_type = None
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
        accessor = XcubeStacItemAccessor(self._catalog, **self._storage_options_s3)
        ds = accessor.open_item(
            item,
            opener_id=opener_id,
            data_type=data_type,
            asset_names=asset_names,
            **open_params,
        )
        return ds

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
        super().__init__(url=CDSE_STAC_URL, **storage_options_s3)
        self._store_id = DATA_STORE_ID_CDSE

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            description="Describes the parameters of the xcube data store 'stac-cdse'.",
            properties=dict(
                key=JsonStringSchema(
                    title="AWS S3 key to access CDSE EO data.",
                    description=(
                        "To get key and secret, follow https://documentation."
                        "dataspace.copernicus.eu/APIs/S3.html#generate-secrets"
                    ),
                ),
                secret=JsonStringSchema(
                    title="AWS S3 secret to access CDSE EO data.",
                    description=(
                        "To get key and secret, follow https://documentation."
                        "dataspace.copernicus.eu/APIs/S3.html#generate-secrets"
                    ),
                ),
            ),
            required=["key", "secret"],
            additional_properties=False,
        )

    @classmethod
    def get_data_types(cls) -> tuple[str, ...]:
        return (DATASET_TYPE.alias,)

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> tuple[str, ...]:
        LOG.info(
            f"In the {self._store_id!r} data store, data openers are specific to each "
            "Sentinel mission which are selected via the *data_id*. A generic "
            "opener ID is returned."
        )
        # We return a generic opener ID here. In this data store, specific data
        # accessors are delegated based on the data ID to handle different
        # Sentinel missions transparently.
        return ("dataset:format:stac-cdse",)

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        include_attrs: Container[str] | bool = False,
    ) -> Iterator[str | tuple[str, dict[str, Any]], None]:
        raise NotImplementedError(
            "Listing all data IDs is not supported in the 'stac-cdse' data store, "
            "because the underlying CDSE STAC API contains too many items. "
            "Use the `search_data()` method to find datasets matching your criteria."
        )

    @staticmethod
    def _is_mldataset_available(item: pystac.Item) -> bool:
        return False


class StacPlanetaryComputerDataStore(StacDataStore):
    """STAC implementation of the data store for CDSE STAC API."""

    def __init__(self):
        super().__init__(url=PC_STAC_URL)
        self._store_id = DATA_STORE_ID_PC

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            description="Describes the parameters of the xcube data store 'stac-pc'.",
            required=[],
            additional_properties=False,
        )

    @classmethod
    def get_data_types(cls) -> tuple[str, ...]:
        return (DATASET_TYPE.alias,)

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> tuple[str, ...]:
        LOG.info(
            f"In the {self._store_id!r} data store, data openers are specific to each "
            "Sentinel mission which are selected via the *data_id*. A generic "
            "opener ID is returned."
        )
        # We return a generic opener ID here. In this data store, specific data
        # accessors are delegated based on the data ID to handle different
        # Sentinel missions transparently.
        return ("dataset:format:stac-pc",)

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        include_attrs: Container[str] | bool = False,
    ) -> Iterator[str | tuple[str, dict[str, Any]], None]:
        raise NotImplementedError(
            "Listing all data IDs is not supported in the 'stac-pc' data store, "
            "because the underlying CDSE STAC API contains too many items. "
            "Use the `search_data()` method to find datasets matching your criteria."
        )

    @staticmethod
    def _is_mldataset_available(item: pystac.Item) -> bool:
        return False


class ArdcStacCdseDataStore(StacCdseDataStore):
    """Data store to generate analysis-ready data cubes from multiple STAC items
    using the CDSE STAC API.
    """

    def __init__(self, **storage_options_s3):
        super().__init__(**storage_options_s3)
        self._store_id = DATA_STORE_ID_CDSE_ARDC

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        include_attrs: Container[str] | bool = False,
    ) -> Iterator[str | tuple[str, dict[str, Any]], None]:
        self._assert_valid_data_type(data_type)
        for collection_id in list_ardc_data_ids(self._store_id):
            if not include_attrs:
                yield collection_id
            else:
                url = f"{self._url}/collections/{collection_id}"
                collection = access_collection(url, self._catalog)
                attrs = get_attrs_from_pystac_object(collection, include_attrs)
                yield collection.id, attrs

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if self._is_valid_data_type(data_type):
            return data_id in list_ardc_data_ids(self._store_id)
        return False

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        self._assert_valid_opener_id(opener_id)
        if data_id is None:
            raise DataStoreError(
                "Please assign the *data_id* in the 'stac-cdse-ardc' data store to "
                "retrieve the open_params for a specific data collection."
            )
        accessor = guess_ardc_accessor(self._store_id, data_id)(
            self._catalog, **self._storage_options_s3
        )
        return accessor.get_open_data_params_schema(
            data_id=data_id, opener_id=opener_id
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        # check input parameter
        self._assert_valid_data_type(data_type)
        self._assert_valid_opener_id(opener_id)
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)

        # search for items
        bbox_wgs84 = reproject_bbox(
            open_params["bbox"], open_params["crs"], "EPSG:4326"
        )
        items = list(
            search_items(
                self._catalog,
                self._searchable,
                collections=[data_id],
                bbox=bbox_wgs84,
                time_range=open_params["time_range"],
                query=open_params.get("query"),
            )
        )

        if len(items) == 0:
            LOG.warn(
                f"No items found in collection {data_id!r} for the "
                f"parameters bbox {bbox_wgs84!r}, time_range "
                f"{open_params['time_range']!r} and "
                f"query {open_params.get('query', 'None')!r}."
            )
            return None

        accessor = guess_ardc_accessor(self._store_id, data_id)(
            self._catalog, **self._storage_options_s3
        )
        ds = accessor.open_ardc(
            items,
            opener_id=opener_id,
            data_type=data_type,
            **open_params,
        )
        ds.attrs["stac_catalog_url"] = self._catalog.get_self_href()
        ds.attrs["xcube_stac_version"] = version
        return ds

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DatasetDescriptor | MultiLevelDatasetDescriptor:
        self._assert_valid_data_type(data_type)

        # get extent from collection
        url = f"{self._url}/collections/{data_id}"
        collection = access_collection(url, self._catalog)
        bbox = collection.extent.spatial.bboxes[0]
        temp_extent = collection.extent.temporal.intervals[0]
        time_start = None
        if temp_extent[0] is not None:
            time_start = convert_datetime2str(temp_extent[0])
        time_end = None
        if temp_extent[1] is not None:
            time_end = convert_datetime2str(temp_extent[1])

        return DatasetDescriptor(data_id, bbox=bbox, time_range=(time_start, time_end))

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DatasetDescriptor | MultiLevelDatasetDescriptor]:
        self._assert_valid_data_type(data_type)
        schema = self.get_search_params_schema()
        schema.validate_instance(search_params)
        for collection in search_collections(self._catalog, **search_params):
            yield self.describe_data(collection.id, data_type=data_type)

    def get_search_params_schema(
        self, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                time_range=SCHEMA_TIME_RANGE,
                bbox=SCHEMA_BBOX,
            ),
            required=[],
            additional_properties=False,
        )


class ArdcStacPlanetaryComputerDataStore(StacPlanetaryComputerDataStore):
    """Data store to generate analysis-ready data cubes from multiple STAC items
    using the CDSE STAC API.
    """

    def __init__(self):
        super().__init__()
        self._store_id = DATA_STORE_ID_PC_ARDC

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        include_attrs: Container[str] | bool = False,
    ) -> Iterator[str | tuple[str, dict[str, Any]], None]:
        self._assert_valid_data_type(data_type)
        for collection_id in list_ardc_data_ids(self._store_id):
            if not include_attrs:
                yield collection_id
            else:
                url = f"{self._url}/collections/{collection_id}"
                collection = access_collection(url, self._catalog)
                attrs = get_attrs_from_pystac_object(collection, include_attrs)
                yield collection.id, attrs

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if self._is_valid_data_type(data_type):
            return data_id in list_ardc_data_ids(self._store_id)
        return False

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        self._assert_valid_opener_id(opener_id)
        if data_id is None:
            raise DataStoreError(
                "Please assign the *data_id* in the 'stac-cdse-ardc' data store to "
                "retrieve the open_params for a specific data collection."
            )
        accessor = guess_ardc_accessor(self._store_id, data_id)(
            self._catalog, **self._storage_options_s3
        )
        return accessor.get_open_data_params_schema(
            data_id=data_id, opener_id=opener_id
        )

    def open_data(
        self,
        data_id: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        # check input parameter
        self._assert_valid_data_type(data_type)
        self._assert_valid_opener_id(opener_id)
        schema = self.get_open_data_params_schema(data_id=data_id, opener_id=opener_id)
        schema.validate_instance(open_params)

        # search for items
        bbox_wgs84 = reproject_bbox(
            open_params["bbox"], open_params["crs"], "EPSG:4326"
        )
        items = list(
            search_items(
                self._catalog,
                self._searchable,
                collections=[data_id],
                bbox=bbox_wgs84,
                time_range=open_params["time_range"],
                query=open_params.get("query"),
            )
        )

        if len(items) == 0:
            LOG.warn(
                f"No items found in collection {data_id!r} for the "
                f"parameters bbox {bbox_wgs84!r}, time_range "
                f"{open_params['time_range']!r} and "
                f"query {open_params.get('query', 'None')!r}."
            )
            return None

        accessor = guess_ardc_accessor(self._store_id, data_id)(
            self._catalog, **self._storage_options_s3
        )
        ds = accessor.open_ardc(
            items,
            opener_id=opener_id,
            data_type=data_type,
            **open_params,
        )
        ds.attrs["stac_catalog_url"] = self._catalog.get_self_href()
        ds.attrs["xcube_stac_version"] = version
        return ds

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DatasetDescriptor | MultiLevelDatasetDescriptor:
        self._assert_valid_data_type(data_type)

        # get extent from collection
        url = f"{self._url}/collections/{data_id}"
        collection = access_collection(url, self._catalog)
        bbox = collection.extent.spatial.bboxes[0]
        temp_extent = collection.extent.temporal.intervals[0]
        time_start = None
        if temp_extent[0] is not None:
            time_start = convert_datetime2str(temp_extent[0])
        time_end = None
        if temp_extent[1] is not None:
            time_end = convert_datetime2str(temp_extent[1])

        return DatasetDescriptor(data_id, bbox=bbox, time_range=(time_start, time_end))

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DatasetDescriptor | MultiLevelDatasetDescriptor]:
        self._assert_valid_data_type(data_type)
        schema = self.get_search_params_schema()
        schema.validate_instance(search_params)
        for collection in search_collections(self._catalog, **search_params):
            yield self.describe_data(collection.id, data_type=data_type)

    def get_search_params_schema(
        self, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                time_range=SCHEMA_TIME_RANGE,
                bbox=SCHEMA_BBOX,
            ),
            required=[],
            additional_properties=False,
        )
