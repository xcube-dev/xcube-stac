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
from xcube.util.jsonschema import JsonObjectSchema

from .constants import (
    CDSE_STAC_URL,
    CDSE_S3_ENDPOINT,
    DATA_OPENER_IDS,
    STAC_STORE_PARAMETERS,
)
from .store_mode import SingleStoreMode
from .store_mode import StackStoreMode
from .stac_objects import GeneralStacItem
from .stac_objects import CdseStacItem
from .stac_objects import XcubeStacItem
from ._utils import (
    assert_valid_data_type,
    assert_valid_opener_id,
    get_attrs_from_pystac_object,
    get_data_id_from_pystac_object,
    is_valid_data_type,
    is_valid_ml_data_type,
    modify_catalog_url,
    update_dict,
)


class StacDataStore(DataStore):
    """STAC implementation of the data store.

    Args:
        url: URL to STAC catalog
        stack_mode: if True, items will be stacked along the time axis;
            defaults to False.
        storage_options_s3: storage option for 's3' data store
    """

    def __init__(
        self, url: str, stack_mode: Union[bool, str] = False, **storage_options_s3
    ):
        self._url = url
        self._url_mod = modify_catalog_url(url)
        self._stack_mode = stack_mode
        self._storage_options_s3 = storage_options_s3

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

        if not hasattr(self, "_stacitem"):
            self._stacitem = GeneralStacItem

        if stack_mode is False:
            self._impl = SingleStoreMode(
                self._catalog,
                self._url_mod,
                self._searchable,
                self._storage_options_s3,
                self._stacitem,
            )
        elif stack_mode is True or stack_mode == "odc-stac":
            self._impl = StackStoreMode(
                self._catalog,
                self._url_mod,
                self._searchable,
                self._storage_options_s3,
                self._stacitem,
            )
        else:
            raise DataStoreError(
                "Invalid parameterization detected: a boolean or"
                " 'odc-stac' was expected"
            )

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        stac_params = STAC_STORE_PARAMETERS
        return JsonObjectSchema(
            description="Describes the parameters of the xcube data store 'stac'.",
            properties=stac_params,
            required=["url"],
            additional_properties=True,
        )

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return DATASET_TYPE.alias, MULTI_LEVEL_DATASET_TYPE.alias

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        item = self._impl.access_item(data_id)
        xitem = self._stacitem.from_pystac_item(item, self._storage_options_s3)
        if xitem.is_mldataset_available():
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
                xitem = self._stacitem.from_pystac_item(item, self._storage_options_s3)
                return xitem.is_mldataset_available()
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
            xitem = self._stacitem.from_pystac_item(item, self._storage_options_s3)
            protocols = xitem.protocols
            format_ids = xitem.format_ids
        else:
            protocols = self._stacitem.supported_protocols()
            format_ids = self._stacitem.supported_format_ids()

        return self._select_opener_id(
            protocols, format_ids, data_id=data_id, data_type=data_type
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

    def _select_opener_id(
        self,
        protocols: list[str],
        format_ids: list[str],
        data_id: str = None,
        data_type: DataTypeLike = None,
    ):
        if data_type is None and data_id is None:
            return DATA_OPENER_IDS
        elif data_type is None and data_id is not None:
            return tuple(
                opener_id
                for opener_id in DATA_OPENER_IDS
                if opener_id.split(":")[1] in format_ids
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
                and opener_id.split(":")[1] in format_ids
                and opener_id.split(":")[2] in protocols
            )


class StacCdseDataStore(StacDataStore):
    """STAC implementation of the data store for CDSE STAC API.

    Args:
        stack_mode: if True, items will be stacked along the time axis;
            defaults to False.
        storage_options_s3: storage option of the S3 data store; the key and secret
            are required for data access. see Note.

    Note:
        Credentials for the authentication can be obtained following the
        documentation https://documentation.dataspace.copernicus.eu/APIs/S3.html.
    """

    def __init__(
        self,
        stack_mode: Union[bool, str] = False,
        **storage_options_s3,
    ):
        storage_options_s3 = update_dict(
            storage_options_s3,
            dict(
                anon=False,
                client_kwargs=dict(endpoint_url=CDSE_S3_ENDPOINT),
            ),
        )
        self._stacitem = CdseStacItem
        super().__init__(url=CDSE_STAC_URL, stack_mode=stack_mode, **storage_options_s3)

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        stac_params = STAC_STORE_PARAMETERS.copy()
        del stac_params["url"]
        return JsonObjectSchema(
            description="Describes the parameters of the xcube data store 'stac-csde'.",
            properties=stac_params,
            required=["key", "secret"],
            additional_properties=True,
        )

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        assert_valid_opener_id(opener_id)
        return self._impl.get_open_data_params_schema(
            data_id=data_id, opener_id=opener_id
        )

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[Union[DatasetDescriptor, MultiLevelDatasetDescriptor]]:
        assert_valid_data_type(data_type)
        proc_level = search_params.pop("processing_level", "L2A")
        proc_baseline = search_params.pop("processing_baseline", 5.00)
        pystac_objs = self._impl.search_data(**search_params)

        for pystac_obj in pystac_objs:
            if not self._stack_mode:
                href = pystac_obj.assets["PRODUCT"].extra_fields["alternate"]["s3"][
                    "href"
                ][1:]
                if not href.startswith(f"eodata/Sentinel-2/MSI/{proc_level}"):
                    continue
                if not f"N0{int(proc_baseline*100)}" in href:
                    continue

            data_id = get_data_id_from_pystac_object(
                pystac_obj, catalog_url=self._url_mod
            )
            yield self.describe_data(data_id, data_type=data_type)


class StacXcubeDataStore(StacDataStore):
    """STAC implementation of the data store for xcube STAC API.

    Args:
        stack_mode: if True, items will be stacked along the time axis;
            defaults to False.
        storage_options_s3: storage option of the S3 data store;
    """

    def __init__(
        self,
        url: str,
        stack_mode: Union[bool, str] = False,
        **storage_options_s3,
    ):
        self._stacitem = XcubeStacItem
        super().__init__(url=url, stack_mode=stack_mode, **storage_options_s3)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        return MULTI_LEVEL_DATASET_TYPE.alias, DATASET_TYPE.alias

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        assert_valid_data_type(data_type)
        protocols = self._stacitem.supported_protocols()
        format_ids = self._stacitem.supported_format_ids()
        return self._select_opener_id(
            protocols, format_ids, data_id=data_id, data_type=data_type
        )
