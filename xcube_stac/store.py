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

from typing import Tuple

import logging
import xarray as xr
import pystac_client

from xcube.core.store import (
    DATASET_TYPE,
    DataDescriptor,
    DataOpener,
    DataStore,
    DataStoreError,
    DataTypeLike,
    DatasetDescriptor
)
from xcube.util.assertions import assert_not_none
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonDateSchema,
    JsonNumberSchema,
    JsonObjectSchema,
    JsonStringSchema,
)

from .constants import DATASET_OPENER_ID
from .stac import Stac

_LOG = logging.getLogger("xcube")


class StacDataOpener(DataOpener):
    """ STAC implementation of the data opener.

    Args:
        DataOpener (xcube.core.store.DataOpener): data opener defined
            in the xcube data store framework.

    Returns:
        StacDataOpener: data opener defined in the ``xcube_stac``  plugin.
    """

    def __init__(self, stac: Stac):
        self.stac = stac

    def get_open_data_params_schema(
        self, data_id: str = None
    ) -> JsonObjectSchema:
        dataset_params = dict(
            variable_names=JsonArraySchema(
                items=(JsonStringSchema(min_length=0)),
                unique_items=True
            ),
            time_range=JsonDateSchema.new_range(),
            bbox=JsonArraySchema(
                items=(
                    JsonNumberSchema(),
                    JsonNumberSchema(),
                    JsonNumberSchema(),
                    JsonNumberSchema(),
                )
            ),
        )
        stac_schema = JsonObjectSchema(
            properties=dict(**dataset_params),
            required=[],
            additional_properties=False
        )
        return stac_schema

    def open_data(
        self, data_id: str, **open_params
    ) -> xr.Dataset:
        assert_not_none(data_id, "data_id")
        stac_schema = self.get_open_data_params_schema(data_id)
        stac_schema.validate_instance(open_params)
        return self.stac.open_dataset(**open_params)

    def describe_data(self, data_id: str) -> DatasetDescriptor:
        assert_not_none(data_id, "data_id")
        stac_schema = self.get_open_data_params_schema(data_id)

        catalog = pystac_client.Client.open(self.stac._url)
        search = catalog.search()

        # ToDo: readout search and gather information
        dsd = DatasetDescriptor(
            data_id=data_id,
            url=search.url,
            bbox="",
            time_range="",
            time_period="",
            attrs="",
        )
        dsd.open_params_schema = stac_schema
        return dsd


class StacDataStore(StacDataOpener, DataStore):
    """ STAC implementation of the data store.

    Args:
        DataStore (xcube.core.store.DataStore): data store defined
            in the xcube data store framework.

    Returns:
        StacDataStore: data store defined in the ``xcube_stac`` plugin.
    """

    def __init__(self, **stac_kwargs):
        super().__init__(stac=Stac(**stac_kwargs))

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        # ToDo: to be added
        stac_params = {}
        return JsonObjectSchema(
            properties=stac_params, required=None, additional_properties=False
        )

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return (DATASET_TYPE.alias,)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        return self.get_data_types()

    # ToDo: to be added, maybe collection ids,
    # or catalog ids, probably not item ids
    def get_data_ids():
        return [f'demo{i}' for i in range(5)]

    def has_data(self, data_id: str, data_type: str = None) -> bool:
        if self._is_valid_data_type(data_type):
            return data_id in self.get_data_ids()
        return False

    def describe_data(self, data_id: str) -> DataDescriptor:
        # data_type_alias is ignored, xcube-sh only provides "dataset"
        return super().describe_data(data_id)

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        self._assert_valid_data_type(data_type)
        if data_id is not None and not self.has_data(data_id, data_type=data_type):
            raise DataStoreError(
                f"Data resource {data_id!r}" f" is not available."
            )
        if data_type is not None and not DATASET_TYPE.is_super_type_of(data_type):
            raise DataStoreError(
                f"Data resource {data_id!r}" f" is not "
                f"available as type {data_type!r}."
            )
        return (DATASET_OPENER_ID,)

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        self._assert_valid_opener_id(opener_id)
        return super().get_open_data_params_schema(data_id)

    def open_data(
        self, data_id: str, opener_id: str = None, **open_params
    ) -> xr.Dataset:
        self._assert_valid_opener_id(opener_id)
        return super().open_data(data_id, **open_params)

    ##########################################################################
    # Implementation helpers

    @classmethod
    def _is_valid_data_type(cls, data_type: DataTypeLike):
        return data_type is None or DATASET_TYPE.is_super_type_of(data_type)

    @classmethod
    def _assert_valid_data_type(cls, data_type):
        if not cls._is_valid_data_type(data_type):
            raise DataStoreError(
                f"Data type must be {DATASET_TYPE!r}," f" but got {data_type!r}"
            )

    @classmethod
    def _assert_valid_opener_id(cls, opener_id):
        if opener_id is not None and opener_id != DATASET_OPENER_ID:
            raise DataStoreError(
                f"Data opener identifier must be"
                f' "{DATASET_OPENER_ID}",'
                f' but got "{opener_id}"'
            )
