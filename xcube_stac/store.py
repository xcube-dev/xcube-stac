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

from typing import Any, Tuple, Iterable, Iterator, Dict, Container, Union, List

from pystac import ItemCollection, Item
import xarray as xr
from xcube.core.store import (
    DATASET_TYPE,
    DataDescriptor,
    DataStore,
    DataStoreError,
    DataTypeLike
)
from xcube.util.jsonschema import (
    JsonObjectSchema,
    JsonStringSchema
)

from .constants import DATASET_OPENER_ID
from .opener import StacDataOpener
from .stac import Stac


class StacDataStore(StacDataOpener, DataStore):
    """STAC implementation of the data store.

    Attributes:
        url: URL to STAC catalog
        data_id_delimiter: Delimiter used to separate
            collections, items and assets from each other.
            Defaults to "/".
    """

    def __init__(
        self,
        url: str,
        data_id_delimiter: str = "/"
    ):
        super().__init__(stac=Stac(
            url,
            data_id_delimiter=data_id_delimiter
        ))

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        stac_params = dict(
            url=JsonStringSchema(
                title="URL to STAC catalog"
            ),
            data_id_delimiter=JsonStringSchema(
                title="Data ID delimiter",
                description=(
                    "Delimiter used to separate collections, "
                    "items and assets from each other"
                ),
            )
        )
        return JsonObjectSchema(
            description=(
                "Describes the parameters of the xcube data store 'stac'."
            ),
            properties=stac_params,
            required=["url"],
            additional_properties=False
        )

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return (DATASET_TYPE.alias,)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        return self.get_data_types()

    def get_item_collection(
        self, **open_params
    ) -> Tuple[ItemCollection, List[str]]:
        """Collects all items within the given STAC catalog
        using the supplied *open_params*.

        Returns:
            A tuple of the item collection containing all items
            identified by *open_params* and data IDs corresponding to items
        """
        return self.stac.get_item_collection(**open_params)

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        items: Iterable[Item] = None,
        item_data_ids: Iterable[str] = None,
        include_attrs: Container[str] = None,
        **open_params
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        """Get an iterator over the data resource identifiers for the
        given type *data_type*. If *data_type* is omitted, all data
        resource identifiers are returned. The data resource identifiers
        follow the following structure:

            `collection_id_0/../collection_id_n/item_id/asset`

        Args:
            data_type: If given, only data identifiers
                that are available as this type are returned. If this is None,
                all available data identifiers are returned. Defaults to None.
            items: collection of items for which data IDs are desired. If None,
                items are collected by :meth:`Stac.get_item_collection`
                using *open_params
            item_data_ids: data IDs corresponding to items. If None,
                item_data_ids are collected by :meth:`Stac.get_item_data_ids`
            include_attrs: A sequence of names of attributes to be returned
                for each dataset identifier. If given, the store will attempt
                to provide the set of requested dataset attributes in addition
                to the data ids. If no attributes are found, empty dictionary
                is returned. So far only the attribute 'title' is supported.
                Defaults to None.

        Returns:
            An iterator over the identifiers (and additional attributes defined
            by *include_attrs* of data resources provided by this data store.
        """
        self._assert_valid_data_type(data_type)
        return self.stac.get_data_ids(
            items=items,
            item_data_ids=item_data_ids,
            include_attrs=include_attrs,
            **open_params
        )

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if self._is_valid_data_type(data_type):
            return data_id in self.list_data_ids()
        return False

    def describe_data(self, data_id: str, **open_params) -> DataDescriptor:
        """Get the descriptor for the data resource given by *data_id*.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Returns:
            Data descriptor containing meta data of
            the data resources identified by *data_id*
        """
        return super().describe_data(data_id, **open_params)

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        self._assert_valid_data_type(data_type)
        if data_id is not None and not self.has_data(data_id, data_type=data_type):
            raise DataStoreError(
                f"Data resource {data_id!r} is not available."
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
        """Open the data given by the data resource identifier *data_id*
        using the data opener identified by *opener_id* and
        the supplied *open_params*.

        Args:
            data_id: An identifier of data that is provided by this
                store.
            opener_id: Data opener identifier. Defaults to None.

        Returns:
            A representation of the data resources identified
            by *data_id* and *open_params*.
        """
        self._assert_valid_opener_id(opener_id)
        return super().open_data(data_id, **open_params)

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DataDescriptor]:
        """Search this store for data resources. If *data_type* is given,
        the search is restricted to data resources of that type.

        Args:
            data_type: Data type that is known to be
                supported by this data store. Defaults to None.

        Raises:
            NotImplementedError: Not implemented yet.

        Yields:
            An iterator of data descriptors for the found data resources.
        """
        # ToDo: implement search_data method.
        raise NotImplementedError("search_data() operation is not supported yet")

    @classmethod
    def get_search_params_schema(
        cls, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        """Get the schema for the parameters that can be passed
        as *search_params* to :meth:`search_data`. Parameters are
        named and described by the properties of the returned JSON object schema.

        Args:
            data_type: Data type that is known to be
                supported by this data store. Defaults to None.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            A JSON object schema whose properties describe this
            store's search parameters.
        """
        # ToDo: implement get_search_params_schema in
        #       combination with search_data method.
        raise NotImplementedError(
            "get_search_params_schema() operation is not supported yet"
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
        return data_type is None or DATASET_TYPE.is_super_type_of(data_type)

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
                f"Data type must be {DATASET_TYPE!r}, "
                f"but got {data_type!r}"
            )

    @classmethod
    def _assert_valid_opener_id(cls, opener_id: str):
        """Auxiliary function to assert if data opener identified by
        *opener_id* is supported by the store.

        Args:
            opener_id (_type_): Data opener identifier

        Raises:
            DataStoreError: Error, if *opener_id* is not
                supported by the store.
        """
        if opener_id is not None and opener_id != DATASET_OPENER_ID:
            raise DataStoreError(
                f"Data opener identifier must be "
                f'{DATASET_OPENER_ID!r}, but got {opener_id!r}'
            )
