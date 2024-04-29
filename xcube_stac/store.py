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

from typing import Any, Tuple, Iterator, Dict, Container, Union

import logging
import xarray as xr

from xcube.util.jsonschema import (
    JsonObjectSchema,
    JsonStringSchema
)
from xcube.core.store import (
    DATASET_TYPE,
    DataDescriptor,
    DataOpener,
    DataStore,
    DataStoreError,
    DataTypeLike,
    DatasetDescriptor
)
from .constants import DATASET_OPENER_ID
from .stac import Stac

_LOG = logging.getLogger("xcube")


class StacDataOpener(DataOpener):
    """ STAC implementation of the data opener.
    """

    def __init__(self, stac: Stac):
        """
        Args:
            stac (Stac): class containing methods handling STAC catalogs
        """
        self.stac = stac

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """ Get the schema for the parameters passed as *open_params* to
        :meth:`open_data`.

        Args:
            data_id (str, optional): An identifier of data that is provided by this
                store. Defaults to None.

        Returns:
            JsonObjectSchema: containing the parameters used by the data opener
                to open data.
        """
        # TODO: to be adjusted
        open_parms = {}
        stac_schema = JsonObjectSchema(
            properties=dict(**open_parms),
            required=[],
            additional_properties=False
        )
        return stac_schema

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        """ Open the data given by the data resource identifier *data_id*
        using the supplied *open_params*.

        Args:
            data_id (str): An identifier of data that is provided by this
                store.

        Returns:
            xr.Dataset: An in-memory representation of the data resources
                identified by *data_id* and *open_params*.
        """
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        return self.stac.open_data(data_id, **open_params)

    def describe_data(
        self, data_id: str, **open_params
    ) -> DatasetDescriptor:
        """ Get the descriptor for the data resource given by *data_id*.

        Args:
            data_id (str): An identifier of data that is provided by this
                store.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            DatasetDescriptor: data descriptor containing meta data of
                the data resources identified by *data_id*
        """
        # TODO: implement describe_data method.
        raise NotImplementedError("describe_data() operation is not supported yet")


class StacDataStore(StacDataOpener, DataStore):
    """ STAC implementation of the data store.
    """

    def __init__(self, **stac_kwargs):
        """
        Args:
            **stac_kwargs: Parameters required by the STAC data store.
                * url (str): URL to STAC catalog (required)
                * collection_prefix (str): Path of collection used as
                    entry point (optional)
                * data_id_delimiter (str): Delimiter used to separate
                    collections, items and assets from each other (optional)
        """
        super().__init__(stac=Stac(**stac_kwargs))

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        """ Get the JSON schema for instantiating a new data store.

        Returns:
            JsonObjectSchema: The JSON schema for the data store's parameters.
        """
        stac_params = dict(
            url=JsonStringSchema(
                title="URL to STAC catalog"
            ),
            collection_prefix=JsonStringSchema(
                title="Collection prefix",
                description="Path of collection used as entry point",
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
            properties=stac_params,
            required=["url"],
            additional_properties=False
        )

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        """ Get alias names for all data types supported by this store.

        Returns:
            Tuple[str, ...]: The tuple of supported data types.
        """
        return (DATASET_TYPE.alias,)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        """ Get alias names for of data types that are supported
        by this store for the given *data_id*.

        Args:
            data_id (str): An identifier of data that is provided by this
                store.

        Returns:
            Tuple[str, ...]: A tuple of data types that apply
                to the given *data_id*.
        """
        return self.get_data_types()

    def get_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        """ Get an iterator over the data resource identifiers for the
        given type *data_type*. If *data_type* is omitted, all data
        resource identifiers are returned. The data resource identifiers
        follow the following structure:

            `collection_id_0/../collection_id_n/item_id/asset`

        Args:
            data_type (DataTypeLike, optional): If given, only data identifiers
                that are available as this type are returned. If this is None,
                all available data identifiers are returned. Defaults to None.
            include_attrs (Container[str], optional): A sequence of names
                of attributes to be returned for each dataset identifier.
                If given, the store will attempt to provide the set of
                requested dataset attributes in addition to the data ids.
                If no attributes are found, empty dictionary is retured.
                So far only the attribute 'title' is supported.
                Defaults to None.
        Returns:
            Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]: An iterator
                over the identifiers (and additional attributes defined by
                *include_attrs* of data resources provided by this data store.
        """
        self._assert_valid_data_type(data_type)
        return self.stac.build_data_ids(self.stac.catalog, include_attrs=include_attrs)

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        """ Check if the data resource given by *data_id* is available
        in this store.

        Args:
            data_id (str): An identifier of data that is provided by this
                store.
            data_type (DataTypeLike, optional): An optional data type. If given,
                it will also bE checked whether the data is available as the
                specifieD type. May be given as type alias name, as a type, or as
                a :class:`DataType` instance. Defaults to None.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            bool: True, if the data resource is available in this store,
                False otherwise.
        """
        if self._is_valid_data_type(data_type):
            return data_id in self.list_data_ids()
        return False

    def describe_data(self, data_id: str, **open_params) -> DataDescriptor:
        """ Get the descriptor for the data resource given by *data_id*.

        Args:
            data_id (str): An identifier of data that is provided by this
                store.

        Returns:
            DataDescriptor: data descriptor containing meta data of
                the data resources identified by *data_id*
        """
        return super().describe_data(data_id, **open_params)

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        """ Get identifiers of data openers that can be used to open data
        resources from this store.

        Args:
            data_id (str, optional): An identifier of data that is provided by this
                store. Defaults to None.
            data_type (DataTypeLike, optional): Data type that is known to be
                supported by this data store. May be given as type alias name,
                as a type, or as a :class:`DataType` instance. Defaults to None.

        Raises:
            DataStoreError: If an error occurs.

        Returns:
            Tuple[str, ...]: A tuple of identifiers of data openers that
            can be used to open data resources.
        """
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
        """ Get the schema for the parameters passed as *open_params* to
        :meth:`open_data`.

        Args:
            data_id (str, optional): An identifier of data that is provided by this
                store. Defaults to None.
            opener_id (str, optional): Data opener identifier. Defaults to None.

        Returns:
            JsonObjectSchema: The schema for the parameters in *open_params*.
        """
        self._assert_valid_opener_id(opener_id)
        return super().get_open_data_params_schema(data_id)

    def open_data(
        self, data_id: str, opener_id: str = None, **open_params
    ) -> xr.Dataset:
        """ Open the data given by the data resource identifier *data_id*
        using the data opener identified by *opener_id* and
        the supplied *open_params*.

        Args:
            data_id (str): An identifier of data that is provided by this
                store.
            opener_id (str, optional): Data opener identifier. Defaults to None.

        Returns:
            xr.Dataset: An in-memory representation of the data resources identified
                by *data_id* and *open_params*.
        """
        self._assert_valid_opener_id(opener_id)
        return super().open_data(data_id, **open_params)

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DataDescriptor]:
        """ Search this store for data resources. If *data_type* is given,
        the search is restricted to data resources of that type.

        Args:
            data_type (DataTypeLike, optional): Data type that is known to be
                supported by this data store. Defaults to None.

        Raises:
            NotImplementedError: Not implemented yet.

        Yields:
            Iterator[DataDescriptor]: An iterator of data descriptors
                for the found data resources.
        """
        # TODO: implement search_data method.
        raise NotImplementedError("search_data() operation is not supported yet")

    @classmethod
    def get_search_params_schema(
        cls, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        """ Get the schema for the parameters that can be passed
        as *search_params* to :meth:`search_data`. Parameters are
        named and described by the properties of the returned JSON object schema.

        Args:
            data_type (DataTypeLike, optional): Data type that is known to be
                supported by this data store. Defaults to None.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            JsonObjectSchema: A JSON object schema whose properties
                describe this store's search parameters.
        """
        # TODO: implement get_search_params_schema in
        #       combination with search_data method.
        raise NotImplementedError(
            "get_search_params_schema() operation is not supported yet"
        )

    ##########################################################################
    # Implementation helpers

    @classmethod
    def _is_valid_data_type(cls, data_type: DataTypeLike) -> bool:
        """ Auxiliary function to check if data type is supported
        by the store.

        Args:
            data_type (DataTypeLike): Data type that is to be checked.

        Returns:
            bool: True if *data_type* is supported by the store, otherwise False
        """
        return data_type is None or DATASET_TYPE.is_super_type_of(data_type)

    @classmethod
    def _assert_valid_data_type(cls, data_type: DataTypeLike):
        """ Auxiliary function to assert if data type is supported
        by the store.

        Args:
            data_type (DataTypeLike): Data type that is to be checked.

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
    def _assert_valid_opener_id(cls, opener_id):
        """ Auxiliary function to assert if data opener identified by
        *opener_id* is supported by the store.

        Args:
            opener_id (_type_): Data opener identifier

        Raises:
            DataStoreError: Error, if *opener_id* is not
                supported by the store.
        """
        if opener_id is not None and opener_id != DATASET_OPENER_ID:
            raise DataStoreError(
                f"Data opener identifier must be"
                f' "{DATASET_OPENER_ID}",'
                f' but got "{opener_id}"'
            )
