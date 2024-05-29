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

from datetime import timezone
from itertools import chain
from typing import Any, Container, Dict, Iterable, Iterator, List, Tuple, Union
import warnings

import pandas as pd
import pystac
import pystac_client
from shapely.geometry import box
import xarray as xr
from xcube.core.store import (
    DATASET_TYPE,
    DataDescriptor,
    DatasetDescriptor,
    DataStore,
    DataStoreError,
    DataTypeLike
)
from xcube.util.jsonschema import (
    JsonObjectSchema,
    JsonStringSchema
)

from .constants import (
    DATASET_OPENER_ID,
    MIME_TYPES,
    STAC_SEARCH_ITEM_PARAMETERS
)


class StacDataStore(DataStore):
    """STAC implementation of the data store.

    Args:
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
        self._url = url
        self._data_id_delimiter = data_id_delimiter

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

        # TODO: Add a data store "file" here when implementing
        # open_data(), which will be used to open the hrefs

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
    ) -> Tuple[pystac.ItemCollection, List[str]]:
        """Collects all items within the given STAC catalog
        using the supplied *open_params*.

        Returns:
            items: item collection containing all items identified by *open_params*
            item_data_ids: data IDs corresponding to items
        """
        if self._searchable:
            # not used
            open_params.pop("variable_names", None)
            # rewrite to "datetime"
            time_range = open_params.pop("time_range", None)
            if time_range:
                open_params["datetime"] = "/".join(time_range)
            items = self._catalog.search(**open_params).item_collection()
        else:
            items = self._get_items_nonsearchable_catalog(
                self._catalog,
                **open_params
            )
            items = pystac.ItemCollection(items)
        item_data_ids = self.list_item_data_ids(items)
        return items, item_data_ids

    def get_item_data_id(self, item: pystac.Item) -> str:
        """Generates the data ID of an item, which follows the structure:

            `collection_id_0/../collection_id_n/item_id`

        Args:
            item: item/feature

        Returns:
            data ID of an item
        """
        id_parts = []
        parent_item = item
        while parent_item.STAC_OBJECT_TYPE != "Catalog":
            id_parts.append(parent_item.id)
            parent_item = parent_item.get_parent()
        id_parts.reverse()
        return self._data_id_delimiter.join(id_parts)

    def get_item_data_ids(self, items: Iterable[pystac.Item]) -> Iterator[str]:
        """Generates the data IDs of an item collection,
        which follows the structure:

            `collection_id_0/../collection_id_n/item_id`

        Args:
            items: item collection

        Yields:
            data ID of an item
        """
        for item in items:
            yield self.get_item_data_id(item)

    def list_item_data_ids(self, items: Iterable[pystac.Item]) -> List[str]:
        """Generates a list of data IDs for a given item collection,
        which follows the structure:

            `collection_id_0/../collection_id_n/item_id`

        Args:
            items: item collection

        Returns:
            list of data IDs for a given item collection
        """
        return list(self.get_item_data_ids(items))

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        items: Iterable[pystac.Item] = None,
        item_data_ids: Iterable[str] = None,
        include_attrs: Container[str] = None,
        **open_params
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        """Get an iterator over the data resource identifiers. The data
        resource identifiers follow the following structure:

            `collection_id_0/../collection_id_n/item_id/asset_id`

        Args:
            items: collection of items for which data IDs are desired. If None,
                items are collected by :meth:`get_item_collection` using
                *open_params*. Defaults to None.
            item_data_ids: data IDs corresponding to items. If None,
                item_data_ids are collected by :meth:`get_item_data_ids`.
                Defaults to None.
            include_attrs: A sequence of names of attributes to be returned
                for each dataset identifier. If given, the store will attempt
                to provide the set of requested dataset attributes in addition
                to the data ids. If no attributes are found, empty dictionary
                is returned. So far only the attribute 'title' is supported.
                Defaults to None.

        Yields:
            An iterator over the identifiers (and additional attributes defined
            by *include_attrs* of data resources provided by this data store).
        """
        if data_type is not None:
            warnings.warn(
                f'data_type is set to {data_type}, but is not used.'
            )
        if items is None:
            items, item_data_ids = self.get_item_collection(**open_params)
        if item_data_ids is None:
            item_data_ids = self.get_item_data_ids(items)

        for (item, item_data_id) in zip(items, item_data_ids):
            for asset in self.get_assets_from_item(
                item, include_attrs, **open_params
            ):
                if include_attrs is not None:
                    (asset, attrs) = asset
                    data_id = (
                        item_data_id + self._data_id_delimiter + asset
                    )
                    yield (data_id, attrs)
                else:
                    data_id = item_data_id + self._data_id_delimiter + asset
                    yield data_id

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if self._is_valid_data_type(data_type):
            return data_id in self.list_data_ids()
        return False

    def get_assets_from_item(
        self,
        item: pystac.Item,
        include_attrs: Container[str] = None,
        **open_params
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        """Get all assets for a given item, which has a MIME data type

        Args:
            item: item/feature
            include_attrs: A sequence of names of attributes to be returned
                for each dataset identifier. If given, the store will attempt
                to provide the set of requested dataset attributes in addition
                to the data ids. If no attributes are found, empty dictionary
                is returned. So far only the attribute 'title' is supported.
                Defaults to None.

        Yields:
            An iterator over the assets (and additional attributes defined
            by *include_attrs* of data resources provided by this data store).
        """
        for k, v in item.assets.items():
            # test if asset is in variable_names and the media type is
            # one of the predefined MIME types
            if (
                k in open_params.get("variable_names", [k]) and
                any(x in MIME_TYPES for x in v.media_type.split("; "))
            ):
                # TODO: support more attributes
                if include_attrs is not None:
                    attrs = {}
                    if "title" in include_attrs and hasattr(v, "title"):
                        attrs["title"] = v.title
                    yield (k, attrs)
                else:
                    yield k

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
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_ITEM_PARAMETERS),
            required=[],
            additional_properties=False
        )

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
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        # ToDo: implement open_data method.
        raise NotImplementedError("open_data() operation is not supported yet")

    def describe_data(
        self, data_id: str, **open_params
    ) -> DatasetDescriptor:
        """Get the descriptor for the data resource given by *data_id*.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            Data descriptor containing meta data of
            the data resources identified by *data_id*
        """
        # ToDo: implement describe_data method.
        raise NotImplementedError("describe_data() operation is not supported yet")

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

    def _get_items_nonsearchable_catalog(
        self,
        pystac_object: Union[pystac.Catalog, pystac.Collection],
        recursive: bool = True,
        **open_params
    ) -> Iterator[pystac.Item]:
        """Get the items of a catalog which does not implement the
        "STAC API - Item Search" conformance class.

        Args:
            pystac_object: either a `pystac.catalog:Catalog` or a
                `pystac.collection:Collection` object
            recursive: If True, the data IDs of a multiple-collection
                and/or nested-collection STAC catalog can be collected. If False,
                a flat STAC catalog hierarchy is assumed, consisting only of items.

        Yields:
            An iterator over the items matching the **open_params.
        """

        if (
            pystac_object.extra_fields["type"] != "Collection" or
            pystac_object.id in open_params.get("collections", [pystac_object.id])
        ):
            if recursive:
                if any(True for _ in pystac_object.get_children()):
                    iterators = (self._get_items_nonsearchable_catalog(
                        child,
                        recursive=True,
                        **open_params
                    ) for child in pystac_object.get_children())
                    yield from chain(*iterators)
                else:
                    iterator = self._get_items_nonsearchable_catalog(
                        pystac_object,
                        recursive=False,
                        **open_params
                    )
                    yield from iterator
            else:
                for item in pystac_object.get_items():
                    # test if item's bbox intersects with the desired bbox
                    if "bbox" in open_params:
                        if not self._do_bboxes_intersect(item, **open_params):
                            continue
                    # test if item fit to desired time range
                    if "time_range" in open_params:
                        if not self._is_datetime_in_range(item, **open_params):
                            continue
                    # iterate through assets of item
                    yield item

    def _is_datetime_in_range(self, item: pystac.Item, **open_params) -> bool:
        """Determine whether the datetime or datetime range of an item
        intersects to the 'time_range' given by *open_params*.

        Args:
            item: item/feature
            open_params: Optional opening parameters which need
                to include 'time_range'


        Returns:
            True, if the datetime of an item is within the 'time_range',
            or if there is any overlap between the 'time_range' and
            the datetime range of an item; otherwise False.

        """
        dt_start = pd.Timestamp(
            open_params["time_range"][0]
        ).to_pydatetime().replace(tzinfo=timezone.utc)
        dt_end = pd.Timestamp(
            open_params["time_range"][1]
        ).to_pydatetime().replace(tzinfo=timezone.utc)
        if item.properties["datetime"] == "null":
            dt_start_data = pd.Timestamp(
                item.properties["start_datetime"]
            ).to_pydatetime()
            dt_end_data = pd.Timestamp(
                item.properties["end_datetime"]
            ).to_pydatetime()
            return dt_end >= dt_start_data and dt_start <= dt_end_data
        else:
            dt_data = pd.Timestamp(item.properties["datetime"]).to_pydatetime()
            return dt_start <= dt_data <= dt_end

    def _do_bboxes_intersect(self, item: pystac.Item, **open_params) -> bool:
        """Determine whether two bounding boxes intersect.

        Args:
            item: item/feature
            open_params: Optional opening parameters which need
                to include 'bbox'

        Returns:
            True if the bounding box given by the item intersects with
            the bounding box given by *open_params*. Otherwise False.
        """
        return box(*item.bbox).intersects(box(*open_params["bbox"]))
