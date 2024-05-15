# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
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
from typing import Tuple, Iterable, Iterator, Union, List

import pandas as pd
import pystac
from pystac import Catalog, Collection, ItemCollection, Item
import pystac_client
from shapely.geometry import box
import xarray as xr
from xcube.util.jsonschema import JsonObjectSchema

from .constants import STAC_SEARCH_ITEM_PARAMETERS


class Stac:
    """Common operations on STAC catalogs.

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

    @property
    def catalog(self) -> Catalog:
        return self._catalog

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """Get the JSON schema for instantiating a new data store.

        Returns:
            The JSON schema for the data store's parameters.
        """
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_ITEM_PARAMETERS),
            required=[],
            additional_properties=False
        )

    def get_item_collection(
        self, **open_params
    ) -> Tuple[ItemCollection, List[str]]:
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
            items = self.catalog.search(**open_params).item_collection()
        else:
            items = self._get_items_nonsearchable_catalog(
                self.catalog,
                **open_params
            )
            items = ItemCollection(items)
        item_data_ids = self.list_item_data_ids(items)
        return items, item_data_ids

    def get_item_data_id(self, item: Item) -> str:
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

    def get_item_data_ids(self, items: Iterable[Item]) -> Iterator[str]:
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

    def list_item_data_ids(self, items: Iterable[Item]) -> List[str]:
        """Generates a list of data IDs for a given item collection,
        which follows the structure:

            `collection_id_0/../collection_id_n/item_id`

        Args:
            items: item collection

        Returns:
            list of data IDs for a given item collection
        """
        return list(self.get_item_data_ids(items))

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        """Open the data given by the data resource identifier *data_id*
        using the supplied *open_params*.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Raises:
            NotImplementedError: Not implemented yet.

        Returns:
            A representation of the data resources
            identified by *data_id* and *open_params*.
        """
        # ToDo: implement this method using data store "file", see __init__()
        raise NotImplementedError("open_data() operation is not supported yet")

    def _get_items_nonsearchable_catalog(
        self,
        pystac_object: Union[Catalog, Collection],
        recursive: bool = True,
        **open_params
    ) -> Iterator[Tuple[Item, str]]:
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

    def _is_datetime_in_range(self, item: Item, **open_params) -> bool:
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

    def _do_bboxes_intersect(self, item: Item, **open_params) -> bool:
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
