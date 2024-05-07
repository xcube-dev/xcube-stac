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

from typing import Tuple, Iterable, Iterator, Union, List

from datetime import datetime, timezone
from shapely.geometry import box
import xarray as xr
from itertools import chain

import pystac

import pystac_client
from pystac import Catalog, Collection, ItemCollection, Item

from xcube.util.jsonschema import JsonObjectSchema

from xcube_stac.constants import STAC_SEARCH_ITEM_PARAMETERS


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
        self.catalog = catalog

        # TODO: Add a data store "file" here when implementing
        # open_data(), which will be used to open the hrefs

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        """Get the JSON schema for instantiating a new data store.

        Returns:
            The JSON schema for the data store's parameters.
        """
        stac_schema = JsonObjectSchema(
            properties=dict(**STAC_SEARCH_ITEM_PARAMETERS),
            required=[],
            additional_properties=False
        )
        return stac_schema

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
            open_params_mod = open_params.copy()
            if "variable_names" in open_params_mod:
                del open_params_mod["variable_names"]
            if "time_range" in open_params_mod:
                open_params_mod["datetime"] = "/".join(open_params_mod["time_range"])
                del open_params_mod["time_range"]
            items = self.catalog.search(**open_params_mod).item_collection()
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
        pystac_object = item
        while pystac_object.STAC_OBJECT_TYPE != "Catalog":
            id_parts.append(pystac_object.id)
            pystac_object = pystac_object.get_parent()
        id_parts.reverse()
        return self._data_id_delimiter.join(id_parts)

    def get_item_data_ids(self, items: Iterable[Item]) -> Iterator[str]:
        """Generates the data ID of an item collection,
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
        """Get the items for a catalog of the catalog, which is not
        conform with the 'ITEM_SEARCH' specifications.

        Args:
            pystac_object: either a `pystac.catalog:Catalog` or a
                `pystac.collection:Collection` object
            recursive: If True, data IDs of a multiple collection
                and/or nested collection STAC catalog can be build. If False,
                flat STAC catalog hierarchy is assumed consisting only of items.
                Defaults to True.

        Yields:
            An iterator over the items matching the **open_params.
        """

        if (
            pystac_object.extra_fields["type"] == "Collection" and
            pystac_object.id not in open_params.get("collections", [pystac_object.id])
        ):
            pass
        else:
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
                        if not self._assert_bbox_intersect(item, **open_params):
                            continue
                    # test if item fit to desired time range
                    if "time_range" in open_params:
                        if not self._assert_datetime(item, **open_params):
                            continue
                    # iterate through assets of item
                    yield item

    def _assert_datetime(self, item: Item, **open_params) -> bool:
        """Assert if the datetime or datetime range of an item fits to the
        'time_range' given by *open_params*.

        Args:
            item: item/feature

        Returns:
            True, if the datetime of an item is within the 'time_range',
            otherwise False. True, if there is any overlap between the
            'time_range' and the datetime range of an item.

        """
        dt_start = datetime.fromisoformat(
            open_params["time_range"][0]
        ).replace(tzinfo=timezone.utc)
        dt_end = datetime.fromisoformat(
            open_params["time_range"][1]
        ).replace(tzinfo=timezone.utc)
        if item.properties["datetime"] == "null":
            dt_start_data = datetime.fromisoformat(
                item.properties["start_datetime"]
            )
            dt_end_data = datetime.fromisoformat(
                item.properties["end_datetime"]
            )
            if dt_end < dt_start_data or dt_start > dt_end_data:
                return False
            else:
                return True
        else:
            dt_data = datetime.fromisoformat(item.properties["datetime"])
            if dt_end < dt_data or dt_start > dt_data:
                return False
            else:
                return True

    def _assert_bbox_intersect(self, item: Item, **open_params) -> bool:
        """Checks if two bounding boxes intersect.

        Args:
            item: item/feature

        Returns:
            True if the bounding box given by the item intersects with
            the bounding box given by *open_params*. Otherwise False.
        """
        return box(*item.bbox).intersects(box(*open_params["bbox"]))
