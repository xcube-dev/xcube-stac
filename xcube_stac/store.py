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

import datetime
import itertools
import json
from typing import Any, Container, Dict, Iterator, List, Tuple, Union

import pandas as pd
import pystac
import pystac_client
import requests
from shapely.geometry import box
from xcube.core.store import (
    DATASET_TYPE,
    DatasetDescriptor,
    DataStore,
    DataStoreError,
    DataTypeLike,
)
from xcube.util.jsonschema import JsonObjectSchema, JsonStringSchema

from .constants import (
    DATASET_OPENER_ID,
    MIME_TYPES,
    STAC_OPEN_PARAMETERS,
    STAC_SEARCH_PARAMETERS,
)


_CATALOG_JSON = "catalog.json"


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

        # TODO: Add a data store "file" here when implementing
        # open_data(), which will be used to open the hrefs

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        stac_params = dict(url=JsonStringSchema(title="URL to STAC catalog"))
        return JsonObjectSchema(
            description=("Describes the parameters of the xcube data store 'stac'."),
            properties=stac_params,
            required=["url"],
            additional_properties=False,
        )

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return (DATASET_TYPE.alias,)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        return self.get_data_types()

    def get_data_ids(
        self, data_type: DataTypeLike = None, include_attrs: Container[str] = None
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        self._assert_valid_data_type(data_type)
        for item in self._catalog.get_items(recursive=True):
            data_id = self._get_data_id_from_item(item)
            if include_attrs is None:
                yield data_id
            else:
                attrs = self._get_attrs_from_item(item, include_attrs)
                yield (data_id, attrs)

    def has_data(self, data_id: str, data_type: DataTypeLike = None) -> bool:
        if self._is_valid_data_type(data_type):
            return data_id in self.list_data_ids()
        return False

    def get_data_opener_ids(
        self, data_id: str = None, data_type: DataTypeLike = None
    ) -> Tuple[str, ...]:
        self._assert_valid_data_type(data_type)
        if data_id is not None and not self.has_data(data_id, data_type=data_type):
            raise DataStoreError(f"Data resource {data_id!r} is not available.")
        return (DATASET_OPENER_ID,)

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        self._assert_valid_opener_id(opener_id)
        # ToDo: Further parameters needs to be added for actual data access.
        return JsonObjectSchema(
            properties=dict(**STAC_OPEN_PARAMETERS),
            required=[],
            additional_properties=False,
        )

    def open_data(
        self, data_id: str, opener_id: str = None, **open_params
    ) -> List[pystac.Asset]:
        # ToDo: Actual access of the data needs to be implemented.
        stac_schema = self.get_open_data_params_schema()
        stac_schema.validate_instance(open_params)
        self._assert_valid_opener_id(opener_id)
        item = self._access_item(data_id)
        assets = self._get_assets_from_item(item, **open_params)
        return list(assets)

    def describe_data(
        self, data_id: str, data_type: DataTypeLike = None
    ) -> DatasetDescriptor:
        self._assert_valid_data_type(data_type)
        item = self._access_item(data_id)

        # prepare metadata
        if "start_datetime" in item.properties and "end_datetime" in item.properties:
            time_range = (
                self._convert_datetime2str(
                    self._convert_str2datetime(item.properties["start_datetime"]).date()
                ),
                self._convert_datetime2str(
                    self._convert_str2datetime(item.properties["end_datetime"]).date()
                ),
            )
        elif "datetime" in item.properties:
            time_range = (
                self._convert_datetime2str(
                    self._convert_str2datetime(item.properties["datetime"]).date()
                ),
                None,
            )
        else:
            DataStoreError(
                f"The item with the data ID {data_id!r} cannot be described."
                "The item`s property needs to contain either 'start_datetime' and "
                "'end_datetime' or 'datetime'."
            )
        metadata = dict(bbox=item.bbox, time_range=time_range)
        return DatasetDescriptor(data_id, **metadata)

    def search_data(
        self, data_type: DataTypeLike = None, **search_params
    ) -> Iterator[DatasetDescriptor]:
        self._assert_valid_data_type(data_type)
        if self._searchable:
            # rewrite to "datetime"
            time_range = search_params.pop("time_range", None)
            if time_range:
                search_params["datetime"] = "/".join(time_range)
            items = self._catalog.search(**search_params).items()
        else:
            items = self._search_nonsearchable_catalog(self._catalog, **search_params)
        for item in items:
            data_id = self._get_data_id_from_item(item)
            yield self.describe_data(data_id, data_type=data_type)

    @classmethod
    def get_search_params_schema(
        cls, data_type: DataTypeLike = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(**STAC_SEARCH_PARAMETERS),
            required=[],
            additional_properties=False,
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
                f"Data type must be {DATASET_TYPE!r}, " f"but got {data_type!r}"
            )

    @classmethod
    def _assert_valid_opener_id(cls, opener_id: str):
        """Auxiliary function to assert if data opener identified by
        *opener_id* is supported by the store.

        Args:
            opener_id: Data opener identifier

        Raises:
            DataStoreError: Error, if *opener_id* is not
                supported by the store.
        """
        if opener_id is not None and opener_id != DATASET_OPENER_ID:
            raise DataStoreError(
                f"Data opener identifier must be "
                f"{DATASET_OPENER_ID!r}, but got {opener_id!r}"
            )

    def _search_nonsearchable_catalog(
        self,
        pystac_object: Union[pystac.Catalog, pystac.Collection],
        recursive: bool = True,
        **search_params,
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

        if pystac_object.extra_fields[
            "type"
        ] != "Collection" or pystac_object.id in search_params.get(
            "collections", [pystac_object.id]
        ):
            if recursive:
                if any(True for _ in pystac_object.get_children()):
                    iterators = (
                        self._search_nonsearchable_catalog(
                            child, recursive=True, **search_params
                        )
                        for child in pystac_object.get_children()
                    )
                    yield from itertools.chain(*iterators)
                else:
                    iterator = self._search_nonsearchable_catalog(
                        pystac_object, recursive=False, **search_params
                    )
                    yield from iterator
            else:
                for item in pystac_object.get_items():
                    # test if item's bbox intersects with the desired bbox
                    if "bbox" in search_params:
                        if not self._do_bboxes_intersect(item, **search_params):
                            continue
                    # test if item fit to desired time range
                    if "time_range" in search_params:
                        if not self._is_datetime_in_range(item, **search_params):
                            continue
                    # iterate through assets of item
                    yield item

    def _convert_str2datetime(self, datetime_str: str) -> datetime.datetime:
        """Converting datetime string to a datetime object, which can handle
        the ISO 8601 suffix 'Z'.

        Args:
            datetime_str: datetime string

        Returns:
            dt: datetime object
        """
        dt = pd.Timestamp(datetime_str).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt

    def _convert_datetime2str(self, dt: datetime.datetime) -> str:
        """Converting datetime to ISO 8601 string.

        Args:
            dt: datetime object

        Returns:
            datetime string
        """
        return dt.isoformat()

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
        dt_start = self._convert_str2datetime(open_params["time_range"][0])
        dt_end = self._convert_str2datetime(open_params["time_range"][1])
        if "start_datetime" in item.properties and "end_datetime" in item.properties:
            dt_start_data = self._convert_str2datetime(
                item.properties["start_datetime"]
            )
            dt_end_data = self._convert_str2datetime(item.properties["end_datetime"])
            return dt_end >= dt_start_data and dt_start <= dt_end_data
        elif "datetime" in item.properties:
            dt_data = self._convert_str2datetime(item.properties["datetime"])
            return dt_start <= dt_data <= dt_end
        else:
            DataStoreError(
                "Either 'start_datetime' and 'end_datetime' or 'datetime' "
                "needs to be determine in the STAC item."
            )

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

    def _access_item(self, data_id: str) -> pystac.Item:
        """Access item for a given data ID.

        Args:
            data_id: An identifier of data that is provided by this
                store.

        Returns:
            item object
        """
        response = requests.request(method="GET", url=self._url_mod + data_id)
        if response.ok:
            return pystac.Item.from_dict(
                json.loads(response.text),
                href=self._url + data_id,
                root=self._catalog,
                preserve_dict=False,
            )
        else:
            DataStoreError(response.raise_for_status())

    def _get_assets_from_item(
        self, item: pystac.Item, **open_params
    ) -> Iterator[pystac.Asset]:
        """Get all assets for a given item, which has a MIME data type

        Args:
            item: item/feature

        Yields:
            An iterator over the assets
        """
        for k, v in item.assets.items():
            # test if asset is in 'asset_names' and the media type is
            # one of the predefined MIME types; note that if asset_names
            # is ot given all assets are returned matching the MINE types;
            if k in open_params.get("asset_names", [k]) and any(
                x in MIME_TYPES for x in v.media_type.split("; ")
            ):
                v.extra_fields["id"] = k
                yield v

    def _get_data_id_from_item(self, item: pystac.Item) -> str:
        """Extracts the data ID from an item object.

        Args:
            item: Item object

        Returns:
            data ID consisting the URL section of an item
            following the catalog URL.
        """
        links = [link for link in item.links if link.rel == "self"]
        assert len(links) == 1
        return links[0].href.replace(self._url_mod, "")

    def _get_attrs_from_item(
        self, item: pystac.Item, include_attrs: Container[str]
    ) -> str:
        """Extracts the desired attributes from an item object.

        Args:
            item: Item object
            include_attrs: A sequence of names of attributes to be returned
                for each dataset identifier. If given, the store will attempt
                to provide the set of requested dataset attributes in addition
                to the data ids. If no attributes are found, empty dictionary
                is returned.

        Returns:
            dictionary containing the attributes defined by *include_attrs*
            of data resources provided by this data store
        """
        attrs = {}
        if "id" in include_attrs and hasattr(item, "id"):
            attrs["id"] = item.id
        if "bbox" in include_attrs and hasattr(item, "bbox"):
            attrs["bbox"] = item.bbox
        if "geometry" in include_attrs and hasattr(item, "geometry"):
            attrs["geometry"] = item.geometry
        if "properties" in include_attrs and hasattr(item, "properties"):
            attrs["properties"] = item.properties
        if "links" in include_attrs and hasattr(item, "links"):
            attrs["links"] = item.links
        if "assets" in include_attrs and hasattr(item, "assets"):
            attrs["assets"] = item.assets
        return attrs
