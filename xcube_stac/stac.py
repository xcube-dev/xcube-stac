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

import itertools
from typing import Any, Container, Dict, Iterator, Union

import pystac

from .constants import MAP_MIME_TYP_DATAOPENER_ID
from .utils import _do_bboxes_intersect, _is_datetime_in_range


def _get_assets_from_item(
    item: pystac.Item,
    asset_names: Container[str] = None,
) -> Iterator[pystac.Asset]:
    """Get all assets for a given item, which has a MIME data type

    Args:
        item: item/feature
        asset_names: Names of assets which will be included
            in the data cube. If None, all assets will be
            included which can be opened by the data store.

    Yields:
        An iterator over the assets
    """
    for k, v in item.assets.items():
        # test if asset is in 'asset_names' and the media type is
        # one of the predefined MIME types; note that if asset_names
        # is ot given all assets are returned matching the MINE types;
        if (asset_names is None or k in asset_names) and v.media_type.split("; ")[
            0
        ] in MAP_MIME_TYP_DATAOPENER_ID:
            v.extra_fields["id"] = k
            yield v


def _search_nonsearchable_catalog(
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
                    _search_nonsearchable_catalog(
                        child, recursive=True, **search_params
                    )
                    for child in pystac_object.get_children()
                )
                yield from itertools.chain(*iterators)
            else:
                iterator = _search_nonsearchable_catalog(
                    pystac_object, recursive=False, **search_params
                )
                yield from iterator
        else:
            for item in pystac_object.get_items():
                # test if item's bbox intersects with the desired bbox
                if "bbox" in search_params:
                    if not _do_bboxes_intersect(item, **search_params):
                        continue
                # test if item fit to desired time range
                if "time_range" in search_params:
                    if not _is_datetime_in_range(item, **search_params):
                        continue
                # iterate through assets of item
                yield item


def _get_attrs_from_item(
    item: pystac.Item, include_attrs: Container[str]
) -> Dict[str, Any]:
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
