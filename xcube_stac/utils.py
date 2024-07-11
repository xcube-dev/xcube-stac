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

import copy
import datetime
import itertools
from typing import Any, Container, Dict, Iterator, Union
import warnings

import numpy as np
import pandas as pd
import pystac
from shapely.geometry import box
import xarray as xr
from xcube.core.store import (
    DATASET_TYPE,
    MULTI_LEVEL_DATASET_TYPE,
    DataStoreError,
    DataTypeLike,
)

from .constants import DATA_OPENER_IDS, MAP_FILE_EXTENSION_FORMAT, MAP_MIME_TYP_FORMAT


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
        ] in MAP_MIME_TYP_FORMAT:
            v.extra_fields["id"] = k
            yield v


def _list_assets_from_item(
    item: pystac.Item,
    asset_names: Container[str] = None,
) -> list[pystac.Asset]:
    """Get all assets for a given item, which has a MIME data type

    Args:
        item: item/feature
        asset_names: Names of assets which will be included
            in the data cube. If None, all assets will be
            included which can be opened by the data store.

    Yields:
        An iterator over the assets
    """
    return list(_get_assets_from_item(item, asset_names=asset_names))


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


def _convert_str2datetime(datetime_str: str) -> datetime.datetime:
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


def _convert_datetime2str(dt: Union[datetime.datetime, datetime.date]) -> str:
    """Converting datetime to ISO 8601 string.

    Args:
        dt: datetime object

    Returns:
        datetime string
    """
    return dt.isoformat()


def _is_datetime_in_range(item: pystac.Item, **open_params) -> bool:
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

    Raises:
        DataStoreError: Error, if either 'start_datetime' and 'end_datetime'
        nor 'datetime' is determined in the STAC item.
    """
    dt_start = _convert_str2datetime(open_params["time_range"][0])
    dt_end = _convert_str2datetime(open_params["time_range"][1])
    if "start_datetime" in item.properties and "end_datetime" in item.properties:
        dt_start_data = _convert_str2datetime(item.properties["start_datetime"])
        dt_end_data = _convert_str2datetime(item.properties["end_datetime"])
        return dt_end >= dt_start_data and dt_start <= dt_end_data
    elif "datetime" in item.properties:
        dt_data = _convert_str2datetime(item.properties["datetime"])
        return dt_start <= dt_data <= dt_end
    else:
        raise DataStoreError(
            "The item`s property needs to contain either 'start_datetime' and "
            "'end_datetime' or 'datetime'."
        )


def _do_bboxes_intersect(item: pystac.Item, **open_params) -> bool:
    """Determine whether two bounding boxes intersect.

    Args:
        item: item/feature
        open_params: Optional opening parameters which need
            to include 'bbox'

    Returns:
        True if the bounding box given by the item intersects with
        the bounding box given by *open_params*, otherwise False.
    """
    return box(*item.bbox).intersects(box(*open_params["bbox"]))


def _update_dict(dic: dict, dic_update: dict, inplace: bool = True) -> dict:
    """It updates a dictionary recursively.

    Args:
        dic: dictionary to be updated
        dic_update: update dictionary
        inplace: if True *dic* will be overwritten; if False copy if *dic* is
            performed before it is updated.

    Returns:
        dic: updated dictionary
    """
    if not inplace:
        dic = copy.deepcopy(dic)
    for key, val in dic_update.items():
        if isinstance(val, dict):
            dic[key] = _update_dict(dic.get(key, {}), val)
        else:
            dic[key] = val
    return dic


def _get_url_from_item(item: pystac.Item) -> str:
    """Extracts the URL an item object.

    Args:
        item: Item object

    Returns:
        the URL of an item.
    """
    links = [link for link in item.links if link.rel == "self"]
    assert len(links) == 1
    return links[0].href


def _get_formats_from_item(
    item: pystac.Item, asset_names: Container[str] = None
) -> np.array:
    """It transforms the MIME-types of selected assets stored within an item to the
    format IDs used in xcube.

    Args:
        item: item/feature
        asset_names: Names of assets which will be included
            in the data cube. If None, all assets will be
            included which can be opened by the data store.

    Returns:
        array containing all formats
    """
    assets = _list_assets_from_item(item, asset_names=asset_names)
    return _get_formats_from_assets(assets)


def _get_formats_from_assets(assets: list[pystac.Asset]) -> np.array:
    """It transforms the MIME-types of multiple assets to the
    format IDs used in xcube.

    Args:
        assets: list of assets

    Returns:
        array containing all format IDs
    """
    return np.unique(np.array([_get_format_from_asset(asset) for asset in assets]))


def _get_format_from_asset(asset: pystac.Asset) -> str:
    """It transforms the MIME-types of one assets to the format IDs used in xcube.

    Args:
        asset: one asset object

    Returns: format ID

    """
    return MAP_MIME_TYP_FORMAT[asset.media_type.split("; ")[0]]


def _xarray_rename_vars(
    ds: Union[xr.Dataset, xr.DataArray], name_dict: dict
) -> Union[xr.Dataset, xr.DataArray]:
    """Auxiliary functions which turns the method xarray.Dataset.rename_vars and
    xarray.DataArray.rename_vars into a function which takes the Dataset or DataArray
    as argument.

    Args:
        ds: Dataset or DataArray
        name_dict: Dictionary whose keys are current variable names and whose values
            are the desired names.

    Returns:
        Dataset with renamed variables
    """
    return ds.rename_vars(name_dict)


def _is_valid_data_type(data_type: DataTypeLike) -> bool:
    """Auxiliary function to check if data type is supported
    by the store.

    Args:
        data_type: Data type that is to be checked.

    Returns:
        True if *data_type* is supported by the store, otherwise False
    """
    return (
        data_type is None
        or DATASET_TYPE.is_super_type_of(data_type)
        or MULTI_LEVEL_DATASET_TYPE.is_super_type_of(data_type)
    )


def _assert_valid_data_type(data_type: DataTypeLike):
    """Auxiliary function to assert if data type is supported
    by the store.

    Args:
        data_type: Data type that is to be checked.

    Raises:
        DataStoreError: Error, if *data_type* is not
            supported by the store.
    """
    if not _is_valid_data_type(data_type):
        raise DataStoreError(
            f"Data type must be {DATASET_TYPE.alias!r} or "
            f"{MULTI_LEVEL_DATASET_TYPE.alias!r}, but got {data_type!r}."
        )


def _is_valid_ml_data_type(data_type: DataTypeLike) -> bool:
    """Auxiliary function to check if data type is a multi-level
    dataset type.

    Args:
        data_type: Data type that is to be checked.

    Returns:
        True if *data_type* is a multi-level dataset type, otherwise False
    """
    return MULTI_LEVEL_DATASET_TYPE.is_super_type_of(data_type)


def _assert_valid_opener_id(opener_id: str):
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
            f"{DATA_OPENER_IDS}, but got {opener_id!r}."
        )
