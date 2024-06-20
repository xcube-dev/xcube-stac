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
from typing import Union
import warnings

import numpy as np
import pandas as pd
import pystac
from shapely.geometry import box
from xcube.core.store import DataStoreError

from .constants import MAP_MIME_TYP_FORMAT
from .stac import _get_assets_from_item


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
            "Either 'start_datetime' and 'end_datetime' or 'datetime' "
            "needs to be determined in the STAC item."
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


def _update_nested_dict(dic: dict, dic_update: dict):
    """It updates a nested dictionary with two levels.

    Args:
        dic: dictionary to be updated
        dic_update: update dictionary

    Returns:
        dic: updated dictionary
    """
    for key, val in dic_update.items():
        if isinstance(val, dict):
            dic[key] = _update_nested_dict(dic.get(key, {}), val)
        else:
            dic[key] = val
    return dic


def _get_formats(item: pystac.Item) -> np.array:
    """It transforms the MIME-types of all assets to the data types used in xcube

    Args:
        item: item/feature

    Returns:
        array containing all formats
    """
    assets = list(_get_assets_from_item(item))
    return np.unique(
        np.array(
            [MAP_MIME_TYP_FORMAT[asset.media_type.split("; ")[0]] for asset in assets]
        )
    )


def _get_opener_id(
    asset: pystac.Asset, formats: np.array, protocol: str, opener_id: str = None
) -> str:
    """It selects the opener ID for a given asset MIME-type. If all assets are geotiffs,
    a mldataset, otherwise a dataset will be used. If *opener_id* does not fot to the
    format of the asset, the prescribed *opener_id* will be overwritten and a warning
    will be raised.

    Args:
        asset: asset object
        formats: formats of all assets in one item extracted from the MIME-type
        protocol: protocol extracted from the href
        opener_id: prescribed opener identifier

    Returns:
        opener_id_asset: opener identifier for the asset
    """
    if opener_id is None:
        opener_id_asset = _select_opener_id(asset, formats, protocol)
    else:
        asset_format = MAP_MIME_TYP_FORMAT[asset.media_type.split("; ")[0]]
        if opener_id.split(":")[0] != asset_format:
            opener_id_asset = _select_opener_id(asset, formats, protocol)
            warnings.warn(
                f"The format of the asset {asset.title} is {asset_format}. "
                f"The opener is changed from {opener_id} to {opener_id_asset}"
            )
        else:
            opener_id_asset = opener_id
    return opener_id_asset


def _select_opener_id(asset: pystac.Asset, formats: np.array, protocol: str) -> str:
    if len(formats) == 1 and formats[0] == "geotiff":
        opener_id_asset = f"mldataset:geotiff:{protocol}"
    else:
        asset_format = MAP_MIME_TYP_FORMAT[asset.media_type.split("; ")[0]]
        opener_id_asset = f"mldataset:{asset_format}:{protocol}"
    return opener_id_asset
