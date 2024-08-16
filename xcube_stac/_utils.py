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
import os
from typing import Any, Container, Dict, Iterator, Union

import numpy as np
import odc.geo
import pandas as pd
import pyproj
import pystac
import rasterio
from shapely.geometry import box
import xarray as xr
from xcube.core.store import (
    DATASET_TYPE,
    MULTI_LEVEL_DATASET_TYPE,
    DataStoreError,
    DataTypeLike,
)

from .constants import (
    CATALOG_JSON,
    DATA_OPENER_IDS,
    FloatInt,
    MAP_FILE_EXTENSION_FORMAT,
    MAP_MIME_TYP_FORMAT,
)


def get_assets_from_item(
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


def list_assets_from_item(
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
    return list(get_assets_from_item(item, asset_names=asset_names))


def search_nonsearchable_catalog(
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
                    search_nonsearchable_catalog(child, recursive=True, **search_params)
                    for child in pystac_object.get_children()
                )
                yield from itertools.chain(*iterators)
            else:
                iterator = search_nonsearchable_catalog(
                    pystac_object, recursive=False, **search_params
                )
                yield from iterator
        else:
            for item in pystac_object.get_items():
                # test if item's bbox intersects with the desired bbox
                if "bbox" in search_params:
                    if not do_bboxes_intersect(item.bbox, **search_params):
                        continue
                # test if item fit to desired time range
                if "time_range" in search_params:
                    if not is_item_in_time_range(item, **search_params):
                        continue
                # iterate through assets of item
                yield item


def search_collections(
    catalog: pystac.Catalog,
    **search_params,
) -> Iterator[pystac.Item]:
    """Get the collections of a catalog for given search parameters

    Args:
        catalog: pystac catalog object

    Yields:
        An iterator over the items matching the **open_params.
    """

    for collection in catalog.get_collections():
        # test if collection's bbox intersects with the desired bbox
        if "bbox" in search_params:
            if not do_bboxes_intersect(
                collection.extent.spatial.bboxes[0], **search_params
            ):
                continue
        # test if collection fit to desired time range
        if "time_range" in search_params:
            if not is_collection_in_time_range(collection, **search_params):
                continue
        # iterate through assets of item
        yield collection


def get_attrs_from_pystac_object(
    pystac_obj: Union[pystac.Item, pystac.Collection], include_attrs: Container[str]
) -> Dict[str, Any]:
    """Extracts the desired attributes from an item object.

    Args:
        pystac_obj: Item or collection object
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
    supported_keys = [
        "id",
        "title",
        "description",
        "keywords",
        "extent",
        "summaries",
        "bbox",
        "geometry",
        "properties",
        "links",
        "assets",
    ]
    for key in supported_keys:
        if key in include_attrs and hasattr(pystac_obj, key):
            attrs[key] = getattr(pystac_obj, key)
    return attrs


def convert_str2datetime(datetime_str: str) -> datetime.datetime:
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


def convert_datetime2str(dt: Union[datetime.datetime, datetime.date]) -> str:
    """Converting datetime to ISO 8601 string.

    Args:
        dt: datetime object

    Returns:
        datetime string
    """
    return dt.isoformat()


def is_item_in_time_range(item: pystac.Item, **open_params) -> bool:
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
    dt_start = convert_str2datetime(open_params["time_range"][0])
    dt_end = convert_str2datetime(open_params["time_range"][1])
    if "start_datetime" in item.properties and "end_datetime" in item.properties:
        dt_start_data = convert_str2datetime(item.properties["start_datetime"])
        dt_end_data = convert_str2datetime(item.properties["end_datetime"])
        return dt_end >= dt_start_data and dt_start <= dt_end_data
    elif "datetime" in item.properties:
        dt_data = convert_str2datetime(item.properties["datetime"])
        return dt_start <= dt_data <= dt_end
    else:
        raise DataStoreError(
            "The item`s property needs to contain either 'start_datetime' and "
            "'end_datetime' or 'datetime'."
        )


def is_collection_in_time_range(collection: pystac.Collection, **open_params) -> bool:
    """Determine whether collection temporal extent
    intersects to the 'time_range' given by *open_params*.

    Args:
        collection: pystac collection object
        open_params: Optional opening parameters which need
            to include 'time_range'

    Returns:
        True, if the temporal extent of a collection is within or overlapping
        the 'time_range'; otherwise False.
    """
    dt_start = convert_str2datetime(open_params["time_range"][0])
    dt_end = convert_str2datetime(open_params["time_range"][1])
    temp_extent = collection.extent.temporal.intervals[0]
    if temp_extent[1] is None:
        return temp_extent[0] <= dt_end
    elif temp_extent[0] is None:
        return dt_start <= temp_extent[1]
    else:
        return dt_end >= temp_extent[0] and dt_start <= temp_extent[1]


def do_bboxes_intersect(
    bbox_test: [FloatInt, FloatInt, FloatInt, FloatInt], **open_params
) -> bool:
    """Determine whether two bounding boxes intersect.

    Args:
        bbox_test: bounding box to be tested against the bounding box given
            in *open_params*.
        open_params: Optional opening parameters which need
            to include 'bbox'.

    Returns:
        True if the bounding box given by the item intersects with
        the bounding box given by *open_params*, otherwise False.
    """
    return box(*bbox_test).intersects(box(*open_params["bbox"]))


def update_dict(dic: dict, dic_update: dict, inplace: bool = True) -> dict:
    """It updates a dictionary recursively.

    Args:
        dic: dictionary to be updated
        dic_update: update dictionary
        inplace: if True *dic* will be overwritten; if False copy of *dic* is
            performed before it is updated.

    Returns:
        dic: updated dictionary
    """
    if not inplace:
        dic = copy.deepcopy(dic)
    for key, val in dic_update.items():
        if isinstance(val, dict):
            dic[key] = update_dict(dic.get(key, {}), val)
        else:
            dic[key] = val
    return dic


def get_url_from_pystac_object(
    pystac_obj: Union[pystac.Item, pystac.collection]
) -> str:
    """Extracts the URL an item object.

    Args:
        pystac_obj: Item or collection object

    Returns:
        the URL of an item.
    """
    links = [link for link in pystac_obj.links if link.rel == "self"]
    assert len(links) == 1
    return links[0].href


def get_format_from_path(path: str) -> str:
    _, file_extension = os.path.splitext(path)
    return MAP_FILE_EXTENSION_FORMAT[file_extension]


def xarray_rename_vars(
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


def is_valid_data_type(data_type: DataTypeLike) -> bool:
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


def assert_valid_data_type(data_type: DataTypeLike):
    """Auxiliary function to assert if data type is supported
    by the store.

    Args:
        data_type: Data type that is to be checked.

    Raises:
        DataStoreError: Error, if *data_type* is not
            supported by the store.
    """
    if not is_valid_data_type(data_type):
        raise DataStoreError(
            f"Data type must be {DATASET_TYPE.alias!r} or "
            f"{MULTI_LEVEL_DATASET_TYPE.alias!r}, but got {data_type!r}."
        )


def is_valid_ml_data_type(data_type: DataTypeLike) -> bool:
    """Auxiliary function to check if data type is a multi-level
    dataset type.

    Args:
        data_type: Data type that is to be checked.

    Returns:
        True if *data_type* is a multi-level dataset type, otherwise False
    """
    return MULTI_LEVEL_DATASET_TYPE.is_super_type_of(data_type)


def assert_valid_opener_id(opener_id: str):
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


def get_data_id_from_pystac_object(
    pystac_obj: Union[pystac.Item, pystac.Collection], catalog_url: str
) -> str:
    """Extracts the data ID from an item object.

    Args:
        pystac_obj: Item or collection object.
        catalog_url: internally modified catalog URL.

    Returns:
        data ID consisting the URL section of an item
        following the catalog URL.
    """
    return get_url_from_pystac_object(pystac_obj).replace(catalog_url, "")


def modify_catalog_url(url: str) -> str:
    url_mod = url
    if url_mod[-len(CATALOG_JSON) :] == "catalog.json":
        url_mod = url_mod[:-12]
    if url_mod[-1] != "/":
        url_mod += "/"
    return url_mod


def get_resolutions_cog(
    item: pystac.Item,
    asset_names: Container[str] = None,
    crs: str = None,
) -> list[odc.geo.Resolution]:
    """This function calculates the resolution for each overview level of
    a cloud-optimized GeoTIFF (COG).

    Args:
        item: item/feature object.
        asset_names: asset names to b included in the dataset.
        crs: crs of the dataset output.

    Returns:
        list of odc-geo resolution objects for each overview layer.
    """
    assets = list_assets_from_item(item, asset_names=asset_names)
    resolutions = np.full(len(assets), np.inf)
    for i, asset in enumerate(assets):
        raster_bands = asset.extra_fields.get("raster:bands")
        if not raster_bands:
            break
        resolutions[i] = asset.extra_fields["raster:bands"][0]["spatial_resolution"]
    idx_min = np.argmin(resolutions)
    with rasterio.open(assets[idx_min].href) as rio_dataset_reader:
        overviews = [1] + rio_dataset_reader.overviews(1)
        data_resolution = rio_dataset_reader.res
        data_crs = rio_dataset_reader.crs
    if crs:
        transformer = pyproj.Transformer.from_crs(data_crs, crs)
        pmin = transformer.transform(0, 0)
        pmax = transformer.transform(data_resolution[0], data_resolution[1])
        res_transformed = [pmax[0] - pmin[0], pmax[1] - pmin[1]]
    else:
        res_transformed = data_resolution

    return [
        odc.geo.resxy_(
            res_transformed[0] * overview,
            res_transformed[1] * overview,
        )
        for overview in overviews
    ]


def apply_scaling_nodata(
    ds: xr.Dataset, items: Union[pystac.Item, list[pystac.Item]]
) -> xr.Dataset:
    """This function applies scaling of the data and fills no-data pixel with np.nan.

    Args:
        ds: dataset
        items: item object or list of item objects (depending on stack-mode equal to
            False and True, respectively.)

    Returns:
        Dataset where scaling and filling nodata values are applied.
    """
    if isinstance(items, pystac.Item):
        items = [items]

    if items[0].ext.has("raster"):
        for data_varname in ds.data_vars.keys():
            scale = np.ones(len(items))
            offset = np.zeros(len(items))
            nodata_val = np.zeros(len(items))
            for i, item in enumerate(items):
                raster_bands = item.assets[data_varname].extra_fields.get(
                    "raster:bands"
                )
                if not raster_bands:
                    break
                nodata_val[i] = raster_bands[0].get("nodata", 0)
                if "scale" in raster_bands[0]:
                    scale[i] = raster_bands[0]["scale"]
                if "offset" in raster_bands[0]:
                    offset[i] = raster_bands[0]["offset"]

            nodata_val = np.unique(nodata_val)
            msg = (
                "Items contain different values in the "
                "asset's field 'raster:bands:nodata'"
            )
            assert len(nodata_val) == 1, msg
            nodata_val = nodata_val[0]
            ds[data_varname] = ds[data_varname].where(ds[data_varname] != nodata_val)

            scale = np.unique(scale)
            msg = (
                "Items contain different values in the "
                "asset's field 'raster:bands:scale'"
            )
            assert len(scale) == 1, msg
            ds[data_varname] *= scale[0]

            offset = np.unique(offset)
            msg = (
                "Items contain different values in the "
                "asset's field 'raster:bands:offset'"
            )
            assert len(offset) == 1, msg
            ds[data_varname] += offset[0]
    return ds
