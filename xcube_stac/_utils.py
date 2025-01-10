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

import collections
import copy
import datetime
import itertools
import os
from typing import Any, Container, Dict, Iterator, Union

import numpy as np
import pandas as pd
import pyproj
import pystac
import pystac_client
from shapely.geometry import box
import xarray as xr
from xcube.core.store import DATASET_TYPE
from xcube.core.store import MULTI_LEVEL_DATASET_TYPE
from xcube.core.store import DataStoreError
from xcube.core.store import DataTypeLike
from xcube.core.geom import clip_dataset_by_geometry
from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_space

from .constants import TILE_SIZE
from .constants import DATA_OPENER_IDS
from .constants import FloatInt
from .constants import MAP_FILE_EXTENSION_FORMAT
from .constants import MAP_MIME_TYP_FORMAT
from .constants import LOG


_CATALOG_JSON = "catalog.json"


def search_items(
    catalog: Union[pystac.Catalog, pystac_client.client.Client],
    searchable: bool,
    **search_params,
) -> Iterator[pystac.Item]:
    if searchable:
        # rewrite to "datetime"
        search_params["datetime"] = search_params.pop("time_range", None)
        items = catalog.search(**search_params).items()
    else:
        items = search_nonsearchable_catalog(catalog, **search_params)
    return items


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


def get_format_id(asset: pystac.Asset) -> str:
    if asset.media_type is None:
        format_id = get_format_from_path(asset.href)
    else:
        format_id = MAP_MIME_TYP_FORMAT.get(asset.media_type.split("; ")[0])
    if format_id is None:
        LOG.debug(f"No format_id found for asset {asset.title!r}")
    return format_id


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


def add_nominal_datetime(items: list[pystac.Item]) -> list[pystac.Item]:
    for item in items:
        item.properties["center_point"] = get_center_from_bbox(item.bbox)
        item.properties["datetime_nominal"] = convert_to_solar_time(
            item.datetime, item.properties["center_point"][0]
        )
    return items


def get_processing_version(item: pystac.Item) -> float:
    return float(
        item.properties.get(
            "processing:version",
            item.properties.get("s2:processing_baseline", "1.0"),
        )
    )


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
    return MAP_FILE_EXTENSION_FORMAT.get(file_extension)


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
    if url_mod[-len(_CATALOG_JSON) :] == "catalog.json":
        url_mod = url_mod[:-12]
    if url_mod[-1] != "/":
        url_mod += "/"
    return url_mod


def reproject_bbox(
    source_bbox: list[FloatInt, FloatInt, FloatInt, FloatInt],
    source_crs: Union[pyproj.CRS, str],
    target_crs: Union[pyproj.CRS, str],
    buffer: float = 0.05,
):
    source_crs = normalize_crs(source_crs)
    target_crs = normalize_crs(target_crs)
    if source_crs == target_crs:
        return source_bbox
    t = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    target_bbox = t.transform_bounds(*source_bbox, densify_pts=21)
    x_min = target_bbox[0]
    x_max = target_bbox[2]
    if target_crs.is_geographic and x_min > x_max:
        x_max += 360
    buffer_x = abs((x_max - x_min)) * buffer
    buffer_y = abs((target_bbox[3] - target_bbox[1])) * buffer
    target_bbox = (
        target_bbox[0] - buffer_x,
        target_bbox[1] - buffer_y,
        target_bbox[2] + buffer_x,
        target_bbox[3] + buffer_y,
    )

    return target_bbox


def convert_to_solar_time(
    utc: datetime.datetime, longitude: float
) -> datetime.datetime:
    # offset_seconds snapped to 1 hour increments
    # 1/15 == 24/360 (hours per degree of longitude)
    offset_seconds = int(longitude / 15) * 3600
    return utc + datetime.timedelta(seconds=offset_seconds)


def normalize_crs(crs: Union[str, pyproj.CRS]) -> pyproj.CRS:
    if isinstance(crs, pyproj.CRS):
        return crs
    else:
        return pyproj.CRS.from_string(crs)


def get_center_from_bbox(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def rename_dataset(ds: xr.Dataset, asset: str) -> xr.Dataset:
    if len(list(ds.keys())) == 1:
        name_dict = {var_name: f"{asset}" for var_name in ds.data_vars.keys()}
    else:
        name_dict = {
            var_name: f"{asset}_{var_name}" for var_name in ds.data_vars.keys()
        }
    return ds.rename_vars(name_dict=name_dict)


def get_gridmapping(
    bbox: list[float],
    spatial_res: float,
    crs: Union[str, pyproj.crs.CRS],
    tile_size: Union[int, tuple[int, int]] = TILE_SIZE,
) -> GridMapping:
    x_size = int((bbox[2] - bbox[0]) / spatial_res) + 1
    y_size = int(abs(bbox[3] - bbox[1]) / spatial_res) + 1
    return GridMapping.regular(
        size=(x_size, y_size),
        xy_min=(bbox[0] - spatial_res / 2, bbox[1] - spatial_res / 2),
        xy_res=spatial_res,
        crs=crs,
        tile_size=tile_size,
    )


def merge_datasets(
    datasets: list[xr.Dataset], target_gm: GridMapping = None
) -> xr.Dataset:
    y_coord, x_coord = get_spatial_dims(datasets[0])
    x_ress = [abs(float((ds[x_coord][1] - ds[x_coord][0]))) for ds in datasets]
    y_ress = [abs(float(ds[y_coord][1] - ds[y_coord][0])) for ds in datasets]
    if (
        np.unique(x_ress).size == 1
        and np.unique(y_ress).size == 1
        and target_gm is None
    ):
        ds = _update_datasets(datasets)
    else:
        if target_gm is None:
            idx = np.argmin(x_ress)
            target_gm = GridMapping.from_dataset(datasets[idx])
        grouped = collections.defaultdict(lambda: collections.defaultdict(list))
        for idx, (x_res, y_res) in enumerate(zip(x_ress, y_ress)):
            grouped[x_res][y_res].append(idx)
        datasets_grouped = []
        for _, val in grouped.items():
            for _, idxs in val.items():
                datasets_grouped.append(
                    _update_datasets([datasets[idx] for idx in idxs])
                )
        datasets_resampled = []
        for ds in datasets_grouped:
            datasets_resampled.append(wrapper_resample_in_space(ds, target_gm))
        ds = _update_datasets(datasets_resampled)
    if "spatial_ref" in ds.coords:
        ds["crs"] = ds.coords["spatial_ref"]
        ds = ds.drop_vars("spatial_ref")
    return ds


def get_spatial_dims(ds: xr.Dataset) -> (str, str):
    if "lat" in ds and "lon" in ds:
        y_coord, x_coord = "lat", "lon"
    elif "y" in ds and "x" in ds:
        y_coord, x_coord = "y", "x"
    else:
        raise DataStoreError("No spatial dimensions found in dataset.")
    return y_coord, x_coord


def _update_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    ds = datasets[0].copy()
    for ds_asset in datasets[1:]:
        ds.update(ds_asset)
    return ds


def wrapper_clip_dataset_by_geometry(ds: xr.Dataset, **open_params) -> xr.Dataset:
    crs_asset = None
    if "crs" in ds:
        crs_asset = ds.crs.attrs["crs_wkt"]
    if "spatial_ref" in ds:
        crs_asset = ds.spatial_ref.attrs["crs_wkt"]
    if crs_asset and "bbox" in open_params and "crs" in open_params:
        bbox = reproject_bbox(
            open_params["bbox"],
            open_params["crs"],
            crs_asset,
        )
        ds = clip_dataset_by_geometry(ds, geometry=bbox)
    return ds


def wrapper_resample_in_space(ds: xr.Dataset, target_gm: GridMapping) -> xr.Dataset:
    ds = resample_in_space(
        ds,
        target_gm=target_gm,
        encode_cf=True,
        # rectify_kwargs=dict(compute_subset=False),
    )
    vars = [
        "spatial_ref",
        "x_bnds",
        "y_bnds",
        "lon_bnds",
        "lat_bnds",
        "transformed_x",
        "transformed_y",
    ]
    vars_sel = []
    for var in vars:
        if var in ds:
            vars_sel.append(var)
    return ds.drop_vars(vars_sel)
