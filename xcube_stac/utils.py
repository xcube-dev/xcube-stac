# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
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
import json
import os
from collections.abc import Sequence, Container, Iterator
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import pyproj
import pystac
import pystac_client
import requests
import xarray as xr
from shapely.geometry import box
from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.dataset import new_grid_mapping_from_dataset
from xcube.core.resampling import affine_transform_dataset
from xcube.core.store import MULTI_LEVEL_DATASET_TYPE, DataStoreError, DataTypeLike

from .constants import (
    MAP_FILE_EXTENSION_FORMAT,
    MAP_MIME_TYP_FORMAT,
    MLDATASET_FORMATS,
    TILE_SIZE,
    FloatInt,
)
from .href_parse import decode_href

_CATALOG_JSON = "catalog.json"


def search_items(
    catalog: pystac.Catalog | pystac_client.client.Client,
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
    pystac_object: pystac.Catalog | pystac.Collection,
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
        An iterator over the items matching the **search_params.
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
) -> Iterator[pystac.Collection]:
    """Get the collections of a catalog for given search parameters

    Args:
        catalog: pystac catalog object

    Yields:
        An iterator over the items matching the **search_params.
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
    format_id = get_format_from_path(asset.href)
    if format_id is None:
        if isinstance(asset.media_type, str):
            format_id = MAP_MIME_TYP_FORMAT.get(asset.media_type.split("; ")[0])
    return format_id


def get_attrs_from_pystac_object(
    pystac_obj: pystac.Item | pystac.Collection, include_attrs: Container[str] | bool
) -> dict[str, Any]:
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
        "type",
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
        "stac_version",
        "stac_extensions",
        "collection",
    ]
    for key in supported_keys:
        if hasattr(pystac_obj, key) and (include_attrs is True or key in include_attrs):
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


def convert_datetime2str(dt: datetime.datetime | datetime.date) -> str:
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


def list_assets_from_item(
    item: pystac.Item, asset_names: Sequence[str] = None
) -> list[pystac.Asset]:
    assets = []
    for key, asset in item.assets.items():
        format_id = get_format_id(asset)
        if (asset_names is None or key in asset_names) and format_id is not None:
            asset.extra_fields["xcube:asset_id"] = key
            asset.extra_fields["xcube:format_id"] = format_id
            assets.append(asset)
    return assets


def add_nominal_datetime(items: list[pystac.Item]) -> list[pystac.Item]:
    """Adds the nominal (solar) time to each STAC item's properties under the key
    "datetime_nominal", based on the item's original UTC datetime.

    Args:
        items: A list of STAC item objects.

    Returns:
        A list of STAC item objects with the "datetime_nominal" field added to their
        properties.
    """

    for item in items:
        item.properties["center_point"] = get_center_from_bbox(item.bbox)
        item.properties["datetime_nominal"] = convert_to_solar_time(
            item.datetime, item.properties["center_point"][0]
        )
    return items


def get_grid_mapping_name(ds: xr.Dataset) -> str | None:
    """Extracts the name of the grid mapping variable from an xarray Dataset.

    The function searches through the dataset's data variables for the "grid_mapping"
    attribute, as well as for commonly used coordinate variables like "crs" and
    "spatial_ref". It ensures that at most one unique grid mapping name is present.

    Args:
        ds: A dataset to inspect for grid mapping information.

    Returns:
        The name of the grid mapping variable if found, otherwise None.

    Raises:
        AssertionError: If more than one unique grid mapping name is detected.
    """
    gm_names = []
    for var in ds.data_vars:
        if "grid_mapping" in ds[var].attrs:
            gm_names.append(ds[var].attrs["grid_mapping"])
    if "crs" in ds:
        gm_names.append("crs")
    if "spatial_ref" in ds.coords:
        gm_names.append("spatial_ref")
    gm_names = np.unique(gm_names)
    assert len(gm_names) <= 1, "Multiple grid mapping names found."
    if len(gm_names) == 1:
        return str(gm_names[0])
    else:
        return None


def normalize_grid_mapping(ds: xr.Dataset) -> xr.Dataset:
    """Normalizes the grid mapping in a dataset to use a standard "spatial_ref"
    coordinate.

    This function replaces any existing grid mapping references in the dataset with a
    unified "spatial_ref" coordinate. It updates the "grid_mapping" attribute of all
    data variables to reference "spatial_ref", removes the original grid mapping
    variable (if present), and adds a new "spatial_ref" coordinate with CF-compliant
    CRS attributes.

    Args:
        ds: A dataset containing geospatial data with grid mapping metadata.

    Returns:
        A dataset with a standardized "spatial_ref" coordinate used for grid mapping.
    """
    gm_name = get_grid_mapping_name(ds)
    if gm_name is None:
        return ds
    for var in ds.data_vars:
        ds[var].attrs["grid_mapping"] = "spatial_ref"
    gm = new_grid_mapping_from_dataset(ds)
    ds = ds.drop_vars(gm_name)
    ds = ds.assign_coords(spatial_ref=xr.DataArray(0, attrs=gm.crs.to_cf()))
    return ds


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


def get_url_from_pystac_object(pystac_obj: pystac.Item | pystac.Collection) -> str:
    """Extracts the URL an item object.

    Args:
        pystac_obj: Item or collection object

    Returns:
        the URL of an item.
    """
    links = [
        link
        for link in pystac_obj.links
        if link.rel == "self" and link.href.startswith("http")
    ]
    assert len(links) == 1
    return links[0].href


def get_format_from_path(path: str) -> str:
    """Returns the data format corresponding to a file's extension derived from a path.

    Args:
        path: The file path from which to extract the format.

    Returns:
        A string representing the data format associated with the file extension.
        Returns None if the extension is not found in the mapping.
    """
    _, file_extension = os.path.splitext(path)
    return MAP_FILE_EXTENSION_FORMAT.get(file_extension)


def is_valid_ml_data_type(data_type: DataTypeLike) -> bool:
    """Auxiliary function to check if data type is a multi-level
    dataset type.

    Args:
        data_type: Data type that is to be checked.

    Returns:
        True if *data_type* is a multi-level dataset type, otherwise False
    """
    return MULTI_LEVEL_DATASET_TYPE.is_super_type_of(data_type)


def get_data_id_from_pystac_object(
    pystac_obj: pystac.Item | pystac.Collection, catalog_url: str
) -> str:
    """Extracts the data ID from an item object.

    Args:
        pystac_obj: Item or collection object.
        catalog_url: internally modified catalog URL.

    Returns:
        data ID consisting the URL section of an item
        following the catalog URL.
    """
    return get_url_from_pystac_object(pystac_obj).replace(f"{catalog_url}/", "")


def modify_catalog_url(url: str) -> str:
    """Normalizes a STAC catalog URL by removing a trailing 'catalog.json' if present.

    Args:
        url: The original STAC catalog URL.

    Returns:
        A normalized URL string without 'catalog.json'.
    """
    if url.endswith(_CATALOG_JSON):
        url = url.replace(_CATALOG_JSON, "")
    if url[-1] == "/":
        url = url[:-1]
    return url


def access_item(url: str, catalog: pystac.Catalog) -> pystac.Item:
    """Retrieves and parses a STAC item associated with the given data ID.

    Args:
        url: url to STAC item
        catalog: pystac catalog object

    Returns:
        A `pystac.Item` object representing the STAC item.

    Raises:
        DataStoreError: If the item cannot be retrieved from the catalog or
        if the response cannot be parsed into a valid STAC item.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise DataStoreError(f"Failed to access STAC item at {url}: {e}")

    try:
        return pystac.Item.from_dict(
            json.loads(response.text),
            href=url,
            root=catalog,
            preserve_dict=False,
        )
    except Exception as e:
        raise DataStoreError(f"Failed to parse STAC item JSON at {url}: {e}")


def access_collection(url: str, catalog: pystac.Catalog) -> pystac.Collection:
    """Retrieves and parses a STAC collection associated with the given data ID.

    Args:
        url: url to STAC collection
        catalog: pystac catalog object

    Returns:
        A `pystac.Collection` object representing the STAC collection.

    Raises:
        DataStoreError: If the collection cannot be retrieved from the catalog or
        if the response cannot be parsed into a valid STAC collection.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise DataStoreError(f"Failed to access STAC collection at {url}: {e}")

    try:
        return pystac.Collection.from_dict(
            json.loads(response.text),
            href=url,
            root=catalog,
            preserve_dict=False,
        )
    except Exception as e:
        raise DataStoreError(f"Failed to parse SATC collection JSON at {url}: {e}")


def is_mldataset_available(
    item: pystac.Item, asset_names: Sequence[int] = None
) -> bool:
    format_ids = list_format_ids(item, asset_names=asset_names)
    return all(format_id in MLDATASET_FORMATS for format_id in format_ids)


def list_format_ids(item: pystac.Item, asset_names: Sequence[str] = None) -> list[str]:
    assets = list_assets_from_item(item, asset_names=asset_names)
    return list(np.unique([asset.extra_fields["xcube:format_id"] for asset in assets]))


def list_protocols(item: pystac.Item, asset_names: Sequence[str] = None) -> list[str]:
    assets = list_assets_from_item(item, asset_names=asset_names)
    protocols = []
    for asset in assets:
        protocol, _, _, _ = decode_href(asset.href)
        protocols.append(protocol)
    return list(np.unique(protocols))


def reproject_bbox(
    source_bbox: tuple[int] | tuple[float] | list[int] | list[float],
    source_crs: pyproj.CRS | str,
    target_crs: pyproj.CRS | str,
    buffer: float = 0.0,
) -> tuple[int] | tuple[float]:
    """Reprojects a bounding box from a source CRS to a target CRS, with optional
    buffering.

    The function transforms a bounding box defined in the source coordinate reference
    system (CRS) to the target CRS using `pyproj`. If the source and target CRS are
    the same, no transformation is performed. An optional buffer (as a fraction of
    width/height) can be applied to expand the resulting bounding box.

    Args:
        source_bbox: The bounding box to reproject, in the form
            (min_x, min_y, max_x, max_y).
        source_crs: The source CRS, as a `pyproj.CRS` or string.
        target_crs: The target CRS, as a `pyproj.CRS` or string.
        buffer: Optional buffer to apply to the transformed bounding box, expressed as
                a fraction (e.g., 0.1 for 10% padding). Default is 0.0 (no buffer).

    Returns:
        A tuple representing the reprojected (and optionally buffered) bounding box:
        (min_x, min_y, max_x, max_y).
    """
    source_crs = normalize_crs(source_crs)
    target_crs = normalize_crs(target_crs)
    if source_crs != target_crs:
        t = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        target_bbox = t.transform_bounds(*source_bbox, densify_pts=21)
    else:
        target_bbox = source_bbox
    if buffer > 0.0:
        x_min = target_bbox[0]
        x_max = target_bbox[2]
        if target_crs.is_geographic and x_min > x_max:
            x_max += 360
        buffer_x = abs(x_max - x_min) * buffer
        buffer_y = abs(target_bbox[3] - target_bbox[1]) * buffer
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
    """Converts a UTC datetime to an approximate solar time based on longitude.

    The conversion assumes that each 15 degrees of longitude corresponds to a 1-hour
    offset from UTC, effectively snapping the time offset to whole-hour increments.
    This provides a simplified approximation of local solar time.

    Args:
        utc: The datetime in UTC.
        longitude: The longitude in degrees, where positive values are east of
        the meridian.

    Returns:
        A datetime object representing the approximate solar time.
    """
    offset_seconds = int(longitude / 15) * 3600
    return utc + datetime.timedelta(seconds=offset_seconds)


def normalize_crs(crs: str | pyproj.CRS) -> pyproj.CRS:
    """Normalizes a CRS input by converting it to a pyproj.CRS object.

    If the input is already a `pyproj.CRS` instance, it is returned unchanged.
    If the input is a string (e.g., an EPSG code or PROJ string), it is converted
    to a `pyproj.CRS` object using `CRS.from_string`.

    Args:
        crs: A CRS specified as a string or a `pyproj.CRS` object.

    Returns:
        A `pyproj.CRS` object representing the normalized CRS.
    """
    if isinstance(crs, pyproj.CRS):
        return crs
    else:
        return pyproj.CRS.from_string(crs)


def get_center_from_bbox(
    bbox: tuple[float] | tuple[int] | list[float] | list[int],
) -> tuple[float, float]:
    """Calculates the center point of a bounding box.

    Args:
        bbox: The bounding box, in the form (min_x, min_y, max_x, max_y).

    Returns:
        A tuple (center_x, center_y) representing the center coordinates of the
        bounding box.
    """
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def rename_dataset(ds: xr.Dataset, asset: str) -> xr.Dataset:
    """Renames the data variables in a dataset based on the provided asset name.

    If the dataset contains only one data variable, it is renamed to the STAC item's
    asset name. If there are multiple data variables, each is renamed using the pattern
    "{asset}_{original_variable_name}".

    Args:
        ds: The input dataset whose data variables are to be renamed.
        asset: STAC asset name used as the prefix (or full name) for renaming the
            variables.

    Returns:
        A modified dataset with renamed data variables.
    """
    if len(list(ds.keys())) == 1:
        name_dict = {var_name: f"{asset}" for var_name in ds.data_vars.keys()}
    else:
        name_dict = {
            var_name: f"{asset}_{var_name}" for var_name in ds.data_vars.keys()
        }
    return ds.rename_vars(name_dict=name_dict)


def get_gridmapping(
    bbox: tuple[float] | tuple[int] | list[float] | list[int],
    spatial_res: float | int | tuple[float | int, float | int],
    crs: str | pyproj.crs.CRS,
    tile_size: int | tuple[int, int] = TILE_SIZE,
) -> GridMapping:
    """Creates a regular GridMapping object based on a bounding box, spatial resolution,
    and CRS.

    Args:
        bbox: The bounding box in the form (min_x, min_y, max_x, max_y).
        spatial_res: Spatial resolution as a single value or a (x_res, y_res) tuple.
        crs: Coordinate reference system as a `pyproj.CRS` or a string
            (e.g., "EPSG:4326").
        tile_size: Optional tile size as a single integer or a (width, height) tuple.
            Defaults to `TILE_SIZE`.

    Returns:
        A xcube `GridMapping` object representing the regular grid layout defined
        by the input parameters.
    """
    if not isinstance(spatial_res, tuple):
        spatial_res = (spatial_res, spatial_res)
    x_size = np.ceil((bbox[2] - bbox[0]) / spatial_res[0]) + 1
    y_size = np.ceil(abs(bbox[3] - bbox[1]) / spatial_res[1]) + 1
    return GridMapping.regular(
        size=(x_size, y_size),
        xy_min=(bbox[0] - spatial_res[0] / 2, bbox[1] - spatial_res[1] / 2),
        xy_res=spatial_res,
        crs=crs,
        tile_size=tile_size,
    )


def merge_datasets(
    datasets: list[xr.Dataset], target_gm: GridMapping = None
) -> xr.Dataset:
    """Merges a list of datasets into a single dataset, optionally resampling
    to a target grid mapping.

    If all datasets share the same spatial resolution and no `target_gm` is provided,
    they are merged directly. Otherwise, the datasets are grouped by resolution,
    merged within each group, resampled to the target grid mapping, and then combined.

    Args:
        datasets: A list of datasets to be merged.
        target_gm: Optional `GridMapping` to which all datasets will be resampled.
                   If not provided, the grid mapping from the dataset with the
                   highest resolution (smallest x spacing) is used.

    Returns:
        A single dataset resulting from merging all input datasets,
        resampled to a common grid if necessary.
    """
    y_coord, x_coord = get_spatial_dims(datasets[0])
    x_ress = [abs(float(ds[x_coord][1] - ds[x_coord][0])) for ds in datasets]
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
            datasets_resampled.append(
                affine_transform_dataset(ds, target_gm=target_gm, encode_cf=False)
            )
        ds = _update_datasets(datasets_resampled)
    return ds


def get_spatial_dims(ds: xr.Dataset) -> (str, str):
    """Identifies the spatial coordinate names in a dataset.

    The function checks for common spatial dimension naming conventions: ("lat", "lon")
    or ("y", "x"). If neither pair is found, it raises a DataStoreError.

    Args:
        ds: The dataset to inspect.

    Returns:
        A tuple of strings representing the names of the spatial dimensions.

    Raises:
        DataStoreError: If no recognizable spatial dimensions are found.
    """
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


def mosaic_spatial_take_first(
    list_ds: list[xr.Dataset], fill_value: int | float = np.nan
) -> xr.Dataset:
    """Creates a spatial mosaic from a list of datasets by taking the first
    non-fill value encountered across datasets at each pixel location.

    The function assumes all datasets share the same spatial dimensions and coordinate
    system. Only variables with 2D spatial dimensions (y, x) are processed. At each
    spatial location, the first non-fill (or non-NaN) value across the dataset stack
    is selected.

    Args:
        list_ds: A list of datasets to be mosaicked.
        fill_value: The value considered as missing data. Defaults to NaN.

    Returns:
        A new dataset representing the mosaicked result, using the first valid
        value encountered across the input datasets for each spatial position.
    """
    if len(list_ds) == 1:
        return list_ds[0]

    y_coord, x_coord = get_spatial_dims(list_ds[0])
    ds_mosaic = xr.Dataset()
    for key in list_ds[0]:
        if list_ds[0][key].dims[-2:] == (y_coord, x_coord):
            da_arr = da.stack([ds[key].data for ds in list_ds], axis=0)
            if np.isnan(fill_value):
                nonnan_mask = ~da.isnan(da_arr)
            else:
                nonnan_mask = da_arr != fill_value
            first_non_nan_index = nonnan_mask.argmax(axis=0)
            da_arr_select = da.choose(first_non_nan_index, da_arr)
            ds_mosaic[key] = xr.DataArray(
                da_arr_select,
                dims=list_ds[0][key].dims,
                coords=list_ds[0][key].coords,
            )

    return ds_mosaic
