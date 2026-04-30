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
from collections.abc import Container, Iterator, Sequence
from typing import Any
import time

import dask.array as da
import numpy as np
import pandas as pd
import pyproj
import pystac
import pystac_client
import requests
import xarray as xr
from shapely.geometry import box
from scipy.interpolate import RBFInterpolator
from xcube.core.store import MULTI_LEVEL_DATASET_TYPE, DataStoreError, DataTypeLike
from xcube_resampling import affine_transform_dataset
from xcube_resampling.constants import FillValues
from xcube_resampling.gridmapping import GridMapping

from .constants import (
    LOG,
    MAP_FILE_EXTENSION_FORMAT,
    MAP_MIME_TYP_FORMAT,
    MLDATASET_FORMATS,
    TILE_SIZE,
    FloatInt,
    _CRS_WGS84,
    CDSE_S3_ENDPOINT,
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


def bbox_to_geojson(bbox):
    min_x, min_y, max_x, max_y = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]
        ],
    }


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
    item: pystac.Item, asset_names: Sequence[str] | None = None
) -> list[pystac.Asset]:
    selected_keys = asset_names if asset_names is not None else item.assets.keys()
    assets = []

    for key in selected_keys:
        asset = item.assets.get(key)
        if asset is None:
            LOG.warning(
                "Asset name '%s' not found in assets of item '%s'.", key, item.id
            )
            continue

        format_id = get_format_id(asset)
        if format_id is not None:
            asset.extra_fields["xcube:asset_id"] = key
            asset.extra_fields["xcube:format_id"] = format_id
            assets.append(asset)

    if not assets:
        raise DataStoreError(
            "No valid assets found in item '%s' for asset_names=%s."
            % (item.id, asset_names)
        )

    return assets


def add_nominal_datetime(items: Sequence[pystac.Item]) -> Sequence[pystac.Item]:
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


def access_item(url: str, catalog: pystac.Catalog, max_retries: int = 5) -> pystac.Item:
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
    headers = {"User-Agent": "my-stac-client/1.0"}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 429:
                # Respect server hint if available
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else 2**attempt

                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            return pystac.Item.from_dict(
                data,
                href=url,
                root=catalog,
                preserve_dict=False,
            )

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise DataStoreError(f"Failed to access STAC item at {url}: {e}")

            # backoff for other transient errors
            time.sleep(2**attempt)

    raise DataStoreError(f"Exceeded retries for STAC item at {url}")


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
        protocol, _, _ = decode_href(asset.href)
        protocols.append(protocol)
    return list(np.unique(protocols))


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


def merge_datasets(
    datasets: list[xr.Dataset],
    target_gm: GridMapping = None,
    fill_values: FillValues = None,
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
        fill_values: fill values propagated to xcube_resampling.affine_transform

    Returns:
        A single dataset resulting from merging all input datasets,
        resampled to a common grid if necessary.
    """
    sourcce_gm = GridMapping.from_dataset(datasets[0])
    x_coord, y_coord = sourcce_gm.xy_var_names
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
                affine_transform_dataset(
                    ds, target_gm=target_gm, fill_values=fill_values
                )
            )
        ds = _update_datasets(datasets_resampled)
    return ds


def _get_tile_size(open_params: dict) -> tuple[int, int]:
    tile_size = open_params.get("tile_size", TILE_SIZE)
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    return tile_size


def _update_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    ds = datasets[0].copy()
    for ds_asset in datasets[1:]:
        ds.update(ds_asset)
    return ds


def mosaic_spatial_take_first(
    list_ds: list[xr.Dataset], var_ref: str, fill_value: int | float
) -> xr.Dataset:
    """Creates a spatial mosaic from a list of datasets by taking the first
    non-fill value encountered across datasets at each pixel location.

    The function assumes all datasets share the same spatial dimensions and coordinate
    system. Only variables with 2D spatial dimensions are processed. At each
    spatial location, the first non-fill (or non-NaN) value across the dataset stack
    is selected.

    Args:
        list_ds: A list of datasets to be mosaicked.
        var_ref: reference variable used for the index selection
        fill_value: The value considered as missing data in the reference variable

    Returns:
        A new dataset representing the mosaicked result, using the first valid
        value encountered across the input datasets for each spatial position.
    """
    if len(list_ds) == 1:
        return list_ds[0]

    arr_ref = da.stack([ds[var_ref].data for ds in list_ds], axis=0)
    if np.isnan(fill_value):
        nonnan_mask = ~da.isnan(arr_ref)
    else:
        nonnan_mask = arr_ref != fill_value
    first_non_nan_index = nonnan_mask.argmax(axis=0)

    ds_mosaic = xr.Dataset(attrs=list_ds[0].attrs)
    for key in list_ds[0]:
        # allow to also merge viewing angles of Sen2 with grid (angle_y, angle_x)
        if list_ds[0][key].ndim >= 2:
            da_arr = da.stack([ds[key].data for ds in list_ds], axis=0)
            da_arr_select = da.choose(first_non_nan_index, da_arr)
            ds_mosaic[key] = xr.DataArray(
                da_arr_select,
                dims=list_ds[0][key].dims,
                coords=list_ds[0][key].coords,
                attrs=list_ds[0][key].attrs,
            )

    return ds_mosaic


def build_footprint_uv_mapping(
    points: np.ndarray,
    orbit_state: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Create geometry control points and normalized image coordinates.

    Args:
        points: Boundary coordinates in ring order.
        orbit_state: Orbit direction, either "ascending" or "descending".

    Returns:
        A tuple `(control_xy, control_uv)` where `control_xy` are boundary
        coordinates and `control_uv` are corresponding normalized image
        coordinates.
    """
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    lon = points[:, 0]
    lat = points[:, 1]

    idx_ll = int(np.argmin(lat + lon))
    idx_ur = int(np.argmax(lat + lon))
    idx_ul = int(np.argmax(lat - lon))
    idx_lr = int(np.argmin(lat - lon))

    control_xy = np.array(
        [
            [lon[idx_ll], lat[idx_ll]],
            [lon[idx_lr], lat[idx_lr]],
            [lon[idx_ul], lat[idx_ul]],
            [lon[idx_ur], lat[idx_ur]],
        ]
    )

    if orbit_state == "ascending":
        control_uv = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    else:
        control_uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    return control_xy, control_uv


def find_relative_bbox(
    item: pystac.Item, bbox: Sequence[float | int]
) -> Sequence[float]:
    points = np.array(item.geometry["coordinates"][0])
    orbit_state = item.properties["sat:orbit_state"]

    # convert to utm
    center = np.mean(points, axis=0)
    utm_zone = int(np.floor((center[0] + 180) / 6) + 1)
    if center[1] >= 0:
        utm_epsg = f"EPSG:326{utm_zone}"
    else:
        utm_epsg = f"EPSG:327{utm_zone}"
    transformer = pyproj.Transformer.from_crs(_CRS_WGS84, utm_epsg, always_xy=True)
    utm_points = transformer.transform(points[:, 0], points[:, 1])
    utm_points = np.stack(utm_points).transpose()
    utm_bbox = transformer.transform_bounds(*bbox, densify_pts=21)

    control_xy, control_uv = build_footprint_uv_mapping(utm_points, orbit_state)
    u_model = RBFInterpolator(control_xy, control_uv[:, 0], kernel="thin_plate_spline")
    v_model = RBFInterpolator(control_xy, control_uv[:, 1], kernel="thin_plate_spline")

    corners = np.array(
        [
            [utm_bbox[0], utm_bbox[1]],
            [utm_bbox[2], utm_bbox[1]],
            [utm_bbox[0], utm_bbox[3]],
            [utm_bbox[2], utm_bbox[3]],
        ]
    )
    us = u_model(corners)
    vs = v_model(corners)

    u_min = np.clip(np.min(us), 0, 1)
    u_max = np.clip(np.max(us), 0, 1)
    v_min = np.clip(np.min(vs), 0, 1)
    v_max = np.clip(np.max(vs), 0, 1)

    return u_min, v_min, u_max, v_max


def clip_dataset_relative_bbox(
    rel_bbox: Sequence[float], ds: xr.Dataset, buffer: int | tuple[int, int] = 20
) -> tuple[xr.Dataset, tuple[int, int, int, int]] | tuple[None, None]:
    if isinstance(buffer, int):
        buffer = (buffer, buffer)

    w, h = ds.sizes["x"] - 1, ds.sizes["y"] - 1
    col_min = int(np.clip((rel_bbox[0] * w) - buffer[0], 0, w))
    row_min = int(np.clip((rel_bbox[1] * h) - buffer[1], 0, h))
    col_max = int(np.clip((rel_bbox[2] * w) + buffer[0], 0, w))
    row_max = int(np.clip((rel_bbox[3] * h) + buffer[1], 0, h))

    ds_sub = ds.isel(y=slice(row_min, row_max), x=slice(col_min, col_max))

    if any(size <= 1 for size in ds_sub.sizes.values()):
        LOG.info(
            "Clipping with the specified bounding box resulted in a "
            "dataset too small to compute a valid grid mapping. Returning None.",
        )
        return None, None

    return ds_sub, (col_min, row_min, col_max, row_max)


def _set_cdse_env_vars(key: str = None, secret: str = None) -> None:
    import os

    if key is not None:
        os.environ.update({"AWS_ACCESS_KEY_ID": key})
    if secret is not None:
        os.environ.update({"AWS_SECRET_ACCESS_KEY": secret})

    missing = [
        name
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(
            f"Missing AWS credentials for DEM download: {missing}."
            "Set these environment variables for CDSE DEM access "
            "(https://documentation.dataspace.copernicus.eu/APIs/S3.html#generate-secrets) "
            "or provide a DEM directly."
        )
    os.environ.update(
        {
            "AWS_S3_ENDPOINT": CDSE_S3_ENDPOINT.split("//")[1],
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_DEFAULT_REGION": "default",
        }
    )
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
