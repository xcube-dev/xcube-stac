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

from collections import defaultdict
from typing import Sequence

import dask.array as da
import numpy as np
import pyproj
import pystac
import xarray as xr
from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import (
    affine_transform_dataset,
    reproject_dataset,
    rectify_dataset,
)
from xcube.util.jsonschema import JsonObjectSchema, JsonBooleanSchema

from xcube_stac.accessor import StacItemAccessor
from xcube_stac.constants import (
    CONVERSION_FACTOR_DEG_METER,
    SCHEMA_ADDITIONAL_QUERY,
    SCHEMA_APPLY_SCALING,
    SCHEMA_ASSET_NAMES,
    SCHEMA_BBOX,
    SCHEMA_CRS,
    SCHEMA_SPATIAL_RES,
    SCHEMA_TIME_RANGE,
    TILE_SIZE,
)
from xcube_stac.stac_extension.raster import apply_offset_scaling, get_stac_extension
from xcube_stac.utils import (
    add_nominal_datetime,
    get_gridmapping,
    get_spatial_dims,
    merge_datasets,
    mosaic_spatial_take_first,
    normalize_crs,
    normalize_grid_mapping,
    rename_dataset,
    reproject_bbox,
)

SCHEMA_APPLY_RECTIFICATION = JsonBooleanSchema(
    title="Apply rectification algorithm.",
    description=("If True, data is presented on a regular grid."),
    default=True,
)


class Sen3CdseStacItemAccessor(StacItemAccessor):
    """Provides methods for accessing the data of one general STAC Item"""

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        self._storage_option_s3 = storage_options_s3

    def open_asset(self, asset: pystac.Asset, **open_params) -> xr.Dataset:
        return xr.open_dataset(
            asset.href,
            engine="h5netcdf",
            chunks=open_params.get("chunks", {}),
            backend_kwargs={},
            storage_options=self._storage_option_s3,
        )

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        coords = dict()
        ds = self.open_asset(item.assets["geolocation"])
        coords["longitude"] = ds["longitude"]
        coords["latitude"] = ds["latitude"]
        coords["altitude"] = ds["altitude"]
        ds = self.open_asset(item.assets["time"])
        coords["time"] = ds["time"]
        ds_item = xr.Dataset(coords=coords)

        assets = self._list_assets_from_item(item, **open_params)
        for asset in assets:
            if asset.extra_fields["xcube:asset_id"] in ["geolocation", "time"]:
                continue
            ds_item.update(self.open_asset(asset))

        if open_params.get("apply_rectification", False):
            ds_item = rectify_dataset(ds_item)
        return ds_item

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                asset_names=SCHEMA_ASSET_NAMES,
                apply_rectification=SCHEMA_APPLY_RECTIFICATION,
            ),
            required=[],
            additional_properties=False,
        )


class Sen3CdseStacArdcAccessor(Sen3CdseStacItemAccessor):
    """Provides methods for access multiple Sentinel-2 STAC Items from the
    CDSE STAC API and build an analysis ready data cube."""

    def open_ardc(
        self,
        items: Sequence[pystac.Item],
        **open_params,
    ) -> xr.Dataset:

        # get STAC assets grouped by solar day
        grouped_items = group_items(items)

        # apply mosaicking and stacking
        ds = self._generate_cube(grouped_items, **open_params)

        # add attributes
        # Gather all used STAC item IDs used in the data cube for each time step
        # and organize them in a dictionary. The dictionary keys are datetime
        # strings, and the values are lists of corresponding item IDs.
        ds.attrs["stac_item_ids"] = dict(
            {
                dt.astype("datetime64[ms]")
                .astype("O")
                .isoformat(): [
                    item.id for item in np.sum(grouped_items.sel(time=dt).values)
                ]
                for dt in grouped_items.time.values
            }
        )

        return ds

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                asset_names=SCHEMA_ASSET_NAMES,
                time_range=SCHEMA_TIME_RANGE,
                bbox=SCHEMA_BBOX,
                spatial_res=SCHEMA_SPATIAL_RES,
                crs=SCHEMA_CRS,
                query=SCHEMA_ADDITIONAL_QUERY,
                apply_rectification=SCHEMA_APPLY_RECTIFICATION,
            ),
            required=["time_range", "bbox", "spatial_res", "crs"],
            additional_properties=False,
        )

    def _generate_cube(self, grouped_items: xr.DataArray, **open_params) -> xr.Dataset:
        """Generate a spatiotemporal data cube from access parameters.

        This function takes grouped access parameters and generates a unified xarray
        dataset, mosaicking and stacking the data across spatial tiles and time.
        Optionally, scaling is applied and solar and viewing angle are calculated.

        Args:
            access_params: A 4D data array indexed by tile_id, asset_name, time,
                and idx, where each element contains data access metadata.
            **open_params: Optional keyword arguments for data opening and processing:
                - apply_scaling (bool): Whether to apply radiometric scaling.
                    Default is True.
                - angles_sentinel2 (bool): Whether to include Sentinel-2 solar angles.
                    Default is False.
                - bbox (list): Bounding box used for spatial subsetting.
                - crs (str): Coordinate reference system of the bounding box.
                - opener_id, data_type, etc.: Passed to the data-opening function.

        Returns:
            A dataset representing the spatiotemporal cube.
        """
        # Group the tile IDs by UTM zones
        utm_tile_id = defaultdict(list)
        for tile_id in grouped_items.tile_id.values:
            item = np.sum(grouped_items.sel(tile_id=tile_id).values)[0]
            crs = item.assets["AOT_10m"].extra_fields["proj:code"]
            utm_tile_id[crs].append(tile_id)

        # Insert the tile data per UTM zone
        list_ds_utm = []
        for crs, tile_ids in utm_tile_id.items():
            ds = self._generate_utm_cube(
                grouped_items.sel(tile_id=tile_ids), crs, **open_params
            )
            list_ds_utm.append(ds)

        # Reproject datasets from different UTM zones to a common grid reference system
        # and merge them into a single unified dataset for seamless spatial analysis.
        ds_final = _merge_utm_zones(list_ds_utm, **open_params)

        if open_params.get("add_angles", False):
            ds_final = self.add_sen2_angles_stack(ds_final, grouped_items)

        return ds_final


def group_items(items: Sequence[pystac.Item]) -> xr.DataArray:
    """Group STAC items by solar day, tile ID, and processing version.

    This method organizes a list of Sentinel-2 STAC items into an `xarray.DataArray`
    with dimensions `(time, tile_id, idx)`, where:
    - `time` corresponds to the solar acquisition date (ignoring time of day),
    - `tile_id` is the Sentinel-2 MGRS tile code,
    - `idx` accounts for up to two acquisitions per tile (e.g., due to multiple
      observations for the same tile),
    - The most recent processing version is selected if multiple exist.

    Args:
        items: List of STAC items to group. Each item must have:
            - `properties["datetime_nominal"]`: Nominal acquisition datetime.
            - `properties["grid:code"]`: Tile ID (MGRS grid).
            - A processing version recognizable by `_get_processing_version`.

    Returns:
        A 3D DataArray of shape (time, tile_id, idx) containing STAC items.
        Time coordinate values are actual datetimes (not just dates), derived
        from the nominal acquisition datetime of the first item per date/tile
        combination.

    Notes:
        - Only up to two items per (date, tile_id) and processing version
          are considered.
        - If more than two items exist for the same (date, tile_id, proc_version),
          a warning is logged and only the first two are used.
        - Among multiple processing versions, only the latest is retained.
    """
    items = add_nominal_datetime(items)

    # get dates and tile IDs of the items
    dates = []
    for item in items:
        dates.append(item.properties["datetime_nominal"].date())
    dates = np.unique(dates)

    # sort items by date and tile ID into a data array
    grouped_items = np.full(len(dates), None, dtype=object)
    for idx, item in enumerate(items):
        date = item.properties["datetime_nominal"].date()
        idx_date = np.where(dates == date)[0][0]
        if grouped_items[idx_date] is None:
            grouped_items[idx_date] = [item]
        else:
            grouped_items[idx_date].append(item)

    grouped_items = xr.DataArray(grouped_items, dims=("time",), coords=dict(time=dates))

    # replace date by datetime from first item
    dts = []
    for date in grouped_items.time.values:
        item = np.sum(grouped_items.sel(time=date).values)[0]
        dts.append(
            np.datetime64(item.properties["datetime"].replace(tzinfo=None)).astype(
                "datetime64[ns]"
            )
        )
    grouped_items = grouped_items.assign_coords(time=dts)

    return grouped_items


def _get_bounding_box(items: xr.DataArray) -> list[float | int]:
    """Compute the overall bounding box that covers all tiles in the given access
    parameters.

    Iterates through each tile in `access_params` to extract the bounding box
    from its metadata and calculates the minimum bounding rectangle encompassing all
    tiles.

    Parameters:
        items: An array containing tile metadata, with coordinates
            including 'tile_id'.

    Returns:
        A list with four elements [xmin, ymin, xmax, ymax] representing the
        bounding box that encloses all tiles.
    """
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for tile_id in items.tile_id.values:
        item = np.sum(items.sel(tile_id=tile_id).values)[0]
        bbox = item.assets["AOT_10m"].extra_fields["proj:bbox"]
        if xmin > bbox[0]:
            xmin = bbox[0]
        if ymin > bbox[1]:
            ymin = bbox[1]
        if xmax < bbox[2]:
            xmax = bbox[2]
        if ymax < bbox[3]:
            ymax = bbox[3]
    return [xmin, ymin, xmax, ymax]


def _resample_dataset_soft(
    ds: xr.Dataset,
    target_gm: GridMapping,
    fill_value: float | int = np.nan,
    interpolation: str = "nearest",
) -> xr.Dataset:
    """Resample a dataset to a target grid mapping, using either affine transform
    or reprojection.

    If the source and target grid mappings are close, the dataset is returned unchanged.
    If they share the same CRS but differ spatially, an affine transformation is applied.
    Otherwise, the dataset is reprojected to the target grid, with optional fill value
    and interpolation.

    Parameters:
        ds: The source xarray Dataset to be resampled.
        target_gm: The target grid mapping
        fill_value: Value to fill in areas without data during reprojection.
            Defaults to 0 (no-data value in the Sentinel-2 product).
        interpolation: Interpolation method to use during reprojection
            (see reproject_dataset function in xcube).

    Returns:
        The resampled dataset aligned with the target grid mapping.
    """
    source_gm = GridMapping.from_dataset(ds)
    if source_gm.is_close(target_gm):
        return ds
    if target_gm.crs == source_gm.crs:
        var_configs = {}
        for var in ds.data_vars:
            var_configs[var] = dict(recover_nan=True)
        ds = affine_transform_dataset(
            ds,
            source_gm=source_gm,
            target_gm=target_gm,
            gm_name="spatial_ref",
            var_configs=var_configs,
        )
    else:
        ds = reproject_dataset(
            ds,
            source_gm=source_gm,
            target_gm=target_gm,
            fill_value=fill_value,
            interpolation=interpolation,
        )
    return ds


def _create_empty_dataset(
    sample_ds: xr.Dataset,
    grouped_items: xr.DataArray,
    items_bbox: list[float | int] | tuple[float | int],
    final_bbox: list[float | int] | tuple[float | int],
    spatial_res: int | float,
) -> xr.Dataset:
    """Create an empty xarray Dataset with spatial and temporal dimensions matching
    the given bounding boxes and grouped items.

    The dataset is constructed using the data variables and types from `sample_ds`,
    creating arrays filled with NaNs. It conforms to the native pixel grid and spatial
    resolution of the Sentinel-2 product, while covering the spatial extent defined
    by the input bounding boxes. The temporal dimension and coordinate values are
    derived from `grouped_items`. The resulting dataset includes coordinates for
    time, y, and x dimensions, along with a matching spatial reference coordinate
    system.

    Args:
        sample_ds: A sample dataset whose data variable names and dtypes will be used.
        grouped_items: A 2D DataArray (time, tile_id) containing grouped STAC items.
        items_bbox: The bounding box covering all input items (minx, miny, maxx, maxy).
        final_bbox: The target bounding box to define the spatial extent of the final
            datacube (minx, miny, maxx, maxy).
        spatial_res: The spatial resolution in CRS units (e.g., meters or degrees).

    Returns:
        A dataset with shape (time, y, x), filled with NaNs and ready to be populated
        with mosaicked data.
    """
    half_res = spatial_res / 2
    y_start = items_bbox[3] - spatial_res * (
        (items_bbox[3] - final_bbox[3]) // spatial_res
    )
    y_end = items_bbox[1] + spatial_res * (
        (final_bbox[1] - items_bbox[1]) // spatial_res
    )
    y = np.arange(y_start - half_res, y_end, -spatial_res)
    x_end = items_bbox[2] - spatial_res * (
        (items_bbox[2] - final_bbox[2]) // spatial_res
    )
    x_start = items_bbox[0] + spatial_res * (
        (final_bbox[0] - items_bbox[0]) // spatial_res
    )
    x = np.arange(x_start + half_res, x_end, spatial_res)

    chunks = (1, TILE_SIZE, TILE_SIZE)
    shape = (grouped_items.sizes["time"], len(y), len(x))
    return xr.Dataset(
        {
            key: (
                ("time", "y", "x"),
                da.full(shape, np.nan, dtype=var.dtype, chunks=chunks),
            )
            for (key, var) in sample_ds.data_vars.items()
        },
        coords={
            "x": x,
            "y": y,
            "time": grouped_items.time,
            "spatial_ref": sample_ds.spatial_ref,
        },
    )


def _insert_tile_data(final_ds: xr.Dataset, ds: xr.Dataset, dt_idx: int) -> xr.Dataset:
    """Insert spatial data from a smaller dataset into a larger asset dataset at
    the correct spatiotemporal indices.

    This method locates the spatial coordinates of the input dataset `ds` within
    the larger `final_ds` along the 'x' and 'y' dimensions, then inserts the data
    from `ds` into the corresponding slice of `final_ds` for the specified time
    index `dt_idx`.

    Args:
        final_ds: The larger dataset representing the final data cube for one UTM zone.
        ds: The smaller xarray Dataset containing data to be inserted.
        dt_idx: The time index in `final_ds` at which to insert the data.

    Returns:
        The updated `final_ds` with data from `ds` inserted at the appropriate
        spatial location.
    """
    xmin = final_ds.indexes["x"].get_loc(ds.x[0].item())
    xmax = final_ds.indexes["x"].get_loc(ds.x[-1].item())
    ymin = final_ds.indexes["y"].get_loc(ds.y[0].item())
    ymax = final_ds.indexes["y"].get_loc(ds.y[-1].item())
    for var in ds.data_vars:
        final_ds[var][dt_idx, ymin : ymax + 1, xmin : xmax + 1] = ds[var]
    return final_ds


def _merge_utm_zones(list_ds_utm: list[xr.Dataset], **open_params) -> xr.Dataset:
    """Merge multiple Sentinel-2 datacubes for different UTM zones into a
    single dataset.

    This function takes a list of Sentinel-2 datasets (each in a different UTM zone),
    resamples them to a common grid defined by a target CRS and spatial resolution,
    and mosaics them into a single output using a "take first" strategy for overlaps.

    Parameters:
        list_ds_utm: A list of xarray Datasets, one for each UTM zone.
        open_params: Dictionary of parameters required for constructing the target grid,
            including:
            - crs: Target coordinate reference system (string or EPSG code).
            - spatial_res: Spatial resolution as a single value or tuple (x_res, y_res).
            - bbox: Bounding box for the output grid (minx, miny, maxx, maxy).
            - tile_size (optional): Tile size for the target grid.

    Returns:
        A single xarray Dataset reprojected to the target CRS and resolution,
        containing merged data from all input UTM zones.

    Notes:
        - If one input dataset already matches the target CRS and resolution,
          its grid mapping is reused unless resolution mismatches are found.
        - Overlapping regions are resolved by selecting the first non-NaN value.
    """
    # get correct target gridmapping
    crss = [pyproj.CRS.from_cf(ds["spatial_ref"].attrs) for ds in list_ds_utm]
    target_crs = pyproj.CRS.from_string(open_params["crs"])
    crss_equal = [target_crs == crs for crs in crss]
    if any(crss_equal):
        true_index = crss_equal.index(True)
        ds = list_ds_utm[true_index]
        target_gm = GridMapping.from_dataset(ds)
        spatial_res = open_params["spatial_res"]
        if not isinstance(spatial_res, tuple):
            spatial_res = (spatial_res, spatial_res)
        if (
            ds.x[1] - ds.x[0] != spatial_res[0]
            or abs(ds.y[1] - ds.y[0]) != spatial_res[1]
        ):
            target_gm = get_gridmapping(
                open_params["bbox"],
                open_params["spatial_res"],
                open_params["crs"],
                open_params.get("tile_size", TILE_SIZE),
            )
    else:
        target_gm = get_gridmapping(
            open_params["bbox"],
            open_params["spatial_res"],
            open_params["crs"],
            open_params.get("tile_size", TILE_SIZE),
        )

    resampled_list_ds = []
    for ds in list_ds_utm:
        resampled_list_ds.append(_resample_dataset_soft(ds, target_gm))
    return mosaic_spatial_take_first(resampled_list_ds)
