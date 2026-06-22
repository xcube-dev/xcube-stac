# The MIT License (MIT)
# Copyright (c) 2024-2026 by the xcube development team and contributors
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
from collections.abc import Sequence

import numpy as np
import planetary_computer
import pyproj
import pystac
import rioxarray
import xarray as xr
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonNumberSchema,
    JsonObjectSchema,
    JsonStringSchema,
)
from xcube_resampling import resample_in_space
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.utils import reproject_bbox, resolution_meters_to_degrees

from xcube_stac.accessor import StacArdcAccessor, StacItemAccessor
from xcube_stac.constants import (
    SCHEMA_ADDITIONAL_QUERY,
    SCHEMA_APPLY_SCALING,
    SCHEMA_CRS,
    SCHEMA_SPATIAL_RES,
    SCHEMA_TILE_SIZE,
    SCHEMA_TIME_RANGE,
)
from xcube_stac.utils import (
    add_nominal_datetime,
    mosaic_spatial_take_first,
    rename_dataset,
)
from xcube_stac.version import version

_SENTINEL2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
    "SAA",
    "SZA",
    "VAA",
    "VZA",
    "Fmask",
]
_LANDSAT_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B09",
    "B10",
    "B11",
    "SAA",
    "SZA",
    "VAA",
    "VZA",
    "Fmask",
]
_CHUNK_SIZE = dict(x=2048, y=2048)
_SPATIAL_RES = 30  # meters

_SCHEMA_APPLY_SCALING_HLS = SCHEMA_APPLY_SCALING
_SCHEMA_APPLY_SCALING_HLS.default = True

_SCHEMA_ASSET_NAMES_LANDSAT = JsonArraySchema(
    items=(JsonStringSchema(min_length=1, enum=_LANDSAT_BANDS)),
    unique_items=True,
    title="Names of assets (spectral bands)",
)
_SCHEMA_BBOX = JsonArraySchema(
    items=(
        JsonNumberSchema(),
        JsonNumberSchema(),
        JsonNumberSchema(),
        JsonNumberSchema(),
    ),
    title="Bounding box [x1,y1,x2,y2] in coordinates of the given CRS.",
)


class Sen2HlsStacItemAccessor(StacItemAccessor):
    """Provides methods for accessing the data of a Planetary Computer
    Harmonized Landsat Sentinel-2 (HLS) Version 2.0 Sentinel-2 STAC Item.
    """

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        self._asset_names_default = _SENTINEL2_BANDS
        self._schema_asset_names = JsonArraySchema(
            items=(JsonStringSchema(min_length=1, enum=self._asset_names_default)),
            unique_items=True,
            title="Names of assets (spectral bands)",
        )

    @staticmethod
    # noinspection PyUnusedLocal
    def open_asset(asset: pystac.Asset, **open_params) -> xr.Dataset:
        return xr.Dataset(
            {
                "band_1": rioxarray.open_rasterio(
                    asset.href, chunks=_CHUNK_SIZE
                ).squeeze(drop=True)
            },
        )

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        if not self._is_pc_signed(item):
            item = planetary_computer.sign_item(item)
        apply_scaling = open_params.pop("apply_scaling", True)
        assets = self._list_assets_from_item(item, **open_params)
        dss = [self.open_asset(asset, **open_params) for asset in assets]
        ds = self._combiner_function(
            dss,
            item=item,
            assets=assets,
            apply_scaling=apply_scaling,
            **open_params,
        )
        ds.attrs = dict(
            stac_catalog_url=self._catalog.get_self_href(),
            stac_item_id=item.id,
            xcube_stac_version=version,
        )
        return ds

    @staticmethod
    def _is_pc_signed(item: pystac.Item) -> bool:
        for asset in item.assets.values():
            if "sig=" in asset.href or "sv=" in asset.href:
                return True
        return False

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                asset_names=self._schema_asset_names,
                apply_scaling=SCHEMA_APPLY_SCALING,
                crs=SCHEMA_CRS,
                bbox=_SCHEMA_BBOX,
                spatial_res=SCHEMA_SPATIAL_RES,
                tile_size=SCHEMA_TILE_SIZE,
            ),
            required=[],
            additional_properties=False,
        )

    def _list_assets_from_item(
        self, item: pystac.Item, **open_params
    ) -> list[pystac.Asset]:
        """Select and return a list of assets from a STAC item based on specified
        asset names.

        Args:
            item: The STAC item containing the assets to filter.
            **open_params: Optional parameters to control asset selection:
                - asset_names (list[str], optional): List of desired asset keys.
                    Defaults to `self._asset_names_default`.

        Returns:
            Filtered list of assets matching the requested names
        """
        asset_names = open_params.get("asset_names", self._asset_names_default)
        assets_sel = []
        for asset_name in asset_names:
            asset = item.assets[asset_name]
            asset.title = asset_name
            assets_sel.append(asset)
        return assets_sel

    def _combiner_function(
        self,
        dss: Sequence[xr.Dataset],
        assets: Sequence[pystac.Asset] = None,
        apply_scaling: bool = True,
        **open_params,
    ) -> xr.Dataset:
        dss = [rename_dataset(ds, asset.title) for (ds, asset) in zip(dss, assets)]
        if apply_scaling:
            dss = [(self._apply_offset_scaling(ds)) for ds, asset in zip(dss, assets)]
        ds = dss[0].copy()
        for ds_asset in dss[1:]:
            ds.update(ds_asset)

        # resample dataset if requested
        crs = open_params.get("crs")
        bbox = open_params.get("bbox")
        spatial_res = open_params.get("spatial_res")
        tile_size = open_params.get("tile_size", _CHUNK_SIZE.values())
        if crs is None and bbox is None and spatial_res is None:
            return ds

        source_gm = GridMapping.from_dataset(ds)
        if bbox is None:
            if crs:
                bbox = reproject_bbox(source_gm.xy_bbox, source_gm.crs, crs)
            else:
                bbox = source_gm.xy_bbox
        if spatial_res is None:
            if crs and crs.is_geographic:
                center_lat = (source_gm.xy_bbox[1] + source_gm.xy_bbox[3]) / 2
                spatial_res = resolution_meters_to_degrees(source_gm.xy_res, center_lat)
            else:
                spatial_res = source_gm.xy_res
        if crs is None:
            crs = source_gm.crs
        target_gm = GridMapping.regular_from_bbox(
            bbox=bbox, xy_res=spatial_res, crs=crs, tile_size=tile_size
        )
        return resample_in_space(
            ds, source_gm=source_gm, target_gm=target_gm, prevent_nan_propagations=True
        )

    @staticmethod
    def _apply_offset_scaling(ds: xr.Dataset) -> xr.Dataset:
        var = list(ds.keys())[0]
        if var == "Fmask":
            return ds
        attrs = ds[var].attrs
        ds[var] = ds[var].where(ds[var] != attrs["_FillValue"])
        ds[var] *= attrs["scale_factor"]
        ds[var] += attrs["add_offset"]
        return ds


class LandsatHlsStacItemAccessor(Sen2HlsStacItemAccessor):
    """Provides methods for accessing the data of a Planetary Computer
    Harmonized Landsat Sentinel-2 (HLS) Version 2.0 Landsat STAC Item.
    """

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        super().__init__(catalog, **storage_options_s3)
        self._asset_names_default = _LANDSAT_BANDS
        self._schema_asset_names = JsonArraySchema(
            items=(JsonStringSchema(min_length=1, enum=self._asset_names_default)),
            unique_items=True,
            title="Names of assets (spectral bands)",
        )


class Sen2HlsStacArdcAccessor(Sen2HlsStacItemAccessor, StacArdcAccessor):
    """Provides utilities to retrieve multiple Sentinel-2 STAC items from the
    Planetary Computer STAC API for the Harmonized Landsat Sentinel-2 (HLS) Collection
    Version 2.0, and to assemble them into an analysis-ready data cube.
    """

    def open_ardc(
        self,
        items: Sequence[pystac.Item],
        **open_params,
    ) -> xr.Dataset:

        for item in items:
            print(item.id, item.datetime, item.bbox, item.properties["proj:code"])

        items = fix_utm_hemisphere(items)

        for item in items:
            print(item.id, item.datetime, item.bbox, item.properties["proj:code"])

        # get STAC assets grouped by solar day
        grouped_items = self._group_items(items)

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
            title="Open parameters to open via a user-defined bounding box.",
            properties=dict(
                asset_names=self._schema_asset_names,
                time_range=SCHEMA_TIME_RANGE,
                bbox=_SCHEMA_BBOX,
                spatial_res=SCHEMA_SPATIAL_RES,
                crs=SCHEMA_CRS,
                query=SCHEMA_ADDITIONAL_QUERY,
                apply_scaling=_SCHEMA_APPLY_SCALING_HLS,
                tile_size=SCHEMA_TILE_SIZE,
            ),
            required=["time_range", "bbox", "spatial_res", "crs"],
            additional_properties=False,
        )

    def _generate_cube(self, grouped_items: xr.DataArray, **open_params) -> xr.Dataset:
        """Generate a spatiotemporal data cube from multiple items.

         This function takes grouped items parameters and generates a unified xarray
         dataset, mosaicking and stacking the data across spatial tiles and time.
         Optionally, scaling is applied and solar and viewing angles are calculated.

         Args:
             grouped_items: A 2D data array indexed by tile_id and time contianing STAC
                 items.
             **open_params: Optional keyword arguments for data opening and processing.

        Returns:
             A dataset representing the spatiotemporal cube.
        """
        # Group the tile IDs by UTM zones
        utm_tile_id = defaultdict(list)
        for tile_id in grouped_items.tile_id.values:
            item = np.sum(grouped_items.sel(tile_id=tile_id).values)[0]
            crs = item.properties["proj:code"]
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

        return ds_final

    @staticmethod
    def _group_items(items: Sequence[pystac.Item]) -> xr.DataArray:
        """Group STAC items by solar day, tile ID.

        This method organizes a list of Sentinel-2 STAC items into an `xarray.DataArray`
        with dimensions `(time, tile_id)`, where:
        - `time` corresponds to the solar acquisition date (ignoring time of day),
        - `tile_id` is the Sentinel-2 MGRS tile code,
        - The most recent processing version is selected if multiple exist.

        Args:
            items: List of STAC items to group. Each item must have:
                - Nominal acquisition datetime.
                - Tile ID (MGRS grid).
                - processing version

        Returns:
            A 2D DataArray of shape (time, tile_id) containing STAC items.
            Time coordinate values are datetimes, derived from the UTC datetime of the
            first item per date.
        """
        items = add_nominal_datetime(items)

        # get dates and tile IDs of the items
        dates = []
        tile_ids = []
        for item in items:
            dates.append(item.properties["datetime_nominal"].date())
            tile_ids.append(item.id.split(".")[2][1:])
        dates = np.unique(dates)
        tile_ids = np.unique(tile_ids)

        # sort items by date and tile ID into a data array
        grouped_items = np.full((len(dates), len(tile_ids)), None, dtype=object)
        for idx, item in enumerate(items):
            date = item.properties["datetime_nominal"].date()
            tile_id = item.id.split(".")[2][1:]
            idx_date = np.where(dates == date)[0][0]
            idx_tile_id = np.where(tile_ids == tile_id)[0][0]
            if grouped_items[idx_date, idx_tile_id] is None:
                grouped_items[idx_date, idx_tile_id] = [item]
            else:
                grouped_items[idx_date, idx_tile_id].append(item)

        for idx_date in range(grouped_items.shape[0]):
            for idx_tile_id in range(grouped_items.shape[1]):
                if grouped_items[idx_date, idx_tile_id] is None:
                    grouped_items[idx_date, idx_tile_id] = []
        grouped_items = xr.DataArray(
            grouped_items,
            dims=("time", "tile_id"),
            coords=dict(time=dates, tile_id=tile_ids),
        )

        # replace date by datetime from first item
        dts = []
        for date in grouped_items.time.values:
            item = np.sum(grouped_items.sel(time=date).values)[0]
            dts.append(item.datetime.replace(tzinfo=None))
        grouped_items = grouped_items.assign_coords(
            time=np.array(dts, dtype="datetime64[ns]")
        )
        grouped_items = grouped_items.assign_coords(time=dts)
        grouped_items["time"].encoding["units"] = "seconds since 1970-01-01"
        grouped_items["time"].encoding["calendar"] = "standard"

        return grouped_items

    def _generate_utm_cube(
        self,
        grouped_items: xr.DataArray,
        crs_utm: str,
        **open_params,
    ) -> xr.Dataset:
        """Load and insert tile data for a specific UTM zone into a unified dataset.

        This method processes the assets from selected STAC items within a particular
        UTM zone. It opens the data for each asset and time step, clips it to the
        target bounding box (reprojected to the UTM CRS), mosaics overlapping spatial
        datasets by taking the first valid pixel, and inserts the processed data into
        an aggregated dataset for that UTM zone.

        Args:
            grouped_items: An xarray DataArray containing STAC items, indexed by
            tile_id and time.
            crs_utm: The target UTM coordinate reference system identifier.
            **open_params: Additional parameters to control data opening and processing,
                including 'bbox' (bounding box) and 'crs' (coordinate reference system).

        Returns:
            A dataset containing mosaicked and stacked 3d datacubes
            (time, y, x) for all assets within the specified UTM zone.
        """
        final_bbox = reproject_bbox(open_params["bbox"], open_params["crs"], crs_utm)
        open_item_open_params = dict(
            asset_names=open_params.get("asset_names"),
            apply_scaling=open_params.get("apply_scaling", True),
        )

        var_names = open_params.get("asset_names", ["B01"])
        fill_value = np.nan
        var_ref = var_names[0]
        if var_names[0] == "Fmask":
            if len(var_names) == 1:
                fill_value = 0
            else:
                var_ref = var_names[1]
        dss = []
        idxs_dt = []
        for dt_idx, dt in enumerate(grouped_items.time.values):
            dss_dt = []
            for tile_id in grouped_items.tile_id.values:
                items = grouped_items.sel(tile_id=tile_id, time=dt).item()
                multi_tiles = []
                for item in items:
                    ds = self.open_item(item, **open_item_open_params)
                    multi_tiles.append(ds)
                if not multi_tiles:
                    continue
                dss_dt.append(
                    mosaic_spatial_take_first(multi_tiles, var_ref, fill_value)
                )
            if not dss_dt:
                continue
            dss.append(
                xr.merge(
                    dss_dt, compat="override", join="outer", fill_value={"Fmask": 255}
                )
            )
            idxs_dt.append(dt_idx)
        ds_final = xr.concat(dss, dim="time", join="outer", fill_value={"Fmask": 255})
        ds_final = ds_final.assign_coords(dict(time=grouped_items.time[idxs_dt]))
        ds_final = ds_final.sortby("y", ascending=False)
        ds_final = ds_final.sel(
            x=slice(final_bbox[0], final_bbox[2]),
            y=slice(final_bbox[3], final_bbox[1]),
        )

        return ds_final


class LandsatHlsStacArdcAccessor(LandsatHlsStacItemAccessor, Sen2HlsStacArdcAccessor):
    """Provides utilities to retrieve multiple Landsat STAC items from the
    Planetary Computer STAC API for the Harmonized Landsat Sentinel-2 (HLS) Collection
    Version 2.0, and to assemble them into an analysis-ready data cube.
    """


def fix_utm_hemisphere(items: Sequence[pystac.Item]) -> Sequence[pystac.Item]:
    """
    Correct STAC proj:code UTM hemisphere based on item bbox.

    If bbox center latitude >= 0 -> use EPSG:326xx (UTM North)
    If bbox center latitude < 0  -> use EPSG:327xx (UTM South)

    Keeps the UTM zone and only fixes the hemisphere.

    Parameters:
        items: Sequence containing items

    Returns:
        Sequence containing corrected items
    """
    for item in items:
        bbox = item.bbox
        minx, miny, maxx, maxy = bbox
        center_lat = (miny + maxy) / 2

        proj_code = item.properties.get("proj:code")
        epsg = int(proj_code.split(":")[1])

        # extract UTM zone (last two digits)
        zone = epsg % 100

        if center_lat >= 0:
            correct_epsg = f"EPSG:{32600 + zone}"
        else:
            correct_epsg = f"EPSG:{32700 + zone}"

        if proj_code != correct_epsg:
            item.properties["proj:code"] = correct_epsg

    return items


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
            target_gm = GridMapping.regular_from_bbox(
                open_params["bbox"],
                open_params["spatial_res"],
                open_params["crs"],
                tile_size=open_params.get("tile_size", _CHUNK_SIZE.values()),
            )
    else:
        target_gm = GridMapping.regular_from_bbox(
            open_params["bbox"],
            open_params["spatial_res"],
            open_params["crs"],
            tile_size=open_params.get("tile_size", _CHUNK_SIZE.values()),
        )

    resampled_list_ds = []
    for ds in list_ds_utm:
        resampled_list_ds.append(
            resample_in_space(ds, target_gm=target_gm, prevent_nan_propagations=True)
        )

    var_names = list(resampled_list_ds[0].keys())
    var_ref = var_names[0]
    fill_value = np.nan
    if var_names[0] == "Fmask":
        if len(var_names) == 1:
            fill_value = 0
        else:
            var_ref = var_names[1]

    ds_final = mosaic_spatial_take_first(resampled_list_ds, var_ref, fill_value)
    x_dim, y_dim = target_gm.xy_var_names
    ds_final = ds_final.chunk(
        {x_dim: _CHUNK_SIZE["x"], y_dim: _CHUNK_SIZE["y"], "time": 1}
    )
    return ds_final
