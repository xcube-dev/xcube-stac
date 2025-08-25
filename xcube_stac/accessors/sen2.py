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

import re
from collections import defaultdict
from collections.abc import Sequence

import boto3
import dask
import dask.array as da
import numpy as np
import planetary_computer
import pyproj
import pystac
import rasterio.session
import requests
import rioxarray
import xarray as xr
import xmltodict
from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import affine_transform_dataset, reproject_dataset
from xcube.core.store import DataStoreError
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonNumberSchema,
    JsonObjectSchema,
    JsonStringSchema,
)

from xcube_stac.accessor import StacArdcAccessor, StacItemAccessor
from xcube_stac.constants import (
    CONVERSION_FACTOR_DEG_METER,
    SCHEMA_ADDITIONAL_QUERY,
    SCHEMA_APPLY_SCALING,
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
]
_SENTINEL2_L2A_BANDS = _SENTINEL2_BANDS + ["AOT", "SCL", "WVP"]
_SENTINEL2_L2A_BANDS.remove("B10")
_SEN2_SPATIAL_RES = np.array([10, 20, 60])
_SENTINEL2_REGEX_ASSET_NAME = "^[A-Z]{3}_[0-9]{2}m$"
_SCHEMA_ANGLES_SENTINEL2 = JsonBooleanSchema(
    title="Add viewing and solar angles from Sentinel2 metadata.",
    description=(
        "Viewing and solar angles will be extracted for all spectral "
        "bands defined in keyword `asset_name`."
    ),
    default=False,
)
_SCHEMA_APPLY_SCALING_SENTINEL2 = SCHEMA_APPLY_SCALING
_SCHEMA_APPLY_SCALING_SENTINEL2.default = True
_SCHEMA_SPATIAL_RES_SEN2_ITEM = JsonNumberSchema(
    title=SCHEMA_SPATIAL_RES.title, enum=_SEN2_SPATIAL_RES, default=10
)
_SCHEMA_ASSET_NAMES = JsonArraySchema(
    items=(JsonStringSchema(min_length=1, enum=_SENTINEL2_L2A_BANDS)),
    unique_items=True,
    title="Names of assets (spectral bands)",
)


class Sen2CdseStacItemAccessor(StacItemAccessor):
    """Provides methods for accessing the data of a CDSE Sentinel-2 STAC Item."""

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        self.session = rasterio.session.AWSSession(
            aws_unsigned=storage_options_s3["anon"],
            endpoint_url=storage_options_s3["client_kwargs"]["endpoint_url"].split(
                "//"
            )[1],
            aws_access_key_id=storage_options_s3["key"],
            aws_secret_access_key=storage_options_s3["secret"],
        )
        self.env = rasterio.env.Env(session=self.session, AWS_VIRTUAL_HOSTING=False)
        # keep the rasterio environment open so that the data can be accessed
        # when plotting or writing the data
        self.env = self.env.__enter__()
        # dask multi-threading needs to be turned off, otherwise the GDAL
        # reader for JP2 raises error.
        dask.config.set(scheduler="single-threaded")
        # need boto2 client to read xml metadata remotely
        self.s3_boto = boto3.client(
            "s3",
            endpoint_url=storage_options_s3["client_kwargs"]["endpoint_url"],
            aws_access_key_id=storage_options_s3["key"],
            aws_secret_access_key=storage_options_s3["secret"],
            region_name="default",
        )
        # define field names in STAC item
        self._stac_item_properties = dict(
            tile_id="grid:code",
            crs="proj:code",
            processing_version="processing:version",
        )

    @staticmethod
    # noinspection PyUnusedLocal
    def open_asset(asset: pystac.Asset, **open_params) -> xr.Dataset:
        tile_size = open_params.get("tile_size", (1024, 1024))
        return rioxarray.open_rasterio(
            asset.href,
            chunks=dict(x=tile_size[0], y=tile_size[1]),
            band_as_variable=True,
        )

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        if not self._is_pc_signed(item):
            item = planetary_computer.sign_item(item)
        apply_scaling = open_params.pop("apply_scaling", True)
        assets = self._list_assets_from_item(item, **open_params)
        dss = [self.open_asset(asset) for asset in assets]
        ds = self._combiner_function(
            dss,
            item=item,
            assets=assets,
            apply_scaling=apply_scaling,
        )
        if open_params.get("add_angles", False):
            ds = self._add_sen2_angles(item, ds)
        ds.attrs = dict(
            stac_catalog_url=self._catalog.get_self_href(),
            stac_item_id=item.id,
            xcube_stac_version=version,
        )
        return ds

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                asset_names=_SCHEMA_ASSET_NAMES,
                apply_scaling=_SCHEMA_APPLY_SCALING_SENTINEL2,
                spatial_res=_SCHEMA_SPATIAL_RES_SEN2_ITEM,
                add_angles=_SCHEMA_ANGLES_SENTINEL2,
            ),
            required=[],
            additional_properties=False,
        )

    @staticmethod
    def _is_pc_signed(item: pystac.Item) -> bool:
        for asset in item.assets.values():
            if "sig=" in asset.href or "sv=" in asset.href:
                return True
        return False

    @staticmethod
    def _list_assets_from_item(item: pystac.Item, **open_params) -> list[pystac.Asset]:
        """Select and return a list of assets from a STAC item based on specified
        asset names and spatial resolution.

        If no asset names are provided, a default set is used. The method attempts to
        match asset names exactly; if an exact match is not found, it tries to append
        a spatial resolution suffix (e.g., "_10m") based on the closest available
        resolution to the requested spatial resolution.

        Args:
            item: The STAC item containing the assets to filter.
            **open_params: Optional parameters to control asset selection:
                - asset_names (list[str], optional): List of desired asset keys.
                    Defaults to SENTINEL2_L2A_BANDS.
                - spatial_res (int, optional): Desired spatial resolution in meters
                  defining the asset selection. *spatial_res* must be one of
                  [10, 20, 60]. Defaults to 10.

        Returns:
            Filtered list of assets matching the requested names and spatial resolution.
            Each asset's extra_fields is augmented with:
                - 'xcube:asset_id': the selected asset name (including spatial resolution
                    suffix if applied).
                - 'xcube:asset_id_origin': the original requested asset name.
        """
        asset_names = open_params.get("asset_names")
        if item.collection_id == "sentinel-2-l2a":
            if not asset_names:
                asset_names = _SENTINEL2_L2A_BANDS
            spatial_res_final = open_params.get("spatial_res", 10)
            assets_sel = []
            for asset_name in asset_names:
                asset_name_res = asset_name
                if not re.fullmatch(_SENTINEL2_REGEX_ASSET_NAME, asset_name):
                    res_diff = abs(spatial_res_final - _SEN2_SPATIAL_RES)
                    for spatial_res in _SEN2_SPATIAL_RES[np.argsort(res_diff)]:
                        asset_name_res = f"{asset_name}_{spatial_res}m"
                        if asset_name_res in item.assets:
                            break
                asset = item.assets[asset_name_res]
                asset.extra_fields["xcube:asset_id"] = asset_name_res
                asset.extra_fields["xcube:asset_id_origin"] = asset_name
                assets_sel.append(asset)
        elif item.collection_id == "sentinel-2-l1c":
            if not asset_names:
                asset_names = _SENTINEL2_BANDS
            assets_sel = []
            for asset_name in asset_names:
                asset = item.assets[asset_name]
                asset.extra_fields["xcube:asset_id"] = asset_name
                asset.extra_fields["xcube:asset_id_origin"] = asset_name
                assets_sel.append(asset)
        else:
            raise DataStoreError(
                "Only collections 'sentinel-2-l2a' and 'sentinel-2-l1c' are supported."
            )
        return assets_sel

    @staticmethod
    def _combiner_function(
        dss: Sequence[xr.Dataset],
        item: pystac.Item = None,
        assets: Sequence[pystac.Asset] = None,
        apply_scaling: bool = True,
    ) -> xr.Dataset:
        dss = [
            rename_dataset(ds, asset.extra_fields["xcube:asset_id_origin"])
            for (ds, asset) in zip(dss, assets)
        ]
        if apply_scaling:
            raster_version = get_stac_extension(item)
            dss = [
                apply_offset_scaling(ds, asset, raster_version)
                for (ds, asset) in zip(dss, assets)
            ]
        ds = merge_datasets(dss)
        return normalize_grid_mapping(ds)

    def _add_sen2_angles(self, item: pystac.Item, ds: xr.Dataset) -> xr.Dataset:
        """Extract Sentinel-2 solar and viewing angle information from the granule
        metadata and add it to the dataset `ds`.

        This method parses the granule metadata XML file associated with
        the given STAC item, then extracts solar and viewing angle information for
        the relevant bands.

        Args:
            item: A STAC item containing a 'granule_metadata' asset with the
                XML metadata.
            ds: A dataset containing Sentinel-2 data bands, used to determine which
                bands' angles to extract.

        Returns:
            An updated version of `ds`, with solar and viewing angle data added
            corresponding to the bands.
        """
        target_gm = _get_angle_target_gm(ds)
        ds_angles = self.get_sen2_angles(item, ds)
        ds_angles = _resample_dataset_soft(
            ds_angles,
            target_gm,
            interpolation="bilinear",
        )
        ds = _add_angles(ds, ds_angles)
        return ds

    def get_sen2_angles(self, item: pystac.Item, ds: xr.Dataset) -> xr.Dataset:
        # read xml file and parse to dict
        href = item.assets["granule_metadata"].href
        protocol, remain = href.split("://")
        root = "eodata"
        fs_path = "/".join(remain.split("/")[1:])
        response = self.s3_boto.get_object(Bucket=root, Key=fs_path)
        xml_content = response["Body"].read().decode("utf-8")
        xml_dict = xmltodict.parse(xml_content)

        # read angles from xml file and add to dataset
        band_names = _get_band_names_from_dataset(ds)
        ds_angles = _get_sen2_angles(xml_dict, band_names)

        return ds_angles


class Sen2CdseStacArdcAccessor(Sen2CdseStacItemAccessor, StacArdcAccessor):
    """Provides methods to access multiple Sentinel-2 STAC Items from the
    CDSE STAC API and build an analysis ready data cube."""

    def open_ardc(
        self,
        items: Sequence[pystac.Item],
        **open_params,
    ) -> xr.Dataset:

        # Remove items with incorrect bounding boxes in the CDSE Sentinel-2 L2A catalog.
        # This issue primarily affects tiles that cross the antimeridian and has been
        # reported as a catalog bug. A single Sentinel-2 tile spans approximately
        # 110 km in width. Near the poles (up to 83°N), this corresponds to a bounding
        # box width of about 8°. To account for inaccuracies, we use a conservative
        # threshold of 20° to detect and exclude faulty items.
        items = [item for item in items if abs(item.bbox[2] - item.bbox[0]) < 20]

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
            properties=dict(
                asset_names=_SCHEMA_ASSET_NAMES,
                time_range=SCHEMA_TIME_RANGE,
                bbox=SCHEMA_BBOX,
                spatial_res=SCHEMA_SPATIAL_RES,
                crs=SCHEMA_CRS,
                query=SCHEMA_ADDITIONAL_QUERY,
                add_angles=_SCHEMA_ANGLES_SENTINEL2,
                apply_scaling=_SCHEMA_APPLY_SCALING_SENTINEL2,
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
            asset = next(iter(item.assets.values()))
            crs = asset.extra_fields.get(
                self._stac_item_properties["crs"],
                item.properties.get(self._stac_item_properties["crs"]),
            )
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

    def _group_items(self, items: Sequence[pystac.Item]) -> xr.DataArray:
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
        proc_versions = []
        for item in items:
            dates.append(item.properties["datetime_nominal"].date())
            tile_ids.append(item.properties.get(self._stac_item_properties["tile_id"]))
            proc_versions.append(
                item.properties.get(self._stac_item_properties["processing_version"])
            )
        dates = np.unique(dates)
        tile_ids = np.unique(tile_ids)
        proc_versions = np.unique(proc_versions)[::-1]

        # sort items by date and tile ID into a data array
        grouped_items = np.full(
            (len(dates), len(tile_ids), len(proc_versions)), None, dtype=object
        )
        for idx, item in enumerate(items):
            date = item.properties["datetime_nominal"].date()
            tile_id = item.properties.get(self._stac_item_properties["tile_id"])
            proc_version = item.properties.get(
                self._stac_item_properties["processing_version"]
            )
            idx_date = np.where(dates == date)[0][0]
            idx_tile_id = np.where(tile_ids == tile_id)[0][0]
            idx_proc_version = np.where(proc_versions == proc_version)[0][0]
            if grouped_items[idx_date, idx_tile_id, idx_proc_version] is None:
                grouped_items[idx_date, idx_tile_id, idx_proc_version] = [item]
            else:
                grouped_items[idx_date, idx_tile_id, idx_proc_version].append(item)

        # take the latest processing version
        # noinspection PyComparisonWithNone
        mask = grouped_items != None
        proc_version_idx = np.argmax(mask, axis=-1)
        grouped_items = np.take_along_axis(
            grouped_items, proc_version_idx[..., np.newaxis], axis=-1
        )[..., 0]
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
        items_bbox = _get_bounding_box(grouped_items)
        final_bbox = reproject_bbox(open_params["bbox"], open_params["crs"], crs_utm)
        spatial_res = _get_spatial_res(open_params)
        open_item_open_params = dict(
            asset_names=open_params.get("asset_names"),
            spatial_res=spatial_res,
            apply_scaling=open_params.get("apply_scaling", True),
            add_angles=False,
        )
        final_ds = None
        for dt_idx, dt in enumerate(grouped_items.time.values):
            for tile_id in grouped_items.tile_id.values:
                items = grouped_items.sel(tile_id=tile_id, time=dt).item()
                multi_tiles = []
                for item in items:
                    ds = self.open_item(item, **open_item_open_params)
                    ds = ds.sel(
                        x=slice(final_bbox[0], final_bbox[2]),
                        y=slice(final_bbox[3], final_bbox[1]),
                    )
                    if any(size == 0 for size in ds.sizes.values()):
                        continue
                    multi_tiles.append(ds)
                if not multi_tiles:
                    continue
                mosaicked_ds = mosaic_spatial_take_first(multi_tiles)
                if final_ds is None:
                    final_ds = _create_empty_dataset(
                        mosaicked_ds, grouped_items, items_bbox, final_bbox, spatial_res
                    )
                final_ds = _insert_tile_data(final_ds, mosaicked_ds, dt_idx)

        return final_ds

    def add_sen2_angles_stack(
        self,
        ds_final: xr.Dataset,
        grouped_items: xr.DataArray,
    ) -> xr.Dataset:
        """Add Sentinel-2 solar and viewing angle information from multiple STAC items
        as a mosaicked and stacked dataset.

        This method processes angle metadata for each tile from the provided
        `access_params`. It retrieves angle datasets for each tile,
        mosaics them spatially, resamples to match the target grid mapping, and
        finally adds the combined angle data to the input dataset `ds_final`.

        Args:
            ds_final: The main dataset containing Sentinel-2 data to which angle data
                will be added.
            grouped_items: A DataArray containing STAC items.

        Returns:
            An updated `ds_final` which includes solar and viewing angle information
            stacked and mosaicked, so that it aligns with the Sentinel-2 spectral data.
        """
        target_gm = _get_angle_target_gm(ds_final)
        list_ds_tiles = []
        for tile_id in grouped_items.tile_id.values:
            list_ds_time = []
            idx_remove_dt = []
            for dt_idx, dt in enumerate(grouped_items.time.values):
                items = grouped_items.sel(tile_id=tile_id, time=dt).item()
                multi_tiles = []
                for item in items:
                    ds = self.get_sen2_angles(item, ds_final)
                    multi_tiles.append(ds)
                if not multi_tiles:
                    idx_remove_dt.append(dt_idx)
                    continue
                else:
                    ds_time = mosaic_spatial_take_first(multi_tiles)
                    list_ds_time.append(ds_time)
            ds_tile = xr.concat(list_ds_time, dim="time", join="exact").chunk(-1)
            np_datetimes_sel = [
                value
                for idx, value in enumerate(grouped_items.time.values)
                if idx not in idx_remove_dt
            ]
            ds_tile = ds_tile.assign_coords(coords=dict(time=np_datetimes_sel))
            ds_tile = _resample_dataset_soft(
                ds_tile,
                target_gm,
                interpolation="bilinear",
            )
            ds_tile = ds_tile.chunk(dict(time=1))
            if len(idx_remove_dt) > 0:
                ds_tile = _fill_nan_slices(
                    ds_tile, grouped_items.time.values, idx_remove_dt
                )
            list_ds_tiles.append(ds_tile)

        ds_angles = mosaic_spatial_take_first(list_ds_tiles)
        ds_final = _add_angles(ds_final, ds_angles)
        return ds_final


class Sen2PlanetaryComputerStacItemAccessor(Sen2CdseStacItemAccessor):
    """Provides methods for accessing the data of a CDSE Sentinel-2 STAC Item."""

    # noinspection PyMissingConstructor
    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        # define field names in STAC items
        self._stac_item_properties = dict(
            tile_id="s2:mgrs_tile",
            crs="proj:code",
            processing_version="s2:processing_baseline",
        )

    def _combiner_function(
        self,
        dss: Sequence[xr.Dataset],
        item: pystac.Item = None,
        assets: Sequence[pystac.Asset] = None,
        apply_scaling: bool = True,
    ) -> xr.Dataset:
        dss = [
            rename_dataset(ds, asset.extra_fields["xcube:asset_id_origin"])
            for (ds, asset) in zip(dss, assets)
        ]
        if apply_scaling:
            dss = [
                self._apply_offset_scaling(ds, item) for (ds, asset) in zip(dss, assets)
            ]
        ds = merge_datasets(dss)
        return normalize_grid_mapping(ds)

    @staticmethod
    def _list_assets_from_item(item: pystac.Item, **open_params) -> list[pystac.Asset]:
        asset_names = open_params.get("asset_names")
        if not asset_names:
            asset_names = _SENTINEL2_L2A_BANDS

        assets_sel = []
        for asset_name in asset_names:
            asset = item.assets[asset_name]
            asset.extra_fields["xcube:asset_id"] = asset_name
            asset.extra_fields["xcube:asset_id_origin"] = asset_name
            assets_sel.append(asset)
        return assets_sel

    @staticmethod
    def _apply_offset_scaling(ds: xr.Dataset, item: pystac.Item) -> xr.Dataset:
        if "SCL" in ds.data_vars:
            return ds

        info = item.properties.get("xcube:offset_scaling")
        if info is None:
            response = requests.get(item.assets["product-metadata"].href)
            response.raise_for_status()
            xml_dict = xmltodict.parse(response.text)
            info = xml_dict[f"n1:Level-2A_User_Product"]["n1:General_Info"][
                "Product_Image_Characteristics"
            ]
            item.properties["xcube:offset_scaling"] = info

        if "AOT" in ds.data_vars:
            scale = float(
                info["QUANTIFICATION_VALUES_LIST"]["AOT_QUANTIFICATION_VALUE"]["#text"]
            )
            offset = 0
        elif "WVP" in ds.data_vars:
            scale = float(
                info["QUANTIFICATION_VALUES_LIST"]["WVP_QUANTIFICATION_VALUE"]["#text"]
            )
            offset = 0
        else:
            scale = float(
                info["QUANTIFICATION_VALUES_LIST"]["BOA_QUANTIFICATION_VALUE"]["#text"]
            )
            if "BOA_ADD_OFFSET_VALUES_LIST" in info:
                offset = int(
                    info["BOA_ADD_OFFSET_VALUES_LIST"]["BOA_ADD_OFFSET"][0]["#text"]
                )
            else:
                offset = 0
        nodata = int(
            next(
                d["SPECIAL_VALUE_INDEX"]
                for d in info["Special_Values"]
                if d["SPECIAL_VALUE_TEXT"] == "NODATA"
            )
        )

        ds = ds.where(ds != nodata)
        ds += offset
        ds /= scale

        return ds

    def get_sen2_angles(self, item: pystac.Item, ds: xr.Dataset) -> xr.Dataset:
        response = requests.get(item.assets["granule-metadata"].href)
        response.raise_for_status()
        xml_dict = xmltodict.parse(response.text)

        # read angles from xml file and add to dataset
        band_names = _get_band_names_from_dataset(ds)
        ds_angles = _get_sen2_angles(xml_dict, band_names)

        return ds_angles


class Sen2PlanetaryComputerStacArdcAccessor(
    Sen2PlanetaryComputerStacItemAccessor, Sen2CdseStacArdcAccessor
):
    """Provides methods for accessing the data of a CDSE Sentinel-2 STAC Item."""

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        super().__init__(catalog, **storage_options_s3)


def _get_angle_target_gm(ds_final: xr.Dataset) -> GridMapping:
    """Determine the target grid mapping for angle datasets based on the input dataset.

    This function computes an appropriate bounding box and spatial resolution
    for resampling angle data to align with the spatial reference of the given
    Sentinel-2 dataset. It handles both geographic (latitude/longitude) and
    projected coordinate reference systems, adjusting resolution accordingly.

    Args:
        ds_final: The dataset whose spatial reference and coordinates define the
            target grid mapping.

    Returns:
        A GridMapping object representing the bounding box, resolution, and CRS
        suitable for resampling angle data to match `ds_final`.

    Notes:
        The native resolution of solar and viewing angle data in Sentinel-2 products
        is 5000 meters. This resolution is preserved in the resulting target grid
        mapping used during datacube generation.
    """
    crs = pyproj.CRS.from_cf(ds_final.spatial_ref.attrs)
    y_coord, x_coord = get_spatial_dims(ds_final)
    if crs.is_geographic:
        y_res = 5000 / 111320
        y_center = (ds_final[x_coord][0].item() + ds_final[x_coord][-1].item()) / 2
        x_res = 5000 / (111320 * np.cos(np.radians(y_center)))
    else:
        y_res = 5000
        x_res = 5000

    bbox = [
        ds_final[x_coord][0].item(),
        ds_final[y_coord][0].item(),
        ds_final[x_coord][-1].item(),
        ds_final[y_coord][-1].item(),
    ]
    if bbox[3] < bbox[1]:
        y_min, y_max = bbox[3], bbox[1]
        bbox[1], bbox[3] = y_min, y_max
    return get_gridmapping(bbox, (x_res, y_res), crs)


def _get_band_names_from_dataset(ds: xr.Dataset) -> list[str]:
    """Extract valid Sentinel-2 band names from the dataset variable names.

    This function scans the keys of the input dataset and collects those that
    start with a 'B' (e.g., 'B02', 'B08A'). It returns only the names that match
    known Sentinel-2 band identifiers defined in the global `SENTINEL2_BANDS` list.

    Parameters:
        ds: The input dataset containing variables named with band prefixes.

    Returns:
        A list of valid Sentinel-2 band names found in the dataset.
    """
    band_names = [
        str(key).split("_")[0] for key in ds.keys() if str(key).startswith("B")
    ]
    return [name for name in _SENTINEL2_BANDS if name in band_names]


def _get_sen2_angles(xml_dict: dict, band_names: list[str]) -> xr.Dataset:
    """Extract solar and viewing angle information from a Sentinel-2 metadata
    dictionary derived from the XML metadata file.

    This function parses geometric metadata from a Sentinel-2 L2A product XML dictionary
    to compute solar and viewing zenith/azimuth angles over a 23x23 grid of 5000 meter
    spacing (native grid of Sentinel-2 angles). Viewing angles are computed per
    detector and band, then averaged across detectors. Solar angles are included as a
    separate band.

    Parameters:
        xml_dict: Parsed XML metadata from a Sentinel-2 L2A SAFE product.
        band_names: List of Sentinel-2 band names to extract angles for.

    Returns:
        A dataset containing the following variables:
            - solar_angle_zenith, solar_angle_azimuth
            - viewing_angle_zenith_{band}, viewing_angle_azimuth_{band}
        Each variable is a 2D grid with coordinates x and y.

    Notes:
        - The grid resolution is 5000 meters, 23x23 in size.
        - Angles are averaged over detector IDs.
        - The dataset includes a spatial reference system as a coordinate.
    """
    # read out solar and viewing angles
    if "n1:Level-1C_Tile_ID" in xml_dict:
        xml_dict = xml_dict["n1:Level-1C_Tile_ID"]
    else:
        xml_dict = xml_dict["n1:Level-2A_Tile_ID"]
    geocode = xml_dict["n1:Geometric_Info"]["Tile_Geocoding"]
    ulx = float(geocode["Geoposition"][0]["ULX"])
    uly = float(geocode["Geoposition"][0]["ULY"])
    x = ulx + 5000 * np.arange(23)
    y = uly - 5000 * np.arange(23)

    angles = xml_dict["n1:Geometric_Info"]["Tile_Angles"]
    map_bandid_name = {idx: name for idx, name in enumerate(_SENTINEL2_BANDS)}
    band_names = band_names + ["solar"]
    detector_ids = np.unique(
        [
            int(angle["@detectorId"])
            for angle in angles["Viewing_Incidence_Angles_Grids"]
        ]
    )

    array = xr.DataArray(
        np.full(
            (2, len(band_names), len(detector_ids), len(x), len(y)),
            np.nan,
            dtype=np.float32,
        ),
        dims=["angle", "band", "detector_id", "y", "x"],
        coords=dict(
            angle=["Zenith", "Azimuth"],
            band=band_names,
            detector_id=detector_ids,
            x=x,
            y=y,
        ),
    )

    # Each band has multiple detectors, so we have to go through all of them
    # and save them in a list to later do a nanmean
    for detector_angles in angles["Viewing_Incidence_Angles_Grids"]:
        band_id = int(detector_angles["@bandId"])
        band_name = map_bandid_name[band_id]
        if band_name not in band_names:
            continue
        detector_id = int(detector_angles["@detectorId"])
        for angle in array.angle.values:
            array.loc[angle, band_name, detector_id] = _get_angle_values(
                detector_angles, angle
            )
    # Do the same for the solar angles
    for angle in array.angle.values:
        array.loc[angle, "solar", detector_ids[0]] = _get_angle_values(
            angles["Sun_Angles_Grid"], angle
        )
    # Apply nanmean along detector ID axis
    array = array.mean(dim="detector_id", skipna=True)
    ds = xr.Dataset()
    for angle in array.angle.values:
        ds[f"solar_angle_{angle.lower()}"] = array.sel(
            band="solar", angle=angle
        ).drop_vars(["band", "angle"])
    for band in array.band.values[:-1]:
        for angle in array.angle.values:
            ds[f"viewing_angle_{angle.lower()}_{band}"] = array.sel(
                band=band, angle=angle
            ).drop_vars(["band", "angle"])
    crs = pyproj.CRS.from_epsg(geocode["HORIZONTAL_CS_CODE"].replace("EPSG:", ""))
    ds = ds.assign_coords(dict(spatial_ref=xr.DataArray(0, attrs=crs.to_cf())))
    ds = ds.chunk()

    return ds


def _get_angle_values(values_list: dict, angle: str) -> np.ndarray:
    """Extract a 2D array of angle values from a nested Sentinel-2 metadata dictionary.

    This function reads angle values (e.g., 'Zenith' or 'Azimuth') from a structured
    dictionary, typically parsed from a Sentinel-2 XML file. The values are expected
    to be strings representing rows of space-separated numbers, which are converted
    into a NumPy array.

    Parameters:
        values_list: A nested dictionary containing angle metadata.
        angle: The key indicating which angle to extract (e.g., "Zenith", "Azimuth").

    Returns:
        A 2D NumPy array of float32 values representing the specified angle grid.
    """
    values = values_list[angle]["Values_List"]["VALUES"]
    array = np.array([row.split(" ") for row in values]).astype(np.float32)
    return array


def _add_angles(ds: xr.Dataset, ds_angles: xr.Dataset) -> xr.Dataset:
    """Add solar and viewing angle information to the datacube.

    This function integrates angle data (solar and viewing) from a separate angle
    dataset, generated by _get_sen2_angles, into the final datacube by combining
    relevant variables into structured DataArrays.

    Parameters:
        ds: The primary dataset containing the spectral data to which angle
            information will be added.
        ds_angles: A dataset containing solar and viewing angle variables,
            generated by _get_sen2_angles.

    Returns:
        The input dataset `ds` with two additional variables:
            - 'solar_angle': A DataArray of shape (angle, angle_y, angle_x).
            - 'viewing_angle': A DataArray of shape (angle, band, angle_y, angle_x).

    Notes:
        - Renames spatial coordinates in `ds_angles` to avoid conflicts.
        - The angle dimension contains 'zenith' and 'azimuth'
    """
    x_coord, y_coord = get_spatial_dims(ds_angles)
    ds_angles = ds_angles.rename(
        {y_coord: f"angle_{y_coord}", x_coord: f"angle_{x_coord}"}
    )
    ds["solar_angle"] = (
        ds_angles[["solar_angle_zenith", "solar_angle_azimuth"]]
        .to_dataarray(dim="angle")
        .assign_coords(angle=["zenith", "azimuth"])
    ).astype(np.float32)
    ds_temp = xr.Dataset()
    bands = [
        str(k).replace("viewing_angle_zenith_", "")
        for k in ds_angles.keys()
        if "viewing_angle_zenith" in k
    ]
    keys = [k for k in ds_angles.keys() if "viewing_angle_zenith" in k]
    ds_temp["zenith"] = (
        ds_angles[keys].to_dataarray(dim="band").assign_coords(band=bands)
    )
    keys = [k for k in ds_angles.keys() if "viewing_angle_azimuth" in k]
    ds_temp["azimuth"] = (
        ds_angles[keys].to_dataarray(dim="band").assign_coords(band=bands)
    )
    ds["viewing_angle"] = (
        ds_temp[["zenith", "azimuth"]].to_dataarray(dim="angle").astype(np.float32)
    )
    ds = ds.chunk(dict(angle=-1, band=-1))

    return ds


def _get_bounding_box(items: xr.DataArray) -> list[float | int]:
    """Compute the overall bounding box that covers all tiles in the given access
    parameters.

    Iterates through each tile in `access_params` to extract the bounding box
    from its metadata and calculates the minimum bounding rectangle encompassing all
    tiles.

    Parameters:
        items: An array containing STAC items from which the native UTM bounding box
            can be derived.

    Returns:
        A list with four elements [xmin, ymin, xmax, ymax] representing the
        bounding box that encloses all tiles.
    """
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for tile_id in items.tile_id.values:
        item = np.sum(items.sel(tile_id=tile_id).values)[0]
        asset = next(iter(item.assets.values()))
        bbox = asset.extra_fields["proj:bbox"]
        if xmin > bbox[0]:
            xmin = bbox[0]
        if ymin > bbox[1]:
            ymin = bbox[1]
        if xmax < bbox[2]:
            xmax = bbox[2]
        if ymax < bbox[3]:
            ymax = bbox[3]
    return [xmin, ymin, xmax, ymax]


def _get_spatial_res(open_params: dict) -> int:
    """Determine the appropriate Sentinel-2 spatial resolution based on the CRS.

    If the CRS is geographic (e.g., EPSG:4326), the requested spatial resolution
    (in degree) is converted to meters. The function then selects the nearest
    equal or coarser supported Sentinel-2 resolution from the native
    resolutions [10, 20, 60].

    Args:
        open_params: Dictionary of open parameters. Must include:
            - "crs": Coordinate reference system as EPSG string or identifier.
            - "spatial_res": Desired spatial resolution in meters.

    Returns:
        An integer representing the selected spatial resolution (10, 20, or 60).
        If no coarser resolution is found, defaults to 60.
    """
    crs = normalize_crs(open_params["crs"])
    if crs.is_geographic:
        spatial_res = open_params["spatial_res"] * CONVERSION_FACTOR_DEG_METER
    else:
        spatial_res = open_params["spatial_res"]
    idxs = np.where(_SEN2_SPATIAL_RES >= spatial_res)[0]
    if len(idxs) == 0:
        spatial_res = 60
    else:
        spatial_res = int(_SEN2_SPATIAL_RES[idxs[0]])

    return spatial_res


def _resample_dataset_soft(
    ds: xr.Dataset,
    target_gm: GridMapping,
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
                TILE_SIZE,
            )
    else:
        target_gm = get_gridmapping(
            open_params["bbox"],
            open_params["spatial_res"],
            open_params["crs"],
            TILE_SIZE,
        )

    resampled_list_ds = []
    for ds in list_ds_utm:
        resampled_list_ds.append(_resample_dataset_soft(ds, target_gm))
    return mosaic_spatial_take_first(resampled_list_ds)


def _fill_nan_slices(ds: xr.Dataset, times: np.array, idx_nan: list[int]) -> xr.Dataset:
    """Insert NaN-filled time slices into a dataset at specified indices along
    the time axis.

    This function takes a dataset and a list of time indices (`idx_nan`) where
    NaN-filled slices should be inserted. It constructs a dataset by
    concatenating segments of the original dataset and NaN-filled slices such that
    the resulting Dataset includes placeholders at the specified positions.

    Parameters:
        ds: The input dataset with a "time" dimension.
        times: array of datetime-like values corresponding to the full time range,
            including positions for NaNs.
        idx_nan: Indices in `times` where NaN slices should be inserted.

    Returns:
        A new dataset with NaN-filled slices inserted at the specified indices,
        maintaining alignment with `times`.
    """
    ds_nan = _create_nan_slice(ds)
    list_ds = []
    if idx_nan[0] > 0:
        list_ds.append(ds.isel(time=slice(None, idx_nan[0])))
    for i, idx in enumerate(idx_nan):
        list_ds.append(ds_nan.assign_coords(coords=dict(time=[times[idx]])))
        if i < len(idx_nan) - 1:
            list_ds.append(ds.isel(time=slice(idx - i, idx_nan[i + 1] - i - 1)))
    if idx_nan[-1] < len(times) - 1:
        list_ds.append(ds.isel(time=slice(idx_nan[-1] - (len(idx_nan) - 1), None)))
    return xr.concat(list_ds, dim="time", join="exact")


def _create_nan_slice(ds: xr.Dataset) -> xr.Dataset:
    """Create a NaN-filled slice of the input dataset for a single time step.

    This function generates a new dataset with the same structure, dimensions,
    coordinates, and attributes as the first time step of the input dataset,
    but replaces all data values with NaNs. This is useful for inserting placeholder
    time steps into time-series datasets.

    Parameters:
        ds: The input dataset with a "time" dimension.

    Returns:
        A dataset with one time step where all variable values are NaN,
        matching the shape and metadata of the original dataset.
    """
    nan_ds = xr.Dataset()
    for var_name, array in ds.data_vars.items():
        array = array.isel(time=slice(0, 1))
        nan_data = da.full(
            array.shape, np.nan, dtype=array.dtype, chunks=array.data.chunksize
        )
        nan_ds[var_name] = xr.DataArray(
            nan_data, dims=array.dims, coords=array.coords, attrs=array.attrs
        )
    return nan_ds
