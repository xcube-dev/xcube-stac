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

import warnings
from collections.abc import Sequence

import numpy as np
import pystac
import rasterio.session
import rioxarray
import shapely
import xarray as xr
from rasterio.errors import NotGeoreferencedWarning
from xcube.core.resampling import rectify_dataset
from xcube.util.jsonschema import (
    JsonBooleanSchema,
    JsonObjectSchema,
    JsonArraySchema,
    JsonStringSchema,
)

from xcube_stac.accessor import StacItemAccessor
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
from xcube_stac.utils import (
    add_nominal_datetime,
    get_gridmapping,
    get_spatial_dims,
    list_assets_from_item,
    merge_datasets,
    mosaic_spatial_take_first,
    normalize_crs,
    normalize_grid_mapping,
    rename_dataset,
    reproject_bbox,
)

_SENTINEL3_ASSETS = [
    "syn_S1N_reflectance",
    "syn_S1O_reflectance",
    "syn_S2N_reflectance",
    "syn_S2O_reflectance",
    "syn_S3N_reflectance",
    "syn_S3O_reflectance",
    "syn_S5N_reflectance",
    "syn_S5O_reflectance",
    "syn_S6N_reflectance",
    "syn_S6O_reflectance",
    "syn_Oa01_reflectance",
    "syn_Oa02_reflectance",
    "syn_Oa03_reflectance",
    "syn_Oa04_reflectance",
    "syn_Oa05_reflectance",
    "syn_Oa06_reflectance",
    "syn_Oa07_reflectance",
    "syn_Oa08_reflectance",
    "syn_Oa09_reflectance",
    "syn_Oa10_reflectance",
    "syn_Oa11_reflectance",
    "syn_Oa12_reflectance",
    "syn_Oa16_reflectance",
    "syn_Oa17_reflectance",
    "syn_Oa18_reflectance",
    "syn_Oa21_reflectance",
]
_SCHEMA_APPLY_RECTIFICATION = JsonBooleanSchema(
    title="Apply rectification algorithm.",
    description="If True, data is presented on a regular grid.",
    default=True,
)
_SCHEMA_ASSET_NAMES = JsonArraySchema(
    items=(JsonStringSchema(min_length=1, enum=_SENTINEL3_ASSETS)),
    unique_items=True,
    title="Names of assets (spectral bands).",
)
_SCHEMA_ADD_ERROR_BANDS = JsonBooleanSchema(
    title="Add error bands.",
    description="If True, error datasets for each band is added.",
    default=True,
)


class Sen3CdseStacItemAccessor(StacItemAccessor):
    """Provides methods for accessing the data of a CDSE Sentinel-3 STAC Item."""

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        self._storage_option_s3 = storage_options_s3
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

        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    def open_asset(self, asset: pystac.Asset, **open_params) -> xr.Dataset:
        ds = rioxarray.open_rasterio(
            asset.href,
            chunks=dict(band=1, y=1023, x=1217),
            band_as_variable=True,
            driver="netCDF",
        )
        ds = ds.squeeze()
        ds = ds.drop_vars(["band", "x", "y", "spatial_ref"])
        if asset.href.endswith("geolocation.nc"):
            ds_final = xr.Dataset()
            ds_final.attrs = ds.attrs
            for var_name in ["lon", "lat"]:
                array = ds[var_name]
                array = array.where(array != array.attrs["_FillValue"])
                array *= array.attrs["scale_factor"]
                ds_final[var_name] = array
            return ds_final
        var_names = list(ds.data_vars)
        if not open_params.get("add_error_bands", True):
            var_names = var_names[:1]
        ds_final = xr.Dataset()
        ds_final.attrs = ds.attrs
        for var_name in var_names:
            array = ds[var_name]
            array = array.where(array != array.attrs["_FillValue"])
            array *= array.attrs["scale_factor"]
            ds_final[var_name] = array

        return ds_final

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        coords = dict()
        ds = self.open_asset(item.assets["geolocation"])
        coords["lon"] = ds["lon"]
        coords["lat"] = ds["lat"]
        ds_item = xr.Dataset(coords=coords, attrs=ds.attrs)
        asset_names = open_params.get("asset_names", _SENTINEL3_ASSETS)
        assets = list_assets_from_item(item, asset_names=asset_names)
        for asset in assets:
            ds = self.open_asset(asset, **open_params)
            ds_item.update(ds)

        if open_params.get("apply_rectification", True):
            ds_item = rectify_dataset(ds_item)
            # TODO: add georeferencing
            # ds_item = normalize_grid_mapping(ds_item)
        return ds_item

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                asset_names=_SCHEMA_ASSET_NAMES,
                apply_rectification=_SCHEMA_APPLY_RECTIFICATION,
                add_error_bands=_SCHEMA_ADD_ERROR_BANDS,
            ),
            required=[],
            additional_properties=True,
        )


class Sen3CdseStacArdcAccessor(Sen3CdseStacItemAccessor):
    """Provides methods for access multiple Sentinel-3 STAC Items from the
    CDSE STAC API and build an analysis ready data cube."""

    def open_ardc(
        self,
        items: Sequence[pystac.Item],
        **open_params,
    ) -> xr.Dataset:

        # filter items by checking if bounding box and polygon of tiles overlap
        items = _filter_items_spatial(items, open_params["bbox"])

        # get STAC assets grouped by solar day
        grouped_items = _group_items(items)

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
                add_error_bands=_SCHEMA_ADD_ERROR_BANDS,
            ),
            required=["time_range", "bbox", "spatial_res", "crs"],
            additional_properties=False,
        )

    def _generate_cube(self, grouped_items: xr.DataArray, **open_params) -> xr.Dataset:
        target_gm = get_gridmapping(
            open_params["bbox"],
            open_params["spatial_res"],
            open_params["crs"],
            TILE_SIZE,
        )
        bbox_latlon = reproject_bbox(
            open_params["bbox"], open_params["crs"], "EPSG:4326"
        )
        dss_time = []
        for dt_idx, dt in enumerate(grouped_items.time.values):
            items = grouped_items.sel(time=dt).item()
            dss_spatial = []
            for item in items:
                ds = self.open_item(
                    item,
                    asset_names=open_params.get("asset_names"),
                    apply_rectification=False,
                )
                ds["lat"] = ds.lat.compute()
                ds["lon"] = ds.lon.compute()
                ds = _clip_sen3_dataset(ds, bbox_latlon)
                if any(size == 0 for size in ds.sizes.values()):
                    continue
                ds = rectify_dataset(ds, target_gm=target_gm)
                if ds is None:
                    continue
                dss_spatial.append(ds)
            if not dss_spatial:
                continue
            dss_time.append(mosaic_spatial_take_first(dss_spatial))
        ds_final = xr.concat(dss_time, dim="time", join="exact")
        ds_final = ds_final.assign_coords(dict(time=grouped_items.time))
        # TODO: add geo-referencing
        # ds_final = normalize_grid_mapping(ds_final)
        return ds_final


def _group_items(items: Sequence[pystac.Item]) -> xr.DataArray:
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
        dts.append(item.datetime.replace(tzinfo=None))
    grouped_items = grouped_items.assign_coords(
        time=np.array(dts, dtype="datetime64[ns]")
    )
    grouped_items["time"].encoding["units"] = "seconds since 1970-01-01"
    grouped_items["time"].encoding["calendar"] = "standard"

    return grouped_items


def _clip_sen3_dataset(ds: xr.Dataset, bbox: Sequence[float | int]):
    mask = (
        (ds.lon >= bbox[0])
        & (ds.lon <= bbox[2])
        & (ds.lat >= bbox[1])
        & (ds.lat <= bbox[3])
    )

    # Find indices where mask is True
    indices = np.where(mask)

    if len(indices[0]) == 0:
        return ds

    # Now compute the minimal slice to clip
    i_min, i_max = indices[0].min(), indices[0].max()
    j_min, j_max = indices[1].min(), indices[1].max()

    return ds.isel(
        x=slice(j_min, j_max + 1),
        y=slice(i_min, i_max + 1),
    )


def _filter_items_spatial(
    items: Sequence[pystac.Item], bbox: Sequence[float | int]
) -> Sequence[pystac.Item]:
    bbox_geom = shapely.box(*bbox)
    sel_items = []
    for item in items:
        polygon_geom = shapely.polygons(item.geometry["coordinates"])
        if bbox_geom.intersects(polygon_geom):
            sel_items.append(item)
    return sel_items
