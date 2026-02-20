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
import planetary_computer
import xarray as xr
from scipy.interpolate import griddata
from rasterio.errors import NotGeoreferencedWarning
from xcube_resampling import rectify_dataset
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonObjectSchema,
    JsonStringSchema,
)
from xcube_resampling.gridmapping import GridMapping

from xcube_stac.accessor import StacArdcAccessor, StacItemAccessor
from xcube_stac.constants import (
    SCHEMA_ADDITIONAL_QUERY,
    SCHEMA_BBOX,
    SCHEMA_CRS,
    SCHEMA_SPATIAL_RES,
    SCHEMA_TIME_RANGE,
    TILE_SIZE,
    MEAN_EARTH_RADIUS,
)
from xcube_stac.utils import (
    add_nominal_datetime,
    list_assets_from_item,
    mosaic_spatial_take_first,
)

_SENTINEL3_SYN_CDSE_ASSETS = [
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
_SENTINEL3_SYN_PC_ASSETS = [
    asset_name.replace("_", "-").lower() for asset_name in _SENTINEL3_SYN_CDSE_ASSETS
]
_SCHEMA_APPLY_RECTIFICATION = JsonBooleanSchema(
    title="Apply rectification algorithm.",
    description="If True, data is presented on a regular grid.",
    default=True,
)
_SCHEMA_APPLY_GEO_ORTHORECTIFICATION = JsonBooleanSchema(
    title="Apply terrain-induced parallax correction to satellite geolocation",
    description=(
        "If True, the latitude and longitude grids will be corrected "
        "for terrain-induced parallax, improving geolocation accuracy."
    ),
    default=True,
)
_SCHEMA_CDSE_ASSET_NAMES = JsonArraySchema(
    items=(JsonStringSchema(min_length=1, enum=_SENTINEL3_SYN_CDSE_ASSETS)),
    unique_items=True,
    title="Names of assets (spectral bands).",
)
_SCHEMA_PC_ASSET_NAMES = JsonArraySchema(
    items=(JsonStringSchema(min_length=1, enum=_SENTINEL3_SYN_PC_ASSETS)),
    unique_items=True,
    title="Names of assets (spectral bands).",
)
_SCHEMA_ADD_ERROR_BANDS = JsonBooleanSchema(
    title="Add error bands.",
    description="If True, the error band for each band is added.",
    default=True,
)


class Sen3CdseStacItemAccessor(StacItemAccessor):
    """Provides methods for accessing a Sentinel-3 SLSTR Level-2 Land Surface
    Temperature product via a CDSE STAC Item.
    """

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        self._asset_names = _SENTINEL3_SYN_CDSE_ASSETS
        self._asset_names_schema = _SCHEMA_CDSE_ASSET_NAMES
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
            chunks="auto",
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
            var_names = [
                var_name for var_name in var_names if not var_name.endswith("_err")
            ]
            ds = ds[var_names]
        for var_name in var_names:
            ds[var_name] = ds[var_name].where(
                ds[var_name] != ds[var_name].attrs["_FillValue"]
            )
            ds[var_name] *= ds[var_name].attrs["scale_factor"]
        return ds

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        coords = dict()
        ds = self.open_asset(item.assets["geolocation"])
        coords["lon"] = ds["lon"]
        coords["lat"] = ds["lat"]
        ds_item = xr.Dataset(coords=coords, attrs=ds.attrs)
        asset_names = open_params.get("asset_names", self._asset_names)
        assets = list_assets_from_item(item, asset_names=asset_names)
        for asset in assets:
            ds = self.open_asset(asset, **open_params)
            ds_item.update(ds)

        if open_params.get("apply_rectification", True):
            ds_item = rectify_dataset(ds_item)
        return ds_item

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                asset_names=self._asset_names_schema,
                apply_rectification=_SCHEMA_APPLY_RECTIFICATION,
                add_error_bands=_SCHEMA_ADD_ERROR_BANDS,
            ),
            required=[],
            additional_properties=True,
        )


class Sen3LstCdseStacItemAccessor(Sen3CdseStacItemAccessor):
    """Provides methods for accessing the data of a CDSE Sentinel-3
    SLSTR Level-2 Land Surface Temperature products STAC Item.
    """

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        super().__init__(catalog, **storage_options_s3)
        self._lst_asset = "LST_in"
        self._geo_asset = "geodetic_in"
        self._angles = "geometry_tn"
        self._angles_geo = "geodetic_tx"

    def open_asset(self, asset: pystac.Asset, **open_params) -> xr.Dataset:
        ds = rioxarray.open_rasterio(
            asset.href,
            chunks={},
            driver="netCDF",
        )
        if isinstance(ds, list):
            ds = ds[0]
        ds = ds.squeeze()
        return ds

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        # get LST data
        ds = self.open_asset(item.assets[self._lst_asset])
        ds = ds[["LST"]]

        # get geolocation
        geo = self.open_asset(item.assets[self._geo_asset])
        ds = ds.assign_coords(
            dict(
                lat=geo["latitude_in"],
                lon=geo["longitude_in"],
                elev=geo["elevation_in"],
            )
        )
        if open_params.get("apply_geo_orthorectification", True):
            angles = self.open_asset(item.assets[self._angles])
            angles_geo = self.open_asset(item.assets[self._angles_geo])
            ds = orthorectify_geolocation(
                ds,
                angles_geo["latitude_tx"],
                angles_geo["longitude_tx"],
                angles["sat_zenith_tn"],
                angles["sat_azimuth_tn"],
            )
        ds = ds.drop_vars(("x", "y", "band", "spatial_ref"))
        if open_params.get("apply_rectification", True):
            ds = rectify_dataset(ds)
        return ds

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                apply_rectification=_SCHEMA_APPLY_RECTIFICATION,
                apply_geo_orthorectification=_SCHEMA_APPLY_GEO_ORTHORECTIFICATION,
            ),
            required=[],
            additional_properties=True,
        )


class Sen3CdseStacArdcAccessor(Sen3CdseStacItemAccessor, StacArdcAccessor):
    """Provides methods for access multiple Sentinel-3 STAC Items from the
    CDSE STAC API and build an analysis ready data cube."""

    def open_ardc(
        self,
        items: Sequence[pystac.Item],
        **open_params,
    ) -> xr.Dataset:

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
                asset_names=self._asset_names_schema,
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
        target_gm = GridMapping.regular_from_bbox(
            open_params["bbox"],
            open_params["spatial_res"],
            open_params["crs"],
            TILE_SIZE,
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
                ds = rectify_dataset(ds, target_gm=target_gm)
                if ds is None:
                    continue
                dss_spatial.append(ds)
            if not dss_spatial:
                continue
            dss_time.append(mosaic_spatial_take_first(dss_spatial))
        ds_final = xr.concat(dss_time, dim="time", join="override")
        ds_final = ds_final.assign_coords(dict(time=grouped_items.time))
        return ds_final


class Sen3LstCdseStacArdcAccessor(
    Sen3CdseStacArdcAccessor, Sen3LstCdseStacItemAccessor
):
    """Provides methods for access multiple Sentinel-3 SLSTR Level-2 Land Surface
    Temperature STAC Items from the CDSE STAC API and build an analysis ready
    data cube."""

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                time_range=SCHEMA_TIME_RANGE,
                bbox=SCHEMA_BBOX,
                spatial_res=SCHEMA_SPATIAL_RES,
                crs=SCHEMA_CRS,
                query=SCHEMA_ADDITIONAL_QUERY,
            ),
            required=["time_range", "bbox", "spatial_res", "crs"],
            additional_properties=False,
        )


class Sen3PlanetaryComputerStacItemAccessor(Sen3CdseStacItemAccessor):
    """Provides methods for accessing the data of a Planetary Computer
    Sentinel-3 STAC Item."""

    # noinspection PyMissingConstructor
    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        self._asset_names = _SENTINEL3_SYN_PC_ASSETS
        self._asset_names_schema = _SCHEMA_PC_ASSET_NAMES
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        if not self._is_pc_signed(item):
            item = planetary_computer.sign_item(item)
        return super().open_item(item, **open_params)

    @staticmethod
    def _is_pc_signed(item: pystac.Item) -> bool:
        for asset in item.assets.values():
            if "sig=" in asset.href or "sv=" in asset.href:
                return True
        return False


class Sen3LstPlanetaryComputerStacItemAccessor(
    Sen3PlanetaryComputerStacItemAccessor, Sen3LstCdseStacItemAccessor
):
    """Provides methods for accessing a Sentinel-3 SLSTR Level-2 Land Surface
    Temperature product via a Planetary Computer STAC Item.
    """

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        super().__init__(catalog, **storage_options_s3)
        self._lst_asset = "lst-in"
        self._geo_asset = "slstr-geodetic-in"
        self._angles = "slstr-geometry-tn"
        self._angles_geo = "slstr-geodetic-tx"


class Sen3PlanetaryComputerStacArdcAccessor(
    Sen3CdseStacArdcAccessor, Sen3PlanetaryComputerStacItemAccessor
):
    """Provides methods for access multiple Sentinel-3 STAC Items from the
    Planetary Computer STAC API and build an analysis ready data cube."""


class Sen3LstPlanetaryComputerStacArdcAccessor(
    Sen3LstCdseStacArdcAccessor, Sen3LstPlanetaryComputerStacItemAccessor
):
    """Provides methods for access multiple Sentinel-3 LST STAC Items from the
    Planetary Computer STAC API and build an analysis ready data cube."""


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


def orthorectify_geolocation(
    dataset: xr.Dataset,
    lat: xr.DataArray,
    lon: xr.DataArray,
    sat_zenith: xr.DataArray,
    sat_azimuth: xr.DataArray,
) -> xr.Dataset:
    """
    Apply terrain-induced parallax correction to satellite geolocation coordinates.

    Args:
        dataset: Dataset containing geolocation coordinates to be corrected. Must
            include `latitude` and `longitude` coordinates.
        elev: Surface elevation in meters above the reference ellipsoid or sphere.
        lat: Latitude values defining the source grid for satellite angle variables.
        lon: Longitude values defining the source grid for satellite angle variables.
        sat_zenith: Viewing zenith angle in degrees.
        sat_azimuth: Viewing azimuth angle in degrees. Sentinel-3 convention is
            clockwise from North.

    Returns:
        A new dataset with corrected `latitude` and `longitude` coordinates.

    Notes:
    This function adjusts latitude and longitude coordinates in the input dataset to
    compensate for horizontal displacement effects caused by viewing elevated terrain
    from an oblique angle. The correction accounts for local surface height and
    satellite viewing geometry, estimating the apparent pixel shift under the
    assumption of a spherical Earth.

    Satellite zenith and azimuth angles are first interpolated from their native
    grid to the geolocation grid of the dataset using `scipy.interpolate.griddata`.
    Displacements are computed in radians and then applied to produce corrected
    latitude and longitude coordinates.

    The following assumptions are made:

        - Assumes a spherical Earth with a fixed radius of 6,370,997 meters.
        - Atmospheric refraction and ellipsoidal geometry effects are not considered.
        - Accuracy may degrade near the poles where `cos(latitude) → 0`.
    """
    # load coordinates and elevation of dataset
    ds_lat = dataset.lat
    ds_lat = ds_lat.where(ds_lat != ds_lat.attrs["_FillValue"], np.nan)
    ds_lat *= ds_lat.attrs["scale_factor"]
    ds_lat = ds_lat.values
    ds_lon = dataset.lon
    ds_lon = ds_lon.where(ds_lon != ds_lon.attrs["_FillValue"], np.nan)
    ds_lon *= ds_lon.attrs["scale_factor"]
    ds_lon = ds_lon.values
    elev = dataset.elev
    elev = elev.where(elev != elev.attrs["_FillValue"], np.nan)
    elev *= elev.attrs["scale_factor"]
    elev = elev.values

    # interpolate satellite zenith and azimuth angle
    def _interpolate(
        angle: np.ndarray,
        lat_source: np.ndarray,
        lon_source: np.ndarray,
        lat_target: np.ndarray,
        lon_target: np.ndarray,
    ) -> np.ndarray:
        pts_source = np.stack([lat_source.ravel(), lon_source.ravel()], axis=-1)
        pts_target = np.stack([lat_target.ravel(), lon_target.ravel()], axis=-1)
        angle_interp = np.asarray(
            griddata(pts_source, angle.ravel(), pts_target, method="linear")
        )

        # Identify NaNs (outside convex hull)
        mask = np.isnan(angle_interp)
        if np.any(mask):
            # Second pass: nearest fill for NaNs only
            angle_interp[mask] = griddata(
                pts_source, angle.ravel(), pts_target[mask], method="nearest"
            )

        return angle_interp.reshape(lat_target.shape)

    sat_zenith_interp = _interpolate(
        sat_zenith.values, lat.values, lon.values, ds_lat, ds_lon
    )
    sat_azimuth_interp = _interpolate(
        sat_azimuth.values, lat.values, lon.values, ds_lat, ds_lon
    )

    # Convert everything to rad
    phi_true = np.deg2rad(ds_lat)
    theta_v = np.deg2rad(sat_zenith_interp)
    phi_v = np.deg2rad(sat_azimuth_interp)

    # Horizontal displacement
    t = elev * np.tan(theta_v)
    delta_phi = t * np.cos(phi_v) / MEAN_EARTH_RADIUS
    delta_lam = t * np.sin(phi_v) / (MEAN_EARTH_RADIUS * np.cos(phi_true))

    # convert back to degree
    lat_diff = np.rad2deg(delta_phi)
    lon_diff = np.rad2deg(delta_lam)

    return dataset.assign_coords(
        dict(
            latitude=(dataset.latitude.dims, ds_lat - lat_diff),
            longitude=(dataset.latitude.dims, ds_lon - lon_diff),
        )
    )
