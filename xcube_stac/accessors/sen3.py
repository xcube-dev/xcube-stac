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
from collections import defaultdict

import dask.array as da
import numpy as np
import planetary_computer
import pystac
import rasterio.session
import rioxarray
import xarray as xr
from rasterio.errors import NotGeoreferencedWarning
from scipy.interpolate import interp1d
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonObjectSchema,
    JsonStringSchema,
)
from xcube_resampling import rectify_dataset
from xcube_resampling.gridmapping import GridMapping

from xcube_stac.accessor import StacArdcAccessor, StacItemAccessor
from xcube_stac.constants import (
    MEAN_EARTH_RADIUS,
    SCHEMA_ADDITIONAL_QUERY,
    SCHEMA_BBOX,
    SCHEMA_CRS,
    SCHEMA_SPATIAL_RES,
    SCHEMA_TIME_RANGE,
    TILE_SIZE,
)
from xcube_stac.utils import (
    add_nominal_datetime,
    list_assets_from_item,
    mosaic_spatial_take_first,
)

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


_SENTINEL3_SYN_CDSE_ASSETS_VAR_NAME = {
    "syn_S1N_reflectance": "SDR_S1N",
    "syn_S1O_reflectance": "SDR_S1O",
    "syn_S2N_reflectance": "SDR_S2N",
    "syn_S2O_reflectance": "SDR_S2O",
    "syn_S3N_reflectance": "SDR_S3N",
    "syn_S3O_reflectance": "SDR_S3O",
    "syn_S5N_reflectance": "SDR_S5N",
    "syn_S5O_reflectance": "SDR_S5O",
    "syn_S6N_reflectance": "SDR_S6N",
    "syn_S6O_reflectance": "SDR_S6O",
    "syn_Oa01_reflectance": "SDR_Oa01",
    "syn_Oa02_reflectance": "SDR_Oa02",
    "syn_Oa03_reflectance": "SDR_Oa03",
    "syn_Oa04_reflectance": "SDR_Oa04",
    "syn_Oa05_reflectance": "SDR_Oa05",
    "syn_Oa06_reflectance": "SDR_Oa06",
    "syn_Oa07_reflectance": "SDR_Oa07",
    "syn_Oa08_reflectance": "SDR_Oa08",
    "syn_Oa09_reflectance": "SDR_Oa09",
    "syn_Oa10_reflectance": "SDR_Oa10",
    "syn_Oa11_reflectance": "SDR_Oa11",
    "syn_Oa12_reflectance": "SDR_Oa12",
    "syn_Oa16_reflectance": "SDR_Oa16",
    "syn_Oa17_reflectance": "SDR_Oa17",
    "syn_Oa18_reflectance": "SDR_Oa18",
    "syn_Oa21_reflectance": "SDR_Oa21",
}
_SENTINEL3_SYN_PC_ASSETS_VAR_NAME = {
    key.replace("_", "-").lower(): value
    for (key, value) in _SENTINEL3_SYN_CDSE_ASSETS_VAR_NAME.items()
}

_SENTINEL3_SLSTR_LST_CDSE_ASSETS_VAR_NAME = {"LST_in": "LST"}
_SENTINEL3_SLSTR_LST_PC_ASSETS_VAR_NAME = {"lst-in": "LST"}

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
_SCHEMA_ADD_FLAGS = JsonBooleanSchema(
    title="Add flags",
    description="If True, flags are added.",
    default=True,
)
_SCHEMA_CDSE_ASSET_NAMES = JsonArraySchema(
    items=(
        JsonStringSchema(
            min_length=1, enum=list(_SENTINEL3_SYN_CDSE_ASSETS_VAR_NAME.keys())
        )
    ),
    unique_items=True,
    title="Names of assets (spectral bands).",
)
_SCHEMA_PC_ASSET_NAMES = JsonArraySchema(
    items=(
        JsonStringSchema(min_length=1, enum=list(_SENTINEL3_SYN_PC_ASSETS_VAR_NAME))
    ),
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
        self._asset_var_names = _SENTINEL3_SYN_CDSE_ASSETS_VAR_NAME
        self._asset_names_schema = _SCHEMA_CDSE_ASSET_NAMES
        self._flags = "flags"
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

    def open_asset(self, asset: pystac.Asset, **open_params) -> xr.Dataset:
        return rioxarray.open_rasterio(asset.href, chunks={}, driver="netCDF").squeeze()

    def open_item(self, item: pystac.Item, **open_params) -> xr.Dataset:
        asset_names = open_params.get("asset_names", list(self._asset_var_names.keys()))
        assets = list_assets_from_item(item, asset_names=asset_names)
        ds = None
        for asset in assets:
            ds_asset = self.open_asset(asset, **open_params)
            if ds is None:
                ds = ds_asset
            else:
                ds.update(ds_asset)
        var_names = list(ds.data_vars)
        if not open_params.get("add_error_bands", True):
            var_names = [
                var_name for var_name in var_names if not var_name.endswith("_err")
            ]
            ds = ds[var_names]
        ds = _apply_scaling(ds)

        # add flags
        if open_params.get("add_flags", True):
            flags = self.open_asset(item.assets[self._flags])
            ds.update(flags)

        # add geolocation
        geo = self.open_asset(item.assets["geolocation"])
        geo = _apply_scaling(geo[["lon", "lat"]])
        ds = ds.assign_coords(dict(lat=geo["lat"], lon=geo["lon"]))

        ds = ds.drop_vars(["band", "x", "y", "spatial_ref"])
        if open_params.get("apply_rectification", True):
            ds = rectify_dataset(ds, prevent_nan_propagations=True)

        for var in ds.data_vars:
            # Remove CF scaling attributes if present
            ds[var].attrs.pop("scale_factor", None)
            ds[var].attrs.pop("add_offset", None)
            fill = ds[var].attrs.pop("_FillValue", None)
            if fill is not None:
                ds[var].encoding["_FillValue"] = fill
        return ds

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                asset_names=self._asset_names_schema,
                apply_rectification=_SCHEMA_APPLY_RECTIFICATION,
                add_error_bands=_SCHEMA_ADD_ERROR_BANDS,
                add_flags=_SCHEMA_ADD_FLAGS,
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
        self._asset_var_names = _SENTINEL3_SLSTR_LST_CDSE_ASSETS_VAR_NAME
        self._geo_asset = "geodetic_in"
        self._angles = "geometry_tn"
        self._angles_geo = "geodetic_tx"
        self._flags = "flags_in"

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
        ds = self.open_asset(item.assets[list(self._asset_var_names.keys())[0]])
        ds = _apply_scaling(ds[["LST"]])

        # get geolocation
        geo = self.open_asset(item.assets[self._geo_asset])
        geo = _apply_scaling(geo[["latitude_in", "longitude_in", "elevation_in"]])
        ds = ds.assign_coords(
            dict(
                lat=geo["latitude_in"],
                lon=geo["longitude_in"],
                elev=geo["elevation_in"],
            )
        )
        if open_params.get("apply_geo_orthorectification", True):
            angles = self.open_asset(item.assets[self._angles])
            angles = angles[["sat_azimuth_tn", "sat_zenith_tn"]]
            angles_geo = self.open_asset(item.assets[self._angles_geo])
            angles = angles.assign_coords(dict(lon=angles_geo["longitude_tx"]))
            ds = orthorectify_geolocation(ds, angles)
        if open_params.get("add_flags", True):
            flags = self.open_asset(item.assets[self._flags])
            ds.update(flags)
        ds = ds.drop_vars(("x", "y", "band", "spatial_ref", "elev"), errors="ignore")
        if open_params.get("apply_rectification", True):
            ds = rectify_dataset(ds, prevent_nan_propagations=True)

        for var in ds.data_vars:
            # Remove CF scaling attributes if present
            ds[var].attrs.pop("scale_factor", None)
            ds[var].attrs.pop("add_offset", None)
            fill = ds[var].attrs.pop("_FillValue", None)
            if fill is not None:
                ds[var].encoding["_FillValue"] = fill
        return ds

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                apply_rectification=_SCHEMA_APPLY_RECTIFICATION,
                apply_geo_orthorectification=_SCHEMA_APPLY_GEO_ORTHORECTIFICATION,
                add_flags=_SCHEMA_ADD_FLAGS,
            ),
            required=[],
            additional_properties=True,
        )


class Sen3CdseStacArdcAccessor(Sen3CdseStacItemAccessor, StacArdcAccessor):
    """Provides methods for access multiple Sentinel-3 STAC Items from the
    CDSE STAC API and build an analysis ready data cube."""

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        super().__init__(catalog, **storage_options_s3)
        self._asset_var_names = _SENTINEL3_SYN_CDSE_ASSETS_VAR_NAME
        self._flags = "flags"

    def open_ardc(
        self,
        items: list[pystac.Item],
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
                add_flags=_SCHEMA_ADD_FLAGS,
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
        asset_names = open_params.get("asset_names", list(self._asset_var_names.keys()))
        var_ref = self._asset_var_names[asset_names[0]]
        for dt_idx, dt in enumerate(grouped_items.time.values):
            items = grouped_items.sel(time=dt).item()
            dss_spatial = []
            for item in items:
                ds = self.open_item(
                    item,
                    asset_names=asset_names,
                    add_error_bands=open_params.get("add_error_bands", True),
                    add_flags=open_params.get("add_flags", True),
                    apply_rectification=False,
                )
                ds = rectify_dataset(
                    ds,
                    target_gm=target_gm,
                    prevent_nan_propagations=True,
                )
                if ds is None:
                    continue
                dss_spatial.append(ds)
            if not dss_spatial:
                continue
            dss_time.append(mosaic_spatial_take_first(dss_spatial, var_ref, np.nan))
        ds_final = xr.concat(dss_time, dim="time", join="override")
        ds_final = ds_final.assign_coords(dict(time=grouped_items.time))

        for var in ds_final.data_vars:
            # Remove CF scaling attributes if present
            ds_final[var].attrs.pop("scale_factor", None)
            ds_final[var].attrs.pop("add_offset", None)
            fill = ds_final[var].attrs.pop("_FillValue", None)
            if fill is not None:
                ds_final[var].encoding["_FillValue"] = fill
        return ds_final


class Sen3LstCdseStacArdcAccessor(
    Sen3CdseStacArdcAccessor, Sen3LstCdseStacItemAccessor
):
    """Provides methods for access multiple Sentinel-3 SLSTR Level-2 Land Surface
    Temperature STAC Items from the CDSE STAC API and build an analysis ready
    data cube."""

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        super().__init__(catalog, **storage_options_s3)
        self._asset_var_names = _SENTINEL3_SLSTR_LST_CDSE_ASSETS_VAR_NAME
        self._flags = "flags_in"

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
                add_flags=_SCHEMA_ADD_FLAGS,
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
        self._asset_var_names = _SENTINEL3_SYN_PC_ASSETS_VAR_NAME
        self._asset_names_schema = _SCHEMA_PC_ASSET_NAMES
        self._flags = "syn-flags"

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
        self._asset_var_names = _SENTINEL3_SLSTR_LST_PC_ASSETS_VAR_NAME
        self._geo_asset = "slstr-geodetic-in"
        self._angles = "slstr-geometry-tn"
        self._angles_geo = "slstr-geodetic-tx"
        self._flags = "slstr-flags-in"


class Sen3PlanetaryComputerStacArdcAccessor(
    Sen3CdseStacArdcAccessor, Sen3PlanetaryComputerStacItemAccessor
):
    """Provides methods for access multiple Sentinel-3 STAC Items from the
    Planetary Computer STAC API and build an analysis ready data cube."""

    # noinspection PyMissingConstructor
    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        self._asset_var_names = _SENTINEL3_SYN_PC_ASSETS_VAR_NAME
        self._asset_names_schema = _SCHEMA_PC_ASSET_NAMES
        self._flags = "syn-flags"


class Sen3LstPlanetaryComputerStacArdcAccessor(
    Sen3LstCdseStacArdcAccessor, Sen3LstPlanetaryComputerStacItemAccessor
):
    """Provides methods for access multiple Sentinel-3 LST STAC Items from the
    Planetary Computer STAC API and build an analysis ready data cube."""

    # noinspection PyMissingConstructor
    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._catalog = catalog
        self._asset_var_names = _SENTINEL3_SLSTR_LST_PC_ASSETS_VAR_NAME
        self._geo_asset = "slstr-geodetic-in"
        self._angles = "slstr-geometry-tn"
        self._angles_geo = "slstr-geodetic-tx"
        self._flags = "slstr-flags-in"


def _group_items(items: list[pystac.Item]) -> xr.DataArray:
    items = add_nominal_datetime(items)

    groups = defaultdict(list)

    for item in items:
        dt = item.properties["datetime_nominal"]
        date = dt.date()
        orbit = item.properties["sat:orbit_state"]

        key = (date, orbit)
        groups[key].append(item)

    # Sort keys chronologically and descending before ascending
    orbit_order = {"descending": 0, "ascending": 1}
    sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], orbit_order[k[1]]))

    grouped_items = np.empty(len(sorted_keys), dtype=object)
    for i, k in enumerate(sorted_keys):
        grouped_items[i] = groups[k]

    # Mean timestamp per group
    dts = np.empty(len(grouped_items), dtype="datetime64[s]")
    for i, items in enumerate(grouped_items):
        times = np.array(
            [np.datetime64(item.datetime.replace(tzinfo=None)) for item in items]
        )
        mean_time = np.datetime64(int(times.view("int64").mean()), "us")
        dts[i] = mean_time.astype("datetime64[s]")

    da = xr.DataArray(grouped_items, dims=("time",), coords=dict(time=dts))

    da["time"].encoding["units"] = "seconds since 1970-01-01"
    da["time"].encoding["calendar"] = "standard"

    return da


def orthorectify_geolocation(
    dataset: xr.Dataset,
    angles: xr.Dataset,
) -> xr.Dataset:
    """
    Apply terrain-induced parallax correction to satellite geolocation coordinates.

    Notes:
    The following assumptions are made:

        - Assumes a spherical Earth with a fixed radius of 6,370,997 meters.
        - Atmospheric refraction and ellipsoidal geometry effects are not considered.
        - Accuracy may degrade near the poles where `cos(latitude) → 0`.
    """

    def _interp_rowwise(lon_t, lon_s, angle_s):
        out = np.empty_like(lon_t)
        for i in range(lon_s.shape[0]):
            f = interp1d(
                lon_s[i, :],
                angle_s[i, :],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            out[i, :] = f(lon_t[i, :])
        return out

    ds_lat = dataset.lat.data.rechunk((100, -1))
    ds_lon = dataset.lon.data.rechunk((100, -1))
    elev = dataset.elev.fillna(0).data.rechunk((100, -1))
    sat_zenith = angles.sat_zenith_tn.data.rechunk((100, -1))
    sat_azimuth = angles.sat_azimuth_tn.data.rechunk((100, -1))
    angles_lon = angles.lon.data.rechunk((100, -1))

    sat_zenith_interp = da.map_blocks(
        _interp_rowwise,
        ds_lon,
        angles_lon,
        sat_zenith,
        dtype=ds_lon.dtype,
        chunks=ds_lon.chunks,
    )
    sat_azimuth_interp = da.map_blocks(
        _interp_rowwise,
        ds_lon,
        angles_lon,
        sat_azimuth,
        dtype=ds_lon.dtype,
        chunks=ds_lon.chunks,
    )

    phi_true = np.deg2rad(ds_lat)
    theta_v = np.deg2rad(sat_zenith_interp)
    phi_v = np.deg2rad(sat_azimuth_interp)

    # Parallax correction
    t = elev * np.tan(theta_v)
    delta_phi = t * np.cos(phi_v) / MEAN_EARTH_RADIUS
    delta_lam = t * np.sin(phi_v) / (MEAN_EARTH_RADIUS * np.cos(phi_true))

    lat_diff = np.rad2deg(delta_phi)
    lon_diff = np.rad2deg(delta_lam)

    final_lat = xr.DataArray(
        (ds_lat - lat_diff).rechunk(dataset.lat.chunks),
        dims=dataset.lat.dims,
        attrs=dataset.lat.attrs,
    )
    final_lon = xr.DataArray(
        (ds_lon - lon_diff).rechunk(dataset.lon.chunks),
        dims=dataset.lon.dims,
        attrs=dataset.lon.attrs,
    )
    return dataset.assign_coords(lat=final_lat, lon=final_lon)


def _apply_scaling(ds: xr.Dataset) -> xr.Dataset:
    ds_out = ds.copy()
    for var, array in ds_out.data_vars.items():
        fill_value = array.attrs.get("_FillValue")
        if fill_value is not None:
            ds_out[var] = array.where(array != fill_value, np.nan)
        scale_factor = array.attrs.get("scale_factor")
        if scale_factor is not None:
            ds_out[var] = ds_out[var] * scale_factor
        offset = array.attrs.get("add_offset")
        if offset is not None:
            ds_out[var] = ds_out[var] + offset

    return ds_out
