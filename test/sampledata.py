# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
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

import dask.array as da
import numpy as np
import xarray as xr


def sentinel_2_band_data_10m():
    mock_data = {
        "band_1": (
            ("y", "x"),
            da.ones((10980, 10980), chunks=(1024, 1024), dtype=np.uint16),
        ),
    }
    spatial_ref = xr.DataArray(
        np.array(0),
        attrs={
            "crs_wkt": (
                'PROJCS["WGS 84 / UTM zone 35N",GEOGCS["WGS 84",DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
                'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG"'
                ',"8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG",'
                '"9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_'
                'Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central'
                '_meridian",27],PARAMETER["scale_factor",0.9996],PARAMETER["false'
                '_easting",500000],PARAMETER["false_northing",0],UNIT["metre"'
                ',1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing"'
                ',NORTH],AUTHORITY["EPSG","32635"]]'
            ),
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245179,
            "inverse_flattening": 298.257223563,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "geographic_crs_name": "WGS 84",
            "horizontal_datum_name": "World Geodetic System 1984",
            "projected_crs_name": "WGS 84 / UTM zone 35N",
            "grid_mapping_name": "transverse_mercator",
            "latitude_of_projection_origin": 0.0,
            "longitude_of_central_meridian": 27.0,
            "false_easting": 500000.0,
            "false_northing": 0.0,
            "scale_factor_at_central_meridian": 0.9996,
            "spatial_ref": (
                'PROJCS["WGS 84 / UTM zone 35N",GEOGCS["WGS 84",DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]]'
                ',AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG"'
                ',"8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG",'
                '"9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_'
                'Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_'
                'meridian",27],PARAMETER["scale_factor",0.9996],PARAMETER["false_'
                'easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,'
                'AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",'
                'NORTH],AUTHORITY["EPSG","32635"]]'
            ),
            "GeoTransform": "600000.0 10.0 0.0 5900040.0 0.0 -10.0",
        },
    )
    coords = {
        "x": np.arange(600005.0, 709796.0, 10.0),
        "y": np.arange(5900035.0, 5790244.0, -10.0),
        "spatial_ref": spatial_ref,
    }
    return xr.Dataset(mock_data, coords=coords)


def sentinel_2_band_data_60m():
    mock_data = {
        "band_1": (
            ("y", "x"),
            da.ones((1830, 1830), chunks=(1024, 1024), dtype=np.uint16),
        ),
    }
    spatial_ref = xr.DataArray(
        np.array(0),
        attrs={
            "crs_wkt": (
                'PROJCS["WGS 84 / UTM zone 32N",GEOGCS["WGS 84",DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
                'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG",'
                '"8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                'AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER'
                '["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER'
                '["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER'
                '["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS'
                '["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32632"]]'
            ),
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245179,
            "inverse_flattening": 298.257223563,
            "reference_ellipsoid_name": "WGS 84",
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "geographic_crs_name": "WGS 84",
            "horizontal_datum_name": "World Geodetic System 1984",
            "projected_crs_name": "WGS 84 / UTM zone 32N",
            "grid_mapping_name": "transverse_mercator",
            "latitude_of_projection_origin": 0.0,
            "longitude_of_central_meridian": 9.0,
            "false_easting": 500000.0,
            "false_northing": 0.0,
            "scale_factor_at_central_meridian": 0.9996,
            "spatial_ref": (
                'PROJCS["WGS 84 / UTM zone 32N",GEOGCS["WGS 84",DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
                'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG",'
                '"8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                'AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER'
                '["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER'
                '["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER'
                '["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS'
                '["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32632"]]'
            ),
            "GeoTransform": "499980.0 60.0 0.0 5900040.0 0.0 -60.0",
        },
    )
    coords = {
        "x": np.arange(500010.0, 609751.0, 60.0),
        "y": np.arange(5999970.0, 5890229.0, -60.0),
        "spatial_ref": spatial_ref,
    }
    return xr.Dataset(mock_data, coords=coords)


def sentinel_3_syn_data():
    mock_data = {
        "SDR_Oa01": (
            ("band", "y", "x"),
            da.ones((1, 150, 200), chunks=(1, 150, 200), dtype=np.float32),
        ),
        "SDR_Oa01_err": (
            ("band", "y", "x"),
            da.ones((1, 150, 200), chunks=(1, 150, 200), dtype=np.float32),
        ),
    }
    coords = {
        "spatial_ref": np.array([0]),
        "band": np.array([0]),
        "x": np.arange(200),
        "y": np.arange(150),
    }
    ds = xr.Dataset(mock_data, coords=coords)
    ds["SDR_Oa01"].attrs = {"_FillValue": -10000, "scale_factor": 0.0001}
    ds["SDR_Oa01_err"].attrs = {"_FillValue": -10000, "scale_factor": 0.0001}
    return ds


def sentinel_3_syn_cloud_data():
    flag_meanings = "a b c d"
    flag_masks = np.array([1, 2, 4, 8], dtype=np.uint8)

    # Create flags array
    data = np.zeros((1, 200, 150), dtype=np.uint8)
    data[:, 0:50, 0:50] = 1
    data[:, 100:150, 100:150] = 2
    data[:, 150:200, 100:150] = 3
    data[:, 50:100, 50:100] = 4
    confidence_in = da.from_array(data, chunks=(1, 200, 150))

    ds_flags = xr.Dataset(
        {"CLOUD_flags": (("band", "y", "x"), confidence_in)},
        coords={
            "band": [0],
            "y": np.arange(200),
            "x": np.arange(150),
            "spatial_ref": 0,
        },
    )
    ds_flags["CLOUD_flags"].attrs = {
        "flag_meanings": flag_meanings,
        "flag_masks": flag_masks,
    }

    return ds_flags


def sentinel_3_syn_geolocation_data():
    lon = da.linspace(0, 15, 200, chunks=200, dtype=np.float64)
    lat = da.linspace(50, 60, 150, chunks=150, dtype=np.float64)
    lon, lat = da.meshgrid(lon, lat, indexing="xy")
    lon /= da.cos(da.radians(lat))

    # skew due to earth curvature
    skew = 0.2
    lon += skew * (lat - lat[0]) / (lat[-1] - lat[0])

    # rotate image
    rotation_deg = -25
    theta = np.radians(rotation_deg)
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)
    x = lon - lon0
    y = lat - lat0
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    lon_final = x_rot + lon0
    lat_final = y_rot + lat0

    coords = {
        "spatial_ref": np.array([0]),
        "band": np.array([0]),
        "x": np.arange(200),
        "y": np.arange(150),
    }
    mock_data = {
        "lon": (("band", "y", "x"), da.expand_dims(lon_final, axis=0)),
        "lat": (("band", "y", "x"), da.expand_dims(lat_final, axis=0)),
    }

    ds = xr.Dataset(mock_data, coords=coords)
    ds["lon"].attrs = {"_FillValue": -10000, "scale_factor": 0.0001}
    ds["lat"].attrs = {"_FillValue": -10000, "scale_factor": 0.0001}
    return ds


def sentinel_3_lst_data():
    mock_data = {
        "LST": (
            ("band", "y", "x"),
            da.ones((1, 500, 500), chunks=(1, 500, 500), dtype=np.float32),
        )
    }
    coords = {
        "spatial_ref": np.array([0]),
        "band": np.array([0]),
        "x": np.arange(500),
        "y": np.arange(500),
    }
    ds = xr.Dataset(mock_data, coords=coords)
    ds["LST"].attrs = {
        "_FillValue": -10000,
        "scale_factor": 0.0001,
        "add_offset": 273.32,
    }
    return ds


def sentinel_3_lst_flag_data():
    # Define bit positions like real Sentinel-3
    flag_meanings = (
        "coastline ocean tidal land inland_water unfilled spare spare "
        "cosmetic duplicate day twilight sun_glint snow summary_cloud summary_pointing"
    )
    flag_masks = np.array(
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        dtype=np.uint16,
    )

    unfilled_bit = 32
    cloud_bit = 16384
    pointing_bit = 32768

    # Create flags array
    data = np.zeros((1, 500, 500), dtype=np.uint16)
    data[:, 0:150, 0:150] = unfilled_bit  # unfilled block
    data[:, 150:300, 150:300] = cloud_bit  # cloud block
    data[:, 300:450, 300:450] = pointing_bit  # bad pointing block
    data[:, 450:500, 450:500] = (
        cloud_bit | pointing_bit
    )  # combined bits (priority test)
    confidence_in = da.from_array(data, chunks=(1, 500, 500))

    ds_flags = xr.Dataset(
        {"confidence_in": (("band", "y", "x"), confidence_in)},
        coords={
            "band": [0],
            "y": np.arange(500),
            "x": np.arange(500),
            "spatial_ref": 0,
        },
    )

    ds_flags["confidence_in"].attrs = {
        "flag_meanings": flag_meanings,
        "flag_masks": flag_masks,
    }

    return ds_flags


def sentinel_3_angles_data():
    mock_data = {
        "sat_azimuth_tn": (
            ("band", "y", "x"),
            da.ones((1, 500, 40), chunks=(1, 500, 40), dtype=np.float32),
        ),
        "sat_zenith_tn": (
            ("band", "y", "x"),
            da.ones((1, 500, 40), chunks=(1, 500, 40), dtype=np.float32),
        ),
    }
    coords = {
        "spatial_ref": np.array([0]),
        "band": np.array([0]),
        "x": np.arange(40),
        "y": np.arange(500),
    }
    return xr.Dataset(mock_data, coords=coords)


def sentinel_3_angles_geolocation_data():
    lon = da.linspace(-2, 17, 40, chunks=40, dtype=np.float64)
    lat = da.linspace(50, 60, 500, chunks=40, dtype=np.float64)
    lon, lat = da.meshgrid(lon, lat, indexing="xy")
    lon /= da.cos(da.radians(lat))

    # skew due to earth curvature
    skew = 0.2
    lon += skew * (lat - lat[0]) / (lat[-1] - lat[0])

    # rotate image
    rotation_deg = -25
    theta = np.radians(rotation_deg)
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)
    x = lon - lon0
    y = lat - lat0
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    lon_final = x_rot + lon0

    coords = {
        "spatial_ref": np.array([0]),
        "band": np.array([0]),
        "x": np.arange(40),
        "y": np.arange(500),
    }
    mock_data = {
        "longitude_tx": (("band", "y", "x"), da.expand_dims(lon_final, axis=0)),
    }

    return xr.Dataset(mock_data, coords=coords)


def sentinel_3_lst_geolocation_data():
    lon = da.linspace(0, 15, 500, chunks=500, dtype=np.float64)
    lat = da.linspace(50, 60, 500, chunks=500, dtype=np.float64)
    lon, lat = da.meshgrid(lon, lat, indexing="xy")
    lon /= da.cos(da.radians(lat))

    # skew due to earth curvature
    skew = 0.2
    lon += skew * (lat - lat[0]) / (lat[-1] - lat[0])

    # rotate image
    rotation_deg = -25
    theta = np.radians(rotation_deg)
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)
    x = lon - lon0
    y = lat - lat0
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    lon_final = x_rot + lon0
    lat_final = y_rot + lat0

    coords = {
        "spatial_ref": np.array([0]),
        "band": np.array([0]),
        "x": np.arange(500),
        "y": np.arange(500),
    }
    elev = da.linspace(0, 1000, 500**2).reshape((500, 500))
    mock_data = {
        "longitude_in": (("band", "y", "x"), da.expand_dims(lon_final, axis=0)),
        "latitude_in": (("band", "y", "x"), da.expand_dims(lat_final, axis=0)),
        "elevation_in": (("band", "y", "x"), da.expand_dims(elev, axis=0)),
    }

    ds = xr.Dataset(mock_data, coords=coords)
    ds["longitude_in"].attrs = {"_FillValue": -10000, "scale_factor": 0.0001}
    ds["latitude_in"].attrs = {"_FillValue": -10000, "scale_factor": 0.0001}
    ds["elevation_in"].attrs = {"_FillValue": -9999, "scale_factor": 1.0}
    return ds
