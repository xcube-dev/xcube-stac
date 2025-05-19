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
