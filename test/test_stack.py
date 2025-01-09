# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
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
import unittest

import numpy as np
import xarray as xr

from xcube_stac.stack import mosaic_take_first


class UtilsTest(unittest.TestCase):

    def test_mosaic_take_first(self):
        list_ds = []
        # first tile
        data = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, np.nan], [np.nan, np.nan, np.nan]],
                [[19, 20, 21], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            ],
            dtype=float,
        )
        dims = ("time", "lat", "lon")
        coords = {
            "time": np.array(
                ["2025-01-01", "2025-01-02", "2025-01-03"], dtype="datetime64"
            ),
            "lat": [10.0, 20.0, 30.0],
            "lon": [100.0, 110.0, 120.0],
        }
        da = xr.DataArray(data, dims=dims, coords=coords)
        crs = xr.DataArray(np.array(0), attrs=dict(crs_wkt="testing"))
        list_ds.append(xr.Dataset({"B01": da, "crs": crs}))
        # second tile
        data = np.array(
            [
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, 106], [107, 108, 109]],
                [[np.nan, np.nan, np.nan], [113, 114, 115], [116, 117, 118]],
                [[np.nan, np.nan, 120], [121, 122, 123], [124, 125, 126]],
            ],
            dtype=float,
        )
        dims = ("time", "lat", "lon")
        coords = {
            "time": np.array(
                ["2025-01-01", "2025-01-02", "2025-01-04"], dtype="datetime64"
            ),
            "lat": [10.0, 20.0, 30.0],
            "lon": [100.0, 110.0, 120.0],
        }
        da = xr.DataArray(data, dims=dims, coords=coords)
        crs = xr.DataArray(np.array(0), attrs=dict(crs_wkt="testing"))
        list_ds.append(xr.Dataset({"B01": da, "crs": crs}))

        # test only one tile
        dts = np.array(["2025-01-01", "2025-01-02", "2025-01-03"], dtype="datetime64")
        ds_test = mosaic_take_first(list_ds[:1], dts)
        xr.testing.assert_allclose(ds_test, list_ds[0])

        # test two tiles
        dts = np.array(
            ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"], dtype="datetime64"
        )
        ds_test = mosaic_take_first(list_ds, dts)
        data = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 115], [116, 117, 118]],
                [[19, 20, 21], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, 120], [121, 122, 123], [124, 125, 126]],
            ],
            dtype=float,
        )
        dims = ("time", "lat", "lon")
        coords = {
            "time": dts,
            "lat": [10.0, 20.0, 30.0],
            "lon": [100.0, 110.0, 120.0],
        }
        da = xr.DataArray(data, dims=dims, coords=coords)
        crs = xr.DataArray(np.array(0), attrs=dict(crs_wkt="testing"))
        ds_expected = xr.Dataset({"B01": da, "crs": crs})
        xr.testing.assert_allclose(ds_test, ds_expected)
