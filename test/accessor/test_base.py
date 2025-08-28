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

import unittest
from unittest.mock import patch

import numpy as np
import pystac
import xarray as xr
from xcube.core.store import DataStore

from xcube_stac.accessors.base import BaseStacItemAccessor


class BaseStacItemAccessorTest(unittest.TestCase):

    def setUp(self):
        self.catalog = pystac.Catalog(
            id="test-catalog",
            description="Test Catalog",
            href="https://example.com/catalog.json",
        )
        self.accessor = BaseStacItemAccessor(self.catalog)

    @patch("rioxarray.open_rasterio")
    def test_open_asset(self, open_rasterio_mock):
        open_rasterio_mock.return_value = xr.DataArray(
            np.ones((10, 10)), dims=("y", "x")
        )

        asset = pystac.Asset(
            href="s3://bucket/data/my-image.tif",
            media_type="image/tiff; application=geotiff",
            roles=["data"],
            title="Test GeoTIFF",
        )
        ds = self.accessor.open_asset(asset)
        self.assertIsInstance(ds, xr.Dataset)
        self.assertIn("band", ds)
        open_rasterio_mock.assert_called_once_with(
            "s3://bucket/data/my-image.tif",
            overview_level=None,
            chunks={"x": 512, "y": 512},
        )

    def test_get_store(self):
        store0 = self.accessor._get_store("https", "root0")
        self.assertIsInstance(store0, DataStore)
        # noinspection PyUnresolvedReferences
        self.assertEqual("root0", store0.root)
        # noinspection PyUnresolvedReferences
        self.assertIs(self.accessor._https_store, store0)

        with self.assertLogs("xcube.stac", level="DEBUG") as cm:
            store1 = self.accessor._get_store("https", "root1")
        self.assertIn(
            "Initializing a new 'https' data store because root changed", cm.output[-1]
        )
        self.assertIsInstance(store1, DataStore)
        # noinspection PyUnresolvedReferences
        self.assertEqual("root1", store1.root)
        # noinspection PyUnresolvedReferences
        self.assertIs(self.accessor._https_store, store1)
