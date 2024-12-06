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
from unittest.mock import patch

import dask
import dask.array as da
import xarray as xr
import rasterio
from xcube.core.mldataset import MultiLevelDataset

from xcube_stac.accessor import S3Sentinel2DataAccessor


class TestS3Sentinel2DataAccessor(unittest.TestCase):
    def setUp(self):
        storage_options = dict(
            anon=False,
            key="xxx",
            secret="xxx",
            client_kwargs=dict(endpoint_url="https://eodata.dataspace.copernicus.eu"),
        )
        self.accessor = S3Sentinel2DataAccessor(
            root="eodata", storage_options=storage_options
        )

    def test_init(self):
        self.assertEqual("eodata", self.accessor._root)
        self.assertIsInstance(self.accessor.session, rasterio.session.AWSSession)
        self.assertIsInstance(self.accessor.env, rasterio.env.Env)
        self.assertEqual("single-threaded", dask.config.get("scheduler"))

    def test_del(self):
        with self.assertLogs("xcube.stac", level="DEBUG") as cm:
            del self.accessor
        self.assertEqual(1, len(cm.output))
        msg = "DEBUG:xcube.stac:Exit rasterio.env.Env for CDSE data access."
        self.assertEqual(msg, str(cm.output[-1]))

    def test_root(self):
        self.assertEqual("eodata", self.accessor.root)

    @patch("rioxarray.open_rasterio")
    def test_open_data(self, mock_open_rasterio):
        # set-up mock
        mock_data = {
            "band_1": (("y", "x"), da.ones((2048, 2048), chunks=(1024, 1024))),
        }
        mock_ds = xr.Dataset(mock_data)
        mock_open_rasterio.return_value = mock_ds

        access_params = dict(protocol="s3", root="eodata", fs_path="test.tif")
        ds = self.accessor.open_data(access_params)
        mock_open_rasterio.assert_called_once_with(
            "s3://eodata/test.tif",
            chunks=dict(x=1024, y=1024),
            band_as_variable=True,
        )
        self.assertTrue("band_1" in ds)
        self.assertEqual(ds["band_1"].shape, (2048, 2048))
        self.assertCountEqual(
            [1024, 1024], [ds.chunksizes["x"][0], ds.chunksizes["y"][0]]
        )

        with self.assertLogs("xcube.stac", level="INFO") as cm:
            ds = self.accessor.open_data(access_params, tile_size=(512, 512))
        self.assertEqual(1, len(cm.output))
        msg = (
            "INFO:xcube.stac:The parameter tile_size is set to (1024, 1024), which "
            "is the native chunk size of the jp2 files in the Sentinel-2 archive."
        )
        self.assertEqual(msg, str(cm.output[-1]))
        self.assertTrue("band_1" in ds)
        self.assertEqual(ds["band_1"].shape, (2048, 2048))
        self.assertCountEqual(
            [1024, 1024], [ds.chunksizes["x"][0], ds.chunksizes["y"][0]]
        )

        mlds = self.accessor.open_data(access_params, data_type="mldataset")
        self.assertIsInstance(mlds, MultiLevelDataset)
        ds = mlds.base_dataset
        mock_open_rasterio.assert_called_with(
            "s3://eodata/test.tif",
            overview_level=None,
            chunks=dict(x=1024, y=1024),
            band_as_variable=True,
        )
        self.assertTrue("band_1" in ds)
        self.assertEqual(ds["band_1"].shape, (2048, 2048))
        self.assertCountEqual(
            [1024, 1024], [ds.chunksizes["x"][0], ds.chunksizes["y"][0]]
        )
