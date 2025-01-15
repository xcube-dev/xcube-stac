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
from unittest.mock import MagicMock

import dask
import dask.array as da
import xarray as xr
import rasterio
import rasterio.session
from xcube.core.mldataset import MultiLevelDataset

from xcube_stac.accessor.sen2 import S3Sentinel2DataAccessor


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

    @patch("rasterio.open")
    @patch("rioxarray.open_rasterio")
    def test_open_data(self, mock_rioxarray_open, mock_rasterio_open):
        # set-up mock for rioxarray.open_rasterio
        mock_data = {
            "band_1": (("y", "x"), da.ones((2048, 2048), chunks=(1024, 1024))),
        }
        mock_ds = xr.Dataset(mock_data)
        mock_rioxarray_open.return_value = mock_ds

        # set-up mock for rasterio.open
        mock_rio_dataset = MagicMock()
        mock_rio_dataset.overviews.return_value = [2, 4, 8]
        mock_rasterio_open.return_value.__enter__.return_value = mock_rio_dataset

        # start tests
        access_params = dict(protocol="s3", root="eodata", fs_path="test.tif")
        ds = self.accessor.open_data(access_params)
        mock_rioxarray_open.assert_called_once_with(
            "s3://eodata/test.tif",
            chunks=dict(x=1024, y=1024),
            band_as_variable=True,
        )
        self.assertTrue("band_1" in ds)
        self.assertEqual(ds["band_1"].shape, (2048, 2048))
        self.assertCountEqual(
            [1024, 1024], [ds.chunksizes["x"][0], ds.chunksizes["y"][0]]
        )

        mlds = self.accessor.open_data(access_params, data_type="mldataset")
        self.assertIsInstance(mlds, MultiLevelDataset)
        self.assertEqual(4, mlds.num_levels)
        mock_rasterio_open.assert_called_once_with("s3://eodata/test.tif")
        ds = mlds.base_dataset
        mock_rioxarray_open.assert_called_with(
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
