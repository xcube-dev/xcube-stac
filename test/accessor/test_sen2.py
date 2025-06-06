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

import datetime
import unittest
from unittest.mock import MagicMock, patch

import dask
import dask.array as da
import pystac
import rasterio
import rasterio.session
import xarray as xr
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
            chunks=dict(y=256, x=256),
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
            chunks=dict(y=256, x=256),
            band_as_variable=True,
        )
        self.assertTrue("band_1" in ds)
        self.assertEqual(ds["band_1"].shape, (2048, 2048))
        self.assertCountEqual(
            [1024, 1024], [ds.chunksizes["x"][0], ds.chunksizes["y"][0]]
        )

    def test_groupby_solar_day(self):
        bbox = [-10.0, -10.0, 10.0, 10.0]
        datetime_value = datetime.datetime(2023, 1, 1, 12, 0, 0)
        grid_code = "MGRS-31TCK"
        processing_version = "5.11"

        # Create the Item
        items = []
        for i in range(3):
            items.append(
                pystac.Item(
                    id=f"example-item{i}",
                    geometry={
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [bbox[0], bbox[1]],
                                [bbox[2], bbox[1]],
                                [bbox[2], bbox[3]],
                                [bbox[0], bbox[3]],
                                [bbox[0], bbox[1]],
                            ]
                        ],
                    },
                    bbox=bbox,
                    datetime=datetime_value,
                    properties={
                        "grid:code": grid_code,
                        "processing:version": processing_version,
                    },
                )
            )
        with self.assertLogs("xcube.stac", level="WARNING") as cm:
            grouped = self.accessor.groupby_solar_day(items)
        self.assertEqual(1, len(cm.output))
        msg = (
            "WARNING:xcube.stac:More that two items found for datetime and tile ID: "
            "[example-item0, example-item1, example-item2]. Only the first "
            "two items are considered."
        )
        self.assertEqual(msg, str(cm.output[-1]))
        self.assertEqual((1, 1, 2), grouped.shape)
