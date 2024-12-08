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
import datetime

import pystac

from xcube_stac.helper import HelperCdse


class HelperCdseTest(unittest.TestCase):

    def setUp(self):
        self.asset = pystac.Asset(
            href="test_href",
            media_type="dummy",
            roles=["data"],
            extra_fields=dict(
                alternate=dict(
                    s3=dict(
                        href=(
                            "/eodata/Sentinel-2/MSI/L2A/2024/11/07/S2A_MSIL2A_20241107"
                            "T113311_N0511_R080_T31VDG_20241107T123948.SAFE"
                        )
                    )
                )
            ),
        )
        self.item = pystac.Item(
            id="cdse_item_parts",
            geometry={
                "type": "Polygon",
                "coordinates": [
                    [
                        [100.0, 0.0],
                        [101.0, 0.0],
                        [101.0, 1.0],
                        [100.0, 1.0],
                        [100.0, 0.0],
                    ]
                ],
            },
            bbox=[100.0, 0.0, 101.0, 1.0],
            datetime=datetime.datetime(2023, 1, 1, 0, 0, 0),
            properties=dict(
                tileId="title_id",
                orbitNumber=0,
            ),
        )
        self.item.add_asset("PRODUCT", self.asset)

    @patch("s3fs.S3FileSystem.glob")
    def test_parse_item(self, mock_glob):
        mock_glob.return_value = [
            "eodata/Sentinel-2/MSI/L2A/2024/11/07/S2A_MSIL2A_20241107T113311_N0511"
            "_R080_T31VDG_20241107T123948.SAFE/GRANULE/L2A_T32TMT_A017394_"
            "20200705T101917/IMG_DATA/dummy.jp2"
        ]

        helper = HelperCdse(
            client_kwargs=dict(endpoint_url="https://eodata.dataspace.copernicus.eu"),
            key="xxx",
            secret="xxx",
        )

        item = self.item
        item.properties["processorVersion"] = "02.14"
        item_parsed = helper.parse_item(
            self.item, asset_names=["B01", "B02"], crs="EPSG:4326", spatial_res=0.001
        )
        self.assertIn("B01", item_parsed.assets)
        self.assertEqual(
            0, item_parsed.assets["B01"].extra_fields["raster:bands"][0]["offset"]
        )
        self.assertIn("B02", item_parsed.assets)
        self.assertEqual(
            (
                "eodata/Sentinel-2/MSI/L2A/2024/11/07/S2A_MSIL2A_20241107T113311_N0511"
                "_R080_T31VDG_20241107T123948.SAFE/GRANULE/L2A_Ttitle_id_A000000_"
                "20200705T101917/IMG_DATA/R60m/Ttitle_id_parts_B02_60m.jp2"
            ),
            item_parsed.assets["B02"].href,
        )
        item = self.item
        item.properties["processorVersion"] = "05.00"
        item_parsed = helper.parse_item(
            self.item, asset_names=["B01", "B02"], crs="EPSG:4326", spatial_res=0.001
        )
        self.assertIn("B01", item_parsed.assets)
        self.assertEqual(
            -0.1, item_parsed.assets["B01"].extra_fields["raster:bands"][0]["offset"]
        )
        self.assertIn("B02", item_parsed.assets)

    @patch("s3fs.S3FileSystem.glob")
    def test_get_data_access_params(self, mock_glob):
        mock_glob.return_value = [
            "eodata/Sentinel-2/MSI/L2A/2024/11/07/S2A_MSIL2A_20241107T113311_N0511"
            "_R080_T31VDG_20241107T123948.SAFE/GRANULE/L2A_T32TMT_A017394_"
            "20200705T101917/IMG_DATA/dummy.jp2"
        ]
        helper = HelperCdse(
            client_kwargs=dict(endpoint_url="https://eodata.dataspace.copernicus.eu"),
            key="xxx",
            secret="xxx",
        )
        item = self.item
        item.properties["processorVersion"] = "05.00"
        item_parsed = helper.parse_item(
            self.item, asset_names=["B01", "B02"], crs="EPSG:3035", spatial_res=20
        )
        data_access_params = helper.get_data_access_params(
            item_parsed, asset_names=["B01", "B02"], crs="EPSG:3035", spatial_res=20
        )
        self.assertEqual("B01", data_access_params["B01"]["name"])
        self.assertEqual("s3", data_access_params["B01"]["protocol"])
        self.assertEqual("eodata", data_access_params["B01"]["root"])
        self.assertEqual(
            (
                "Sentinel-2/MSI/L2A/2024/11/07/S2A_MSIL2A_20241107T113311_N0511_R080_"
                "T31VDG_20241107T123948.SAFE/GRANULE/L2A_T32TMT_A017394_20200705T101917"
                "/IMG_DATA/dummy.jp2"
            ),
            data_access_params["B01"]["fs_path"],
        )
