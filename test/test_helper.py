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
import datetime

import pystac

from xcube_stac.helper import Helper


# class HelperTest(unittest.TestCase):
#     def test_list_assets_from_item(self):
#         geometry = {
#             "type": "Polygon",
#             "coordinates": [
#                 [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]
#             ],
#         }
#         bbox = [100.0, 0.0, 101.0, 1.0]
#         dt = datetime.datetime(2023, 1, 1, 0, 0, 0)
#         item = pystac.Item(
#             id="test_item", geometry=geometry, bbox=bbox, datetime=dt, properties={}
#         )
#         supported_format_ids = ["geotiff", "netcdf"]
#
#         asset_names = ["asset1", "asset2", "asset3"]
#         media_types = ["image/tiff", "application/zarr", "meta/xml"]
#         for asset_name, media_type in zip(asset_names, media_types):
#             asset_href = f"https://example.com/data/{asset_name}.tif"
#             asset = pystac.Asset(href=asset_href, media_type=media_type, roles=["data"])
#             item.add_asset(asset_name, asset)
#
#         helper = Helper()
#         list_assets = helper.list_assets_from_item(item)
#         self.assertCountEqual(
#             ["asset1", "asset2"], [asset.extra_fields["id"] for asset in list_assets]
#         )
#         list_assets = helper.list_assets_from_item(item, asset_names=["asset2"])
#         self.assertCountEqual(
#             ["asset2"], [asset.extra_fields["id"] for asset in list_assets]
#         )
