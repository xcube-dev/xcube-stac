# The MIT License (MIT)
# Copyright (c) 2024-2026 by the xcube development team and contributors
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

import unittest
import datetime

import pystac

from xcube_stac.accessors.hls import fix_utm_hemisphere


class HlsStacItemAccessorTest(unittest.TestCase):

    @staticmethod
    def make_item(bbox, proj_code):
        return pystac.Item(
            id="test-item",
            geometry=None,
            bbox=bbox,
            datetime=datetime.datetime.fromisoformat("20200101T11:23:23"),
            properties={"proj:code": proj_code},
        )

    def test_fix_utm_hemisphere_northern(self):
        item = self.make_item(
            bbox=[500000, 10, 600000, 20],  # center latitude = 15
            proj_code="EPSG:32733",  # incorrectly southern
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:code"], "EPSG:32633")

    def test_fix_utm_hemisphere_southern(self):
        item = self.make_item(
            bbox=[500000, -20, 600000, -10],  # center latitude = -15
            proj_code="EPSG:32633",  # incorrectly northern
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:code"], "EPSG:32733")

    def test_fix_utm_hemisphere_equator_is_northern(self):
        item = self.make_item(
            bbox=[500000, -1, 600000, 1],  # center latitude = 0
            proj_code="EPSG:32733",
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:code"], "EPSG:32633")

    def test_fix_utm_hemisphere_keeps_correct_code(self):
        item = self.make_item(
            bbox=[500000, 10, 600000, 20],
            proj_code="EPSG:32633",
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:code"], "EPSG:32633")

    def test_fix_utm_hemisphere_preserves_zone(self):
        item = self.make_item(
            bbox=[500000, -10, 600000, -5],
            proj_code="EPSG:32621",
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:code"], "EPSG:32721")
