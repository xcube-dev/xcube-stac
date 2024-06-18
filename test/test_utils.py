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

import datetime
import unittest

import pystac
from xcube.core.store import DataStoreError

from xcube_stac.utils import (
    _convert_datetime2str,
    _convert_str2datetime,
    _do_bboxes_intersect,
    _get_opener_id,
    _is_datetime_in_range,
    _update_dict,
)


class UtilsTest(unittest.TestCase):

    def test_convert_datetime2str(self):
        dt = datetime.datetime(2024, 1, 1, 12, 00, 00)
        self.assertEqual("2024-01-01T12:00:00", _convert_datetime2str(dt))
        dt = datetime.date(2024, 1, 1)
        self.assertEqual("2024-01-01", _convert_datetime2str(dt))

    def test_convert_str2datetime(self):
        dt = datetime.datetime(2024, 1, 1, 12, 00, 00, tzinfo=datetime.timezone.utc)
        self.assertEqual(dt, _convert_str2datetime("2024-01-01T12:00:00.000000Z"))
        self.assertEqual(dt, _convert_str2datetime("2024-01-01T12:00:00"))

    def test_is_datetime_in_range(self):
        item1 = pystac.Item(
            "test_item",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=datetime.datetime(2024, 5, 1, 9, 19, 38),
            properties=dict(datetime="2024-05-02T09:19:38.000000Z"),
        )
        item2 = pystac.Item(
            "test_item",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=None,
            properties=dict(
                datetime="null",
                start_datetime="2023-12-02T09:19:38.543000Z",
                end_datetime="2024-05-02T09:19:38.543000Z",
            ),
        )
        item3 = pystac.Item(
            "test_item",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=datetime.datetime(2024, 5, 1, 9, 19, 38),
            properties=dict(),
        )

        item1_test_paramss = [
            ("2024-04-30", "2024-05-03", self.assertTrue),
            ("2024-04-26", "2024-05-02", self.assertFalse),
            ("2024-04-26", "2024-05-01", self.assertFalse),
        ]

        item2_test_paramss = [
            ("2024-05-05", "2024-05-08", self.assertFalse),
            ("2024-04-30", "2024-05-03", self.assertTrue),
            ("2024-04-26", "2024-04-29", self.assertTrue),
            ("2023-11-26", "2023-12-31", self.assertTrue),
            ("2023-11-26", "2023-11-30", self.assertFalse),
            ("2023-11-26", "2024-05-08", self.assertTrue),
        ]

        for time_start, time_end, fun in item1_test_paramss:
            fun(_is_datetime_in_range(item1, time_range=[time_start, time_end]))

        for time_start, time_end, fun in item2_test_paramss:
            fun(_is_datetime_in_range(item2, time_range=[time_start, time_end]))

        with self.assertRaises(DataStoreError) as cm:
            _is_datetime_in_range(
                item3, time_range=[item1_test_paramss[0][0], item1_test_paramss[0][1]]
            )
        self.assertEqual(
            "Either 'start_datetime' and 'end_datetime' or 'datetime' "
            "needs to be determined in the STAC item.",
            f"{cm.exception}",
        )

    def test_do_bboxes_intersect(self):
        item = pystac.Item(
            "test_item",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=datetime.datetime(2024, 1, 1, 12, 00, 00),
            properties={},
        )

        item_test_paramss = [
            (0, 0, 1, 1, self.assertTrue),
            (0.5, 0.5, 1.5, 1.5, self.assertTrue),
            (-0.5, -0.5, 0.5, 0.5, self.assertTrue),
            (1, 1, 2, 2, self.assertTrue),
            (2, 2, 3, 3, self.assertFalse),
        ]

        for west, south, east, north, fun in item_test_paramss:
            fun(_do_bboxes_intersect(item, bbox=[west, south, east, north]))

    def test_update_nested_dict(self):
        dic = dict(a=1, b=dict(c=3))
        dic_update = dict(d=1, b=dict(c=5, e=8))
        dic_expected = dict(a=1, d=1, b=dict(c=5, e=8))
        self.assertDictEqual(dic_expected, _update_dict(dic, dic_update))

    def test_get_opener_id(self):
        asset = pystac.Asset(
            href="https://path/to/data/resource",
            title="data",
            media_type="image/tiff; application=…profile=cloud-optimized",
        )
        formats = ["geotiff"]
        protocol = "https"
        self.assertEqual(
            "mldataset:geotiff:https", _get_opener_id(asset, formats, protocol)
        )
        self.assertEqual(
            "dataset:geotiff:https",
            _get_opener_id(asset, formats, protocol, opener_id="dataset:geotiff:https"),
        )
        with self.assertLogs("xcube.stac", level="WARNING") as cm:
            opener_id = _get_opener_id(
                asset, formats, protocol, opener_id="dataset:netcdf:https"
            )
        self.assertEqual("dataset:geotiff:https", opener_id)
        self.assertEqual(1, len(cm.output))
        msg = (
            "WARNING:xcube.stac:The format of the asset 'data' is 'geotiff'. "
            "The opener is changed from 'dataset:netcdf:https' "
            "to 'dataset:geotiff:https'."
        )
        self.assertEqual(msg, str(cm.output[-1]))

    def test_get_opener_id_zarr(self):
        asset = pystac.Asset(
            href="https://path/to/data/resource",
            title="data",
            media_type="application/zarr",
        )
        formats = ["geotiff", "zarr", "netcdf"]
        protocol = "s3"
        self.assertEqual("dataset:zarr:s3", _get_opener_id(asset, formats, protocol))
