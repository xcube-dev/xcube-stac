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

from pystac import ItemCollection
import pytest
from xcube.util.jsonschema import JsonObjectSchema

from xcube_stac.stac import Stac


class StacTest(unittest.TestCase):

    def setUp(self):
        self.url_nonsearchable = (
            "https://raw.githubusercontent.com/stac-extensions/"
            "label/main/examples/multidataset/catalog.json"
        )
        self.url_searchable = "https://earth-search.aws.element84.com/v1"

    @pytest.mark.vcr()
    def test_get_open_data_params_schema(self):
        stac_instance = Stac(self.url_nonsearchable)
        schema = stac_instance.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("variable_names", schema.properties)
        self.assertIn("time_range", schema.properties)
        self.assertIn("bbox", schema.properties)
        self.assertIn("collections", schema.properties)

    @pytest.mark.vcr()
    def test_get_item_collection(self):
        stac_instance = Stac(self.url_nonsearchable)
        items, data_id_items = stac_instance.get_item_collection()
        data_id_items_expected = [
            "zanzibar-collection/znz001",
            "zanzibar-collection/znz029",
            "spacenet-buildings-collection/AOI_2_Vegas_img2636",
            "spacenet-buildings-collection/AOI_3_Paris_img1648",
            "spacenet-buildings-collection/AOI_4_Shanghai_img3344"
        ]
        self.assertIsInstance(items, ItemCollection)
        self.assertCountEqual(data_id_items, data_id_items_expected)
        self.assertEqual(len(items), len(data_id_items))

    @pytest.mark.vcr()
    def test_get_item_collection_open_params(self):
        stac_instance = Stac(self.url_nonsearchable)
        items, data_id_items = stac_instance.get_item_collection(
            collections="zanzibar-collection",
            bbox=[39.28, -5.74, 39.31, -5.72],
            time_range=["2019-04-23", "2019-04-24"]
        )
        data_id_items_expected = [
            "zanzibar-collection/znz001",
        ]
        self.assertIsInstance(items, ItemCollection)
        self.assertCountEqual(data_id_items, data_id_items_expected)
        self.assertEqual(len(items), len(data_id_items))

        items, data_id_items = stac_instance.get_item_collection(
            collections="zanzibar-collection",
            bbox=[39, -5., 41, -6],
            time_range=["2019-04-28", "2019-04-30"]
        )
        self.assertIsInstance(items, ItemCollection)
        self.assertEqual(len(items), 0)
        self.assertEqual(len(items), len(data_id_items))

    @pytest.mark.vcr()
    def test_get_item_collection_searchable_catalog(self):
        stac_instance = Stac(self.url_searchable)
        items, data_id_items = stac_instance.get_item_collection(
            variable_names="red",
            collections=["sentinel-2-l2a"],
            bbox=[9, 47, 10, 48],
            time_range=["2020-03-01", "2020-03-05"]
        )
        data_id_items_expected = [
            "sentinel-2-l2a/S2A_32TMT_20200305_0_L2A",
            "sentinel-2-l2a/S2A_32TNT_20200305_0_L2A",
            "sentinel-2-l2a/S2A_32UMU_20200305_0_L2A",
            "sentinel-2-l2a/S2A_32UNU_20200305_0_L2A",
            "sentinel-2-l2a/S2A_32TMT_20200302_1_L2A",
            "sentinel-2-l2a/S2A_32TMT_20200302_0_L2A",
            "sentinel-2-l2a/S2A_32TNT_20200302_1_L2A",
            "sentinel-2-l2a/S2A_32TNT_20200302_0_L2A",
            "sentinel-2-l2a/S2A_32UMU_20200302_1_L2A",
            "sentinel-2-l2a/S2A_32UMU_20200302_0_L2A",
            "sentinel-2-l2a/S2A_32UNU_20200302_1_L2A",
            "sentinel-2-l2a/S2A_32UNU_20200302_0_L2A"
        ]
        self.assertIsInstance(items, ItemCollection)
        self.assertCountEqual(data_id_items, data_id_items_expected)
        self.assertEqual(len(items), len(data_id_items))

    @pytest.mark.vcr()
    def test_is_datetime_in_range(self):
        class Item1():

            def __init__(self) -> None:
                self.properties = dict(
                    datetime="2024-05-02T09:19:38.543000Z"
                )

        class Item2():

            def __init__(self) -> None:
                self.properties = dict(
                    datetime="null",
                    start_datetime="2023-12-02T09:19:38.543000Z",
                    end_datetime="2024-05-02T09:19:38.543000Z"
                )

        item1_test_paramss = [
            ("2024-04-30", "2024-05-03", self.assertTrue),
            ("2024-04-26", "2024-05-02", self.assertFalse),
            ("2024-04-26", "2024-05-01", self.assertFalse)
        ]

        item2_test_paramss = [
            ("2024-05-05", "2024-05-08", self.assertFalse),
            ("2024-04-30", "2024-05-03", self.assertTrue),
            ("2024-04-26", "2024-04-29", self.assertTrue),
            ("2023-11-26", "2023-12-31", self.assertTrue),
            ("2023-11-26", "2023-11-30", self.assertFalse),
            ("2023-11-26", "2024-05-08", self.assertTrue),
        ]

        stac_instance = Stac(self.url_nonsearchable)

        item1 = Item1()
        for (time_start, time_end, fun) in item1_test_paramss:
            fun(
                stac_instance._is_datetime_in_range(
                    item1,
                    time_range=[time_start, time_end]
                )
            )

        item1 = Item2()
        for (time_start, time_end, fun) in item2_test_paramss:
            fun(
                stac_instance._is_datetime_in_range(
                    item1,
                    time_range=[time_start, time_end]
                )
            )

    @pytest.mark.vcr()
    def test_do_bboxes_intersect(self):
        class Item():

            def __init__(self) -> None:
                self.bbox = [0, 0, 1, 1]

        item_test_paramss = [
            (0, 0, 1, 1, self.assertTrue),
            (0.5, 0.5, 1.5, 1.5, self.assertTrue),
            (-0.5, -0.5, 0.5, 0.5, self.assertTrue),
            (1, 1, 2, 2, self.assertTrue),
            (2, 2, 3, 3, self.assertFalse)
        ]

        stac_instance = Stac(self.url_nonsearchable)

        item = Item()
        for (west, south, east, north, fun) in item_test_paramss:
            fun(
                stac_instance._do_bboxes_intersect(
                    item,
                    bbox=[west, south, east, north]
                )
            )
