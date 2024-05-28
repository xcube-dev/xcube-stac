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

import pytest
import requests
from xcube.core.store import (
    DatasetDescriptor,
    DataStoreError
)
from xcube.core.store.store import new_data_store
from xcube.util.jsonschema import JsonObjectSchema

from xcube_stac.constants import DATA_STORE_ID


class StacDataStoreTest(unittest.TestCase):

    def setUp(self):
        self.url_nonsearchable = (
            "https://raw.githubusercontent.com/stac-extensions/"
            "label/main/examples/multidataset/catalog.json"
        )
        self.url_searchable = (
            "https://earth-search.aws.element84.com/v1"
        )
        self.url_time_range = (
            "https://s3.eu-central-1.wasabisys.com/stac/odse/catalog.json"
        )
        self.data_id_nonsearchable = "zanzibar/znz001.json"
        self.data_id_searchable = (
            "collections/sentinel-1-grd/items/"
            "S1A_EW_GRDM_1SDH_20240528T100847_20240528T100950_054070_06930C"
        )
        self.data_id_time_range = (
            "lcv_blue_landsat.glad.ard/lcv_blue_landsat.glad.ard_1999.12.02"
            "..2000.03.20/lcv_blue_landsat.glad.ard_1999.12.02..2000.03.20.json"
        )

    @pytest.mark.vcr()
    def test_get_data_store_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        schema = store.get_data_store_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("url", schema.properties)
        self.assertIn("url", schema.required)

    @pytest.mark.vcr()
    def test_get_data_types(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        self.assertEqual(("dataset",), store.get_data_types())

    @pytest.mark.vcr()
    def test_get_data_types_for_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        self.assertEqual(
            ("dataset",),
            store.get_data_types_for_data(self.data_id_nonsearchable)
        )

    @pytest.mark.vcr()
    def test_get_data_ids(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        data_ids = store.get_data_ids()
        data_ids_expected = [
            "zanzibar/znz001.json",
            "zanzibar/znz029.json",
            "spacenet-buildings/AOI_2_Vegas_img2636.json",
            "spacenet-buildings/AOI_3_Paris_img1648.json",
            "spacenet-buildings/AOI_4_Shanghai_img3344.json"
        ]
        self.assertCountEqual(data_ids_expected, data_ids)

    @pytest.mark.vcr()
    def test_get_data_ids_include_attrs(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        include_attrs = [
            "id", "bbox", "geometry", "properties", "links", "assets"
        ]
        data_id, attrs = next(store.get_data_ids(include_attrs=include_attrs))
        self.assertEqual(self.data_id_searchable, data_id)
        self.assertCountEqual(include_attrs, list(attrs.keys()))

    @pytest.mark.vcr()
    def test_get_data_ids_optional_args_empty_args(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        data_id, attrs = next(store.get_data_ids(include_attrs=["dtype"]))
        self.assertEqual("zanzibar/znz001.json", data_id)
        self.assertFalse(attrs)

    @pytest.mark.vcr()
    def test_has_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        self.assertTrue(store.has_data(self.data_id_nonsearchable))
        self.assertFalse(store.has_data(self.data_id_nonsearchable, data_type=str))

    @pytest.mark.vcr()
    def test_get_data_opener_ids(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        self.assertEqual(
            ("dataset:zarr:stac",),
            store.get_data_opener_ids()
        )

    @pytest.mark.vcr()
    def test_get_data_opener_ids_optional_args(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(DataStoreError) as cm:
            store.get_data_opener_ids(data_id="wrong_data_id")
        self.assertEqual(
            "Data resource 'wrong_data_id' is not available.",
            f"{cm.exception}",
        )
        with self.assertRaises(DataStoreError) as cm:
            store.get_data_opener_ids(data_type=str)
        self.assertEqual(
            "Data type must be 'dataset', but got <class 'str'>",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_get_open_data_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        schema = store.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("asset_names", schema.properties)

    @pytest.mark.vcr()
    def test_open_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)

        # open data without open_params
        assets = store.open_data(self.data_id_nonsearchable)
        self.assertTrue(1, len(assets))
        self.assertEqual("znz001_previewcog", assets[0].title)
        self.assertEqual(
            ("https://oin-hotosm.s3.amazonaws.com/5afeda152b6a08001185f11a/"
             "0/5afeda152b6a08001185f11b.tif"),
            assets[0].href
        )

        # open data with open_params
        assets = store.open_data(self.data_id_nonsearchable, asset_names=["raster"])
        self.assertTrue(1, len(assets))
        self.assertEqual("znz001_previewcog", assets[0].title)
        self.assertEqual(
            ("https://oin-hotosm.s3.amazonaws.com/5afeda152b6a08001185f11a/"
             "0/5afeda152b6a08001185f11b.tif"),
            assets[0].href
        )

    @pytest.mark.vcr()
    def test_open_data_wrong_opener_id(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(DataStoreError) as cm:
            store.open_data(
                self.data_id_nonsearchable,
                opener_id="wrong_opener_id"
            )
        self.assertEqual(
            "Data opener identifier must be 'dataset:zarr:stac', "
            "but got 'wrong_opener_id'",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_search_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        descriptors = list(store.search_data(
            collections="zanzibar-collection",
            bbox=[39.28, -5.74, 39.31, -5.72],
            time_range=["2019-04-23", "2019-04-24"]
        ))

        expected_descriptor = dict(
            data_id="zanzibar/znz001.json",
            data_type="dataset",
            bbox=[
                39.28919876472999,
                -5.743028283012867,
                39.31302874892266,
                -5.722212794937691
            ],
            time_range=["2019-04-23", None]
        )

        self.assertEqual(1, len(descriptors))
        self.assertIsInstance(descriptors[0], DatasetDescriptor)
        self.assertEqual(expected_descriptor, descriptors[0].to_dict())

    @pytest.mark.vcr()
    def test_search_data_searchable_catalog(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_searchable)
        descriptors = list(store.search_data(
            collections=["sentinel-2-l2a"],
            bbox=[9, 47, 10, 48],
            time_range=["2020-03-01", "2020-03-05"]
        ))

        prefix = "collections/sentinel-2-l2a/items/"
        data_ids_expected = [
            "S2A_32TMT_20200305_0_L2A", "S2A_32TNT_20200305_0_L2A",
            "S2A_32UMU_20200305_0_L2A", "S2A_32UNU_20200305_0_L2A",
            "S2A_32TMT_20200302_1_L2A", "S2A_32TMT_20200302_0_L2A",
            "S2A_32TNT_20200302_1_L2A", "S2A_32TNT_20200302_0_L2A",
            "S2A_32UMU_20200302_1_L2A", "S2A_32UMU_20200302_0_L2A",
            "S2A_32UNU_20200302_1_L2A", "S2A_32UNU_20200302_0_L2A"
        ]
        data_ids_expected = [prefix + id for id in data_ids_expected]

        expected_descriptor = dict(
            data_id=data_ids_expected[0],
            data_type="dataset",
            bbox=[
                7.662878883910047,
                46.85818510451771,
                9.130456971519783,
                47.85361872923358
            ],
            time_range=["2020-03-05", None]
        )

        self.assertEqual(12, len(descriptors))
        for d in descriptors:
            self.assertIsInstance(d, DatasetDescriptor)
        self.assertCountEqual(data_ids_expected, [d.data_id for d in descriptors])
        self.assertEqual(expected_descriptor, descriptors[0].to_dict())

    @pytest.mark.vcr()
    def test_search_data_time_range(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_time_range)
        descriptors = list(store.search_data(
            collections=["lcv_blue_landsat.glad.ard"],
            bbox=[-10, 40, 40, 70],
            time_range=["2000-01-01", "2000-04-01"]
        ))

        expected_descriptors = [
            dict(
                data_id=(
                    "lcv_blue_landsat.glad.ard/lcv_blue_landsat.glad.ard_1999"
                    ".12.02..2000.03.20/lcv_blue_landsat.glad.ard_1999.12.02"
                    "..2000.03.20.json"
                ),
                data_type="dataset",
                bbox=[
                    -23.550818268711048,
                    24.399543432891665,
                    63.352379098951936,
                    77.69295185585888
                ],
                time_range=["1999-12-02", "2000-03-20"]
            ),
            dict(
                data_id=(
                    "lcv_blue_landsat.glad.ard/lcv_blue_landsat.glad.ard_2000"
                    ".03.21..2000.06.24/lcv_blue_landsat.glad.ard_2000.03.21"
                    "..2000.06.24.json"
                ),
                data_type="dataset",
                bbox=[
                    -23.550818268711048,
                    24.399543432891665,
                    63.352379098951936,
                    77.69295185585888
                ],
                time_range=["2000-03-21", "2000-06-24"]
            )
        ]

        self.assertEqual(2, len(descriptors))
        for d in descriptors:
            self.assertIsInstance(d, DatasetDescriptor)
        self.assertEqual(expected_descriptors, [d.to_dict() for d in descriptors])

    @pytest.mark.vcr()
    def test_get_search_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        schema = store.get_search_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("time_range", schema.properties)
        self.assertIn("bbox", schema.properties)
        self.assertIn("collections", schema.properties)

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

        store = new_data_store(
            DATA_STORE_ID,
            url=self.url_nonsearchable
        )

        item1 = Item1()
        for (time_start, time_end, fun) in item1_test_paramss:
            fun(
                store._is_datetime_in_range(
                    item1,
                    time_range=[time_start, time_end]
                )
            )

        item1 = Item2()
        for (time_start, time_end, fun) in item2_test_paramss:
            fun(
                store._is_datetime_in_range(
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

        store = new_data_store(
            DATA_STORE_ID,
            url=self.url_nonsearchable
        )

        item = Item()
        for (west, south, east, north, fun) in item_test_paramss:
            fun(
                store._do_bboxes_intersect(
                    item,
                    bbox=[west, south, east, north]
                )
            )

    def test_access_item_failed(self):
        store = new_data_store(DATA_STORE_ID, url=self.url_nonsearchable)
        with self.assertRaises(requests.exceptions.HTTPError) as cm:
            store._access_item(self.data_id_nonsearchable.replace("z", "s"))
        self.assertIn(
            "404 Client Error: Not Found for url",
            f"{cm.exception}"
        )
