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

from pystac import ItemCollection
from xcube.util.jsonschema import JsonObjectSchema
from xcube.core.store import DataStoreError
from xcube.core.store.store import new_data_store
from xcube_stac.constants import DATA_STORE_ID


class StacDataStoreTest(unittest.TestCase):

    def setUp(self):
        self.url = (
            "https://raw.githubusercontent.com/stac-extensions/"
            "label/main/examples/multidataset/catalog.json"
        )
        self.data_id = "zanzibar-collection/znz001/raster"

    @pytest.mark.vcr()
    def test_get_data_store_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        schema = store.get_data_store_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("url", schema.properties)
        self.assertIn("data_id_delimiter", schema.properties)
        self.assertIn("url", schema.required)

    @pytest.mark.vcr()
    def test_get_data_types(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        self.assertEqual(("dataset",), store.get_data_types())

    @pytest.mark.vcr()
    def test_get_data_types_for_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        self.assertEqual(
            ("dataset",),
            store.get_data_types_for_data(self.data_id)
        )

    @pytest.mark.vcr()
    def test_get_data_ids(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        with self.assertRaises(NotImplementedError) as cm:
            store.get_data_ids()
        self.assertEqual(
            "get_data_ids() operation is not supported yet",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_has_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        with self.assertRaises(NotImplementedError) as cm:
            store.has_data("data_id1")
        self.assertEqual(
            "has_data() operation is not supported yet",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_get_item_collection(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        items, data_id_items = store.get_item_collection(
            collections="zanzibar-collection"
        )
        data_id_items_expected = [
            "zanzibar-collection/znz001",
            "zanzibar-collection/znz029"
        ]
        self.assertIsInstance(items, ItemCollection)
        self.assertListEqual(data_id_items, data_id_items_expected)
        self.assertEqual(len(items), len(data_id_items))

    @pytest.mark.vcr()
    def test_describe_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        with self.assertRaises(NotImplementedError) as cm:
            store.describe_data(self.data_id)
        self.assertEqual(
            "describe_data() operation is not supported yet",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_get_data_opener_ids(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        self.assertEqual(
            ("dataset:zarr:stac",),
            store.get_data_opener_ids()
        )

    # TODO: add this test when has_data() works
    # @pytest.mark.vcr()
    # def test_get_data_opener_ids_optional_args(self):
    #     store = new_data_store(DATA_STORE_ID, url=self.url)
    #     with self.assertRaises(DataStoreError) as cm:
    #         store.get_data_opener_ids(data_id="wrong_data_id")
    #     self.assertEqual(
    #         "Data resource 'wrong_data_id' is not available.",
    #         f"{cm.exception}",
    #     )
    #     with self.assertRaises(DataStoreError) as cm:
    #         store.get_data_opener_ids(data_type=str)
    #     self.assertEqual(
    #         "Data type must be 'dataset', but got <class 'str'>",
    #         f"{cm.exception}",
    #     )

    @pytest.mark.vcr()
    def test_get_open_data_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        schema = store.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("variable_names", schema.properties)
        self.assertIn("time_range", schema.properties)
        self.assertIn("bbox", schema.properties)
        self.assertIn("collections", schema.properties)

    @pytest.mark.vcr()
    def test_open_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        with self.assertRaises(NotImplementedError) as cm:
            store.open_data(self.data_id)
        self.assertEqual(
            "open_data() operation is not supported yet",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_open_data_wrong_opener_id(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        with self.assertRaises(DataStoreError) as cm:
            store.open_data(self.data_id, opener_id="wrong_opener_id")
        self.assertEqual(
            "Data opener identifier must be 'dataset:zarr:stac', "
            "but got 'wrong_opener_id'",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_search_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        with self.assertRaises(NotImplementedError) as cm:
            store.search_data()
        self.assertEqual(
            "search_data() operation is not supported yet",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_get_search_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        with self.assertRaises(NotImplementedError) as cm:
            store.get_search_params_schema()
        self.assertEqual(
            "get_search_params_schema() operation is not supported yet",
            f"{cm.exception}",
        )
