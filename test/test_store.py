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

from xcube.util.jsonschema import JsonObjectSchema
from xcube_stac.store import StacDataStore


class StacDataStoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.store = StacDataStore(url="url")

    def test_get_data_store_params_schema(self):
        schema = self.store.get_data_store_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)
        self.assertIn("url", schema.properties)
        self.assertIn("collection_prefix", schema.properties)
        self.assertIn("data_id_delimiter", schema.properties)
        self.assertIn("url", schema.required)

    def test_get_data_types(self):
        self.assertEqual(("dataset",), self.store.get_data_types())

    def test_get_data_types_for_data(self):
        self.assertEqual(
            ("dataset",),
            self.store.get_data_types_for_data("data_id1")
        )

    def test_get_data_ids(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.store.get_data_ids()
        self.assertEqual(
            "get_data_ids() operation is not supported yet",
            f"{cm.exception}",
        )

    def test_has_data(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.store.has_data("data_id1")
        self.assertEqual(
            "has_data() operation is not supported yet",
            f"{cm.exception}",
        )

    def test_describe_data(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.store.describe_data("data_id1")
        self.assertEqual(
            "describe_data() operation is not supported yet",
            f"{cm.exception}",
        )

    def test_get_data_opener_ids(self):
        self.assertEqual(
            ("dataset:zarr:stac",),
            self.store.get_data_opener_ids()
        )

    def test_get_open_data_params_schema(self):
        schema = self.store.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)

    def test_open_data(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.store.open_data("data_id1")
        self.assertEqual(
            "open_data() operation is not supported yet",
            f"{cm.exception}",
        )

    def test_search_data(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.store.search_data()
        self.assertEqual(
            "search_data() operation is not supported yet",
            f"{cm.exception}",
        )

    def test_get_search_params_schema(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.store.get_search_params_schema()
        self.assertEqual(
            "get_search_params_schema() operation is not supported yet",
            f"{cm.exception}",
        )
