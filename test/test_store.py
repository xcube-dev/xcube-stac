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

from xcube.util.jsonschema import JsonObjectSchema
from xcube.core.store.store import new_data_store
from xcube_stac.store import StacDataOpener
from xcube_stac.stac import Stac
from xcube_stac.constants import DATA_STORE_ID


class StacDataOpenerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.url = (
            "https://raw.githubusercontent.com/stac-extensions/"
            "label/main/examples/multidataset/catalog.json"
        )
        self.data_id = "zanzibar-collection/znz001/raster"

    @pytest.mark.vcr()
    def test_get_open_data_params_schema(self):
        opener = StacDataOpener(Stac(self.url))
        schema = opener.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)

    @pytest.mark.vcr()
    def test_open_data(self):
        opener = StacDataOpener(Stac(self.url))
        with self.assertRaises(NotImplementedError) as cm:
            opener.open_data(self.data_id)
        self.assertEqual(
            "open_data() operation is not supported yet",
            f"{cm.exception}",
        )

    @pytest.mark.vcr()
    def test_describe_data(self):
        opener = StacDataOpener(Stac(self.url))
        with self.assertRaises(NotImplementedError) as cm:
            opener.describe_data(self.data_id)
        self.assertEqual(
            "describe_data() operation is not supported yet",
            f"{cm.exception}",
        )


class StacDataStoreTest(unittest.TestCase):

    def setUp(self) -> None:
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
        self.assertIn("collection_prefix", schema.properties)
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
    def test_get_data_ids(self) -> None:
        store = new_data_store(DATA_STORE_ID, url=self.url)
        data_ids = store.get_data_ids()
        data_ids_expected = [
            "zanzibar-collection/znz001/raster",
            "zanzibar-collection/znz029/raster",
            "spacenet-buildings-collection/AOI_2_Vegas_img2636/raster",
            "spacenet-buildings-collection/AOI_3_Paris_img1648/raster",
            "spacenet-buildings-collection/AOI_4_Shanghai_img3344/raster"
        ]
        for (data_id, data_id_expected) in zip(data_ids, data_ids_expected):
            self.assertEqual(data_id, data_id_expected)

    @pytest.mark.vcr()
    def test_get_data_ids_optional_args(self) -> None:
        # test collection_prefix and data_id_delimiter
        store = new_data_store(
            DATA_STORE_ID,
            url=self.url,
            collection_prefix="zanzibar-collection",
            data_id_delimiter=":"
        )
        data_ids = store.get_data_ids()
        data_ids_expected = [
            "znz001:raster",
            "znz029:raster"
        ]
        for (data_id, data_id_expected) in zip(data_ids, data_ids_expected):
            self.assertEqual(data_id, data_id_expected)

    @pytest.mark.vcr()
    def test_has_data(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        assert store.has_data(self.data_id)

    @pytest.mark.vcr()
    def test_has_data_optional_args(self):
        store = new_data_store(
            DATA_STORE_ID,
            url=self.url,
            collection_prefix="zanzibar-collection",
            data_id_delimiter=":"
        )
        assert store.has_data("znz001:raster")
        assert not store.has_data("znz001/raster")

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

    @pytest.mark.vcr()
    def test_get_open_data_params_schema(self):
        store = new_data_store(DATA_STORE_ID, url=self.url)
        schema = store.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)

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
