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
from xcube_stac.store import StacDataOpener
from xcube_stac.stac import Stac


class StacDataOpenerTest(unittest.TestCase):

    def setUp(self):
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
        self.assertIn("variable_names", schema.properties)
        self.assertIn("time_range", schema.properties)
        self.assertIn("bbox", schema.properties)
        self.assertIn("collections", schema.properties)

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
