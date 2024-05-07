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
from xcube_stac.store import StacDataOpener
from xcube_stac.stac import Stac


class StacDataOpenerTest(unittest.TestCase):

    def setUp(self) -> None:
        stac_instance = Stac("url")
        self.opener = StacDataOpener(stac_instance)

    def test_get_open_data_params_schema(self):
        schema = self.opener.get_open_data_params_schema()
        self.assertIsInstance(schema, JsonObjectSchema)

    def test_open_data(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.opener.open_data("data_id1")
        self.assertEqual(
            "open_data() operation is not supported yet",
            f"{cm.exception}",
        )

    def test_describe_data(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.opener.describe_data("data_id1")
        self.assertEqual(
            "describe_data() operation is not supported yet",
            f"{cm.exception}",
        )
