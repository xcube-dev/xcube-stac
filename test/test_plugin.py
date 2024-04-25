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

from xcube.util.extension import ExtensionRegistry
from xcube_stac.plugin import init_plugin


class XcubePluginTest(unittest.TestCase):
    def test_plugin(self):
        """Assert xcube extensions registered by xcube-stac"""
        registry = ExtensionRegistry()
        init_plugin(registry)
        self.assertEqual(
            {
                "xcube.core.store": {
                    "stac": {
                        "component": "<not loaded yet>",
                        "description": "STAC DataStore",
                        "name": "stac",
                        "point": "xcube.core.store",
                    }
                },
                "xcube.core.store.opener": {
                    "dataset:zarr:stac": {
                        "component": "<not loaded yet>",
                        "description": "xarray.Dataset from STAC API",
                        "name": "dataset:zarr:stac",
                        "point": "xcube.core.store.opener",
                    }
                },
            },
            registry.to_dict(),
        )
