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

import os
import unittest

from xcube_stac.stac import Stac
from unittest.mock import patch, MagicMock


class StacTest(unittest.TestCase):

    def setUp(self):
        url = "https://earth-search.aws.element84.com/v1"
        self.stac = Stac(url)
        self._open_params = dict(
            intersects=dict(type="Point", coordinates=[-105.78, 35.79]),
            collections=["sentinel-2-l2a"],
            datetime="2020-03-01/2020-03-03"
        )


    def test_open_dataset(self):
        open_params = 
        

        # Mock the response from cm.open_dataset
        mock_dataset = MagicMock()
        mock_open_dataset.return_value = mock_dataset
        cmems_instance = Cmems()
        result = cmems_instance.open_dataset("dataset1")
        self.assertEqual(result, mock_dataset)

        # Testing with a non-existing dataset
        mock_open_dataset.side_effect = KeyError("Dataset not found")
        result = cmems_instance.open_dataset("non_existing_dataset")
        self.assertIsNone(result)

    @patch("click.confirm", return_value=True)
    def test_open_data_for_not_exsiting_dataset(self, mock_confirm):
        cmems = Cmems()
        self.assertIsNone(
            cmems.open_dataset("dataset-bal-analysis-forecast" "-wav-hourly"),
            "Expected the method to return None for a " "non-existing dataset",
        )
