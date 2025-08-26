# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
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

from xcube_stac.accessors import (
    guess_item_accessor,
    guess_ardc_accessor,
    list_ardc_data_ids,
    BaseStacItemAccessor,
    Sen2CdseStacItemAccessor,
    Sen3CdseStacItemAccessor,
    Sen2CdseStacArdcAccessor,
    Sen3CdseStacArdcAccessor,
    Sen2PlanetaryComputerStacItemAccessor,
    Sen2PlanetaryComputerStacArdcAccessor,
)
from xcube_stac.constants import (
    DATA_STORE_ID_CDSE,
    DATA_STORE_ID_CDSE_ARDC,
    DATA_STORE_ID_PC,
    DATA_STORE_ID_PC_ARDC,
)


class AccessorInitTest(unittest.TestCase):

    # ------------------------------------------------------------------
    # guess_item_accessor
    # ------------------------------------------------------------------

    def test_guess_item_accessor_cdse_l2a(self):
        result = guess_item_accessor(DATA_STORE_ID_CDSE, "sentinel-2-l2a")
        self.assertIs(result, Sen2CdseStacItemAccessor)

    def test_guess_item_accessor_cdse_l1c(self):
        result = guess_item_accessor(DATA_STORE_ID_CDSE, "sentinel-2-l1c")
        self.assertIs(result, Sen2CdseStacItemAccessor)

    def test_guess_item_accessor_cdse_s3(self):
        result = guess_item_accessor(DATA_STORE_ID_CDSE, "sentinel-3-syn-2-syn-ntc")
        self.assertIs(result, Sen3CdseStacItemAccessor)

    def test_guess_item_accessor_pc(self):
        result = guess_item_accessor(DATA_STORE_ID_PC, "sentinel-2-l2a")
        self.assertIs(result, Sen2PlanetaryComputerStacItemAccessor)

    def test_guess_item_accessor_base_for_unknown_store(self):
        result = guess_item_accessor("unknown-store", "random-data-id")
        self.assertIs(result, BaseStacItemAccessor)

    def test_guess_item_accessor_base_for_non_matching_id(self):
        result = guess_item_accessor(DATA_STORE_ID_CDSE, "non-matching-id")
        self.assertIs(result, BaseStacItemAccessor)

    # ------------------------------------------------------------------
    # guess_ardc_accessor
    # ------------------------------------------------------------------

    def test_guess_ardc_accessor_cdse_l2a(self):
        result = guess_ardc_accessor(DATA_STORE_ID_CDSE_ARDC, "sentinel-2-l2a")
        self.assertIs(result, Sen2CdseStacArdcAccessor)

    def test_guess_ardc_accessor_cdse_l1c(self):
        result = guess_ardc_accessor(DATA_STORE_ID_CDSE_ARDC, "sentinel-2-l1c")
        self.assertIs(result, Sen2CdseStacArdcAccessor)

    def test_guess_ardc_accessor_cdse_s3(self):
        result = guess_ardc_accessor(
            DATA_STORE_ID_CDSE_ARDC, "sentinel-3-syn-2-syn-ntc"
        )
        self.assertIs(result, Sen3CdseStacArdcAccessor)

    def test_guess_ardc_accessor_pc_l2a(self):
        result = guess_ardc_accessor(DATA_STORE_ID_PC_ARDC, "sentinel-2-l2a")
        self.assertIs(result, Sen2PlanetaryComputerStacArdcAccessor)

    def test_guess_ardc_accessor_raises_for_unknown_store(self):
        with self.assertRaises(NotImplementedError):
            guess_ardc_accessor("unknown-store", "sentinel-2-l2a")

    def test_guess_ardc_accessor_raises_for_non_matching_id(self):
        with self.assertRaises(NotImplementedError):
            guess_ardc_accessor(DATA_STORE_ID_CDSE_ARDC, "non-matching-id")

    # ------------------------------------------------------------------
    # list_ardc_data_ids
    # ------------------------------------------------------------------

    def test_list_ardc_data_ids_cdse(self):
        result = list_ardc_data_ids(DATA_STORE_ID_CDSE_ARDC)
        self.assertCountEqual(
            result, ["sentinel-2-l2a", "sentinel-2-l1c", "sentinel-3-syn-2-syn-ntc"]
        )

    def test_list_ardc_data_ids_pc(self):
        result = list_ardc_data_ids(DATA_STORE_ID_PC_ARDC)
        self.assertEqual(result, ["sentinel-2-l2a"])
