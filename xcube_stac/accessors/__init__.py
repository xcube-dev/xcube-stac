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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Type

from xcube_stac.accessor import StacArdcAccessor, StacItemAccessor
from xcube_stac.constants import (
    DATA_STORE_ID_CDSE,
    DATA_STORE_ID_CDSE_ARDC,
    DATA_STORE_ID_PC,
    DATA_STORE_ID_PC_ARDC,
)
from .base import BaseStacItemAccessor
from .sen2 import (
    Sen2CdseStacArdcAccessor,
    Sen2CdseStacItemAccessor,
    Sen2PlanetaryComputerStacArdcAccessor,
    Sen2PlanetaryComputerStacItemAccessor,
)
from .sen3 import Sen3CdseStacArdcAccessor, Sen3CdseStacItemAccessor

ACCESSOR_MAPPING = {
    DATA_STORE_ID_CDSE: {
        "sentinel-2-l2a": Sen2CdseStacItemAccessor,
        "sentinel-2-l1c": Sen2CdseStacItemAccessor,
        "sentinel-3-syn-2-syn-ntc": Sen3CdseStacItemAccessor,
    },
    DATA_STORE_ID_CDSE_ARDC: {
        "sentinel-2-l2a": Sen2CdseStacArdcAccessor,
        "sentinel-2-l1c": Sen2CdseStacArdcAccessor,
        "sentinel-3-syn-2-syn-ntc": Sen3CdseStacArdcAccessor,
    },
    DATA_STORE_ID_PC: {
        "sentinel-2-l2a": Sen2PlanetaryComputerStacItemAccessor,
    },
    DATA_STORE_ID_PC_ARDC: {
        "sentinel-2-l2a": Sen2PlanetaryComputerStacArdcAccessor,
    },
}


def guess_item_accessor(store_id: str, data_id: str = None) -> Type[StacItemAccessor]:
    if store_id in ACCESSOR_MAPPING.keys():
        if data_id is not None:
            for key in ACCESSOR_MAPPING[store_id].keys():
                if key in data_id:
                    return ACCESSOR_MAPPING[store_id][key]
    return BaseStacItemAccessor


def guess_ardc_accessor(store_id: str, data_id: str = None) -> Type[StacArdcAccessor]:
    accesor = None
    if store_id in ACCESSOR_MAPPING.keys():
        accesor = ACCESSOR_MAPPING[store_id].get(data_id)
    if accesor is None:
        raise NotImplementedError(
            f"No ARDC accessor implemented for store_id {store_id!r} "
            f"and data_id {data_id!r}."
        )
    # noinspection PyTypeChecker
    return accesor


def list_ardc_data_ids(store_id: str) -> list:
    return list(ACCESSOR_MAPPING[store_id].keys())
