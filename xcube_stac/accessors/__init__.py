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

from xcube_stac.accessor import StacArdcAccessor, StacItemAccessor

from .base import BaseStacItemAccessor, XcubeStacItemAccessor
from .sen2 import Sen2CdseStacArdcAccessor, Sen2CdseStacItemAccessor

ITEM_ACCESSOR_MAPPING = {
    "sentinel-2-l2a": Sen2CdseStacItemAccessor,
    "xcube": XcubeStacItemAccessor,
}
ARDC_ACCESSOR_MAPPING = {"sentinel-2-l2a": Sen2CdseStacArdcAccessor}


def guess_item_accessor(data_id: str = None) -> StacItemAccessor:
    if data_id is not None:
        for key in ITEM_ACCESSOR_MAPPING.keys():
            if key in data_id:
                return ITEM_ACCESSOR_MAPPING[key]
    return BaseStacItemAccessor


def guess_ardc_accessor(data_id: str) -> StacArdcAccessor:
    accesor = ARDC_ACCESSOR_MAPPING.get(data_id)
    if accesor is None:
        raise NotImplementedError(
            f"No ARDC accessor implemented for data_id {data_id!r}."
        )
    return accesor
