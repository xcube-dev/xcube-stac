# The MIT License (MIT)
# Copyright (c) 2024-2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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

from abc import ABC, abstractmethod
from typing import Sequence

import pystac
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataTypeLike


class StacItemAccessor(ABC):
    """Provides methods for accessing the data of one STAC Item"""

    @abstractmethod
    def open_asset(
        self,
        item: pystac.Item,
        asset_name: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        """Open and return the dataset corresponding to a STAC asset."""

    @abstractmethod
    def open_item(
        self,
        item: pystac.Item,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        """Open and return the dataset corresponding to a STAC item."""


class StacARDCAccessor(ABC):
    """Provides methods for access multiple STAC Items and build an analysis ready
    data cube."""

    @abstractmethod
    def open_ardc(
        self,
        items: Sequence[pystac.Item],
        **open_params,
    ) -> xr.Dataset:
        """Open and return the dataset corresponding to a STAC item."""
