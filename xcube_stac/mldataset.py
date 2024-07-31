# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
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

from typing import Any, Optional

import odc.stac
import odc.geo
import pystac
import xarray as xr
from xcube.core.mldataset import (
    MultiLevelDataset,
    LazyMultiLevelDataset,
)

from .utils import (
    _apply_scaling_nodata,
    _get_resolutions_cog,
)


class SingleItemMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset for single item implementation (stack_mode=False).

    Args:
        ml_datasets: The multi-level datasets to be combined. At least
            two must be provided.
        data_id: Optional data identifier.
    """

    def __init__(
        self,
        ml_datasets: list[MultiLevelDataset],
        item: pystac.Item,
        data_id: Optional[str] = None,
    ):
        if not ml_datasets or len(ml_datasets) < 2:
            raise ValueError("ml_datasets must have at least two elements")
        super().__init__(ds_id=data_id)
        self._ml_datasets = ml_datasets
        self._data_id = data_id
        self._item = item

    def _get_num_levels_lazily(self) -> int:
        return self._ml_datasets[0].num_levels

    def _get_dataset_lazily(
        self, index: int, combiner_params: dict[str, Any]
    ) -> xr.Dataset:
        datasets = [ml_dataset.get_dataset(index) for ml_dataset in self._ml_datasets]
        combined_dataset = datasets[0].copy()
        for dataset in datasets[1:]:
            combined_dataset.update(dataset)
        combined_dataset = _apply_scaling_nodata(combined_dataset, self._item)
        return combined_dataset


class StackModeMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset for stack-mode.

    Args:
        data_id: data identifier
        items: list of items to be stacked
        open_params: opening parameters of odc.stack.load
    """

    def __init__(
        self,
        data_id: str,
        items: list[pystac.Item],
        **open_params: dict[str, Any],
    ):
        super().__init__(ds_id=data_id)
        self._data_id = data_id
        self._open_params = open_params
        self._items = sorted(items, key=lambda item: item.properties.get("datetime"))

        self._resolutions = _get_resolutions_cog(
            self._items[0],
            asset_names=self._open_params.get("asset_names", None),
            crs=self._open_params.get("crs", None),
        )

    def _get_num_levels_lazily(self):
        return len(self._resolutions)

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        ds = odc.stac.load(
            self._items,
            resolution=self._resolutions[index],
            **self._open_params,
        )
        ds = _apply_scaling_nodata(ds, self._items)
        return ds
