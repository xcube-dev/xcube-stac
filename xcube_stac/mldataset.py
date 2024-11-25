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

import pystac
import rasterio
import rasterio.session
import rioxarray
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset, LazyMultiLevelDataset
from xcube.core.gridmapping import GridMapping

from .constants import LOG
from ._utils import rename_dataset
from ._utils import merge_datasets
from .stac_extension.raster import apply_offset_scaling


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
        assets: list[str],
        data_id: Optional[str] = None,
        target_gm: GridMapping = None,
        open_params: dict = None,
        attrs: dict = None,
    ):
        super().__init__(ds_id=data_id)
        self._ml_datasets = ml_datasets
        self._data_id = data_id
        self._item = item
        self._assets = assets
        self._target_gm = target_gm
        self._open_params = open_params
        self._attrs = attrs

    def _get_num_levels_lazily(self) -> int:
        return self._ml_datasets[0].num_levels

    def _get_dataset_lazily(
        self, index: int, combiner_params: dict[str, Any]
    ) -> xr.Dataset:
        datasets = []
        for ml_dataset, asset in zip(self._ml_datasets, self._assets):
            ds = ml_dataset.get_dataset(index)
            ds = rename_dataset(ds, asset)
            if self._open_params.get("apply_scaling", False):
                ds = apply_offset_scaling(ds, self._item, asset)
            datasets.append(ds)
        combined_dataset = merge_datasets(datasets, target_gm=self._target_gm)
        combined_dataset.attrs = self._attrs
        return combined_dataset


class Jp2MultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset for accessing .jp2 files.

    Args:
        data_id: data identifier
        items: list of items to be stacked
        open_params: opening parameters of odc.stack.load
    """

    def __init__(
        self,
        access_params: dict,
        **open_params: dict[str, Any],
    ):
        file_path = (
            f"{access_params["protocol"]}://{access_params["root"]}"
            f"/{access_params["fs_path"]}"
        )
        self._file_path = file_path
        self._access_params = access_params
        self._open_params = open_params
        super().__init__(ds_id=file_path)

    def _get_num_levels_lazily(self) -> int:
        with rasterio.open(self._file_path) as rio_dataset:
            overviews = rio_dataset.overviews(1)
        return len(overviews) + 1

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        return rioxarray.open_rasterio(
            self._file_path,
            overview_level=index - 1 if index > 0 else None,
            chunks=dict(x=1024, y=1024),
            band_as_variable=True,
        )
