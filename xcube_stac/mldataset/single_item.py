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

from typing import Any

import pystac
import xarray as xr
from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import LazyMultiLevelDataset, MultiLevelDataset

from xcube_stac.accessor.s3 import S3DataAccessor
from xcube_stac.accessor.sen2 import S3Sentinel2DataAccessor
from xcube_stac.stac_extension.raster import apply_offset_scaling
from xcube_stac.utils import (
    clip_dataset_by_bbox,
    merge_datasets,
    normalize_grid_mapping,
    rename_dataset,
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
        access_params: dict,
        data_id: str | None = None,
        target_gm: GridMapping = None,
        open_params: dict = None,
        attrs: dict = None,
        item: pystac.Item = None,
        s3_accessor: S3Sentinel2DataAccessor | S3DataAccessor = None,
    ):
        super().__init__(ds_id=data_id)
        self._ml_datasets = ml_datasets
        self._data_id = data_id
        self._access_params = access_params
        self._target_gm = target_gm
        self._open_params = open_params
        self._attrs = attrs
        self._item = item
        self._s3_accessor = s3_accessor

    def _get_num_levels_lazily(self) -> int:
        return self._ml_datasets[0].num_levels

    def _get_dataset_lazily(
        self, index: int, combiner_params: dict[str, Any]
    ) -> xr.Dataset:
        datasets = []
        for ml_dataset, (asset_name, params) in zip(
            self._ml_datasets, self._access_params.items()
        ):
            ds = ml_dataset.get_dataset(index)
            ds = normalize_grid_mapping(ds)
            ds = clip_dataset_by_bbox(ds, **self._open_params)
            ds = rename_dataset(ds, params["name_origin"])
            if self._open_params.get("apply_scaling", False):
                ds[params["name_origin"]] = apply_offset_scaling(
                    ds[params["name_origin"]], params["item"], params["name"]
                )
            datasets.append(ds)
        combined_dataset = merge_datasets(datasets, target_gm=self._target_gm)
        if self._open_params.get("angles_sentinel2", False):
            combined_dataset = self._s3_accessor.add_sen2_angles(
                self._item, combined_dataset
            )
        combined_dataset.attrs = self._attrs
        return combined_dataset
