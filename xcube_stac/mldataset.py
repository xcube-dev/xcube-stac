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

import dask.array as da
import pystac
import rasterio
import rasterio.session
import rioxarray
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset, LazyMultiLevelDataset
from xcube.core.resampling import resample_in_space
from xcube.core.gridmapping import GridMapping

from .constants import LOG
from ._utils import get_resolutions_cog


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
        # combined_dataset = apply_scaling_nodata(combined_dataset, self._item)
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
        file_path = f"{access_params["root"]}/{access_params["fs_path"]}"
        self._file_path = file_path
        self._access_params = access_params
        self._open_params = open_params
        super().__init__(ds_id=file_path)

    def _get_num_levels_lazily(self) -> int:
        with rasterio.open(self._file_path) as rio_dataset:
            overviews = rio_dataset.overviews(1)
        return len(overviews) + 1

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        ds = rioxarray.open_rasterio(
            self._file_path,
            overview_level=index,
            chunks=dict(
                zip(("x", "y"), self._open_params.get("tile_size", (1024, 1024)))
            ),
            band_as_variable=True,
        )
        ds = ds.rename(dict(band_1=self._access_params["name"]))

        # resampling in space
        resolution = 10 * 2**index
        if not hasattr(self, f"_target_gm_{index}"):
            x_res = abs(ds.x.values[1] - ds.x.values[0])
            x_coords = xr.DataArray(
                da.arange(
                    ds.x.values[0] - (x_res / 2) + resolution,
                    ds.x.values[-1] + (x_res / 2) - (resolution / 2),
                    resolution,
                ),
                dims="x",
            )
            y_res = abs(ds.y.values[1] - ds.y.values[0])
            y_coords = xr.DataArray(
                da.arange(
                    ds.y.values[0] + (y_res / 2) - resolution,
                    ds.y.values[-1] - (y_res / 2) + (resolution / 2),
                    -resolution,
                ),
                dims="y",
            )
            crs = ds.spatial_ref.attrs["spatial_ref"]
            tile_size = self._open_params.get("tile_size", (1024, 1024))
            setattr(
                self,
                f"_target_gm_{index}",
                GridMapping.from_coords(
                    x_coords, y_coords, crs=crs, tile_size=tile_size
                ),
            )
        ds = resample_in_space(
            ds, target_gm=getattr(self, f"_target_gm_{index}"), encode_cf=False
        )
        return ds


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
        if "resolution" in open_params:
            del open_params["resolution"]
            LOG.warn(
                "The keyword 'resolution' is not considered when "
                "opening the data as multi-resolution dataset."
            )
        self._open_params = open_params
        self._items = sorted(items, key=lambda item: item.properties.get("datetime"))

        self._resolutions = get_resolutions_cog(
            self._items[0],
            asset_names=self._open_params.get("bands", None),
            crs=self._open_params.get("crs", None),
        )

        # open data for each resolution/overview level, so that odc.stac.load is
        # not called in the method _get_dataset_lazily()
        self._datasets = []
        for resolution in self._resolutions:
            ds = xr.Dataset()
            self._datasets.append(ds)
            # self._datasets.append(apply_scaling_nodata(ds, self._items))

    def _get_num_levels_lazily(self) -> int:
        return len(self._resolutions)

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        return self._datasets[index]
