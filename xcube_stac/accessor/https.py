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

import xarray as xr
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataTypeLike
from xcube.core.store import new_data_store

from .._utils import is_valid_ml_data_type
from ..constants import LOG


class HttpsDataAccessor:
    """Implementation of the data accessor supporting
    the zarr, geotiff and netcdf format via the https protocol.
    """

    def __init__(self, root: str):
        self._root = root
        self._https_accessor = new_data_store("https", root=root, read_only=True)

    @property
    def root(self) -> str:
        return self._root

    def open_data(
        self,
        access_params: dict,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset | MultiLevelDataset:
        if access_params["format_id"] == "netcdf":
            if is_valid_ml_data_type(data_type):
                LOG.warn(
                    f"No data opener found for format {access_params['format_id']!r} "
                    f"and data type {data_type!r}. Data type is changed to the default "
                    f"data type 'dataset'."
                )
            fs_path = f"https://{self._root}/{access_params['fs_path']}#mode=bytes"
            return xr.open_dataset(fs_path, chunks={})
        else:
            return self._https_accessor.open_data(
                data_id=access_params["fs_path"],
                opener_id=opener_id,
                data_type=data_type,
                **open_params,
            )
