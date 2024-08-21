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

from typing import Union

import rioxarray
import s3fs
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataTypeLike
from xcube.core.store import new_data_store

from ._utils import is_valid_ml_data_type
from .constants import LOG
from .stac_objects import StacAsset
from .mldataset import Jp2MultiLevelDataset


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
        asset: StacAsset,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        if asset.format_id == "netcdf":
            if is_valid_ml_data_type(data_type):
                LOG.warn(
                    f"No data opener found for format {asset.format_id!r} and data type "
                    f"{data_type!r}. Data type is changed to the default data type "
                    "'dataset'."
                )
            fs_path = f"https://{self._root}/{asset.fs_path}#mode=bytes"
            return xr.open_dataset(fs_path, chunks={})
        else:
            return self._https_accessor.open_data(
                data_id=asset.fs_path,
                opener_id=opener_id,
                data_type=data_type,
                **open_params,
            )


class S3DataAccessor:
    """Implementation of the data accessor supporting
    the zarr, geotiff and netcdf format via the AWS S3 protocol.
    """

    def __init__(
        self,
        root: str,
        storage_options: dict,
    ):
        self._root = root
        self._storage_options = storage_options
        if "anon" not in storage_options:
            storage_options["anon"] = True
        self._s3_accessor = new_data_store(
            "s3",
            root=root,
            read_only=True,
            storage_options=storage_options,
        )

    @property
    def root(self) -> str:
        return self._root

    def open_data(
        self,
        asset: StacAsset,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        return self._s3_accessor.open_data(
            data_id=asset.fs_path,
            opener_id=opener_id,
            data_type=data_type,
            **open_params,
        )


class S3Sentinel2DataAccessor:
    """Implementation of the data accessor supporting
    the jp2 format  of Sentinel-2 data via the AWS S3 protocol.
    """

    def __init__(
        self,
        root: str,
        storage_options: dict,
    ):
        self._root = root
        self._storage_options = storage_options
        source_storage_options = dict(
            endpoint_url=self._storage_options["client_kwargs"]["endpoint_url"],
            anon=self._storage_options["anon"],
            key=self._storage_options["key"],
            secret=self._storage_options["secret"],
        )
        self._s3_accessor = s3fs.S3FileSystem(**source_storage_options)

    @property
    def root(self) -> str:
        return self._root

    def open_data(
        self,
        asset: StacAsset,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> Union[xr.Dataset, MultiLevelDataset]:
        if opener_id is None:
            opener_id = ""
        if is_valid_ml_data_type(data_type) or opener_id.split(":")[0] == "mldataset":
            ds = Jp2MultiLevelDataset(self._s3_accessor, asset, **open_params)
        else:
            ds = Jp2MultiLevelDataset(self._s3_accessor, asset, **open_params)
            ds = ds.get_dataset(open_params.get("overview_level"))
        return ds
