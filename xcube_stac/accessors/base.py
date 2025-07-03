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

from typing import Sequence

import pystac
import xarray as xr
from xcube.core.mldataset import MultiLevelDataset, CombinedMultiLevelDataset
from xcube.core.store import new_data_store, DataStoreError, DataStore

from xcube_stac.accessor import StacItemAccessor
from xcube_stac.stac_extension.raster import apply_offset_scaling, get_stac_extension
from xcube_stac.constant import LOG
from xcube_stac._href_parse import decode_href
from xcube_atac.utils import (
    update_dict,
    list_assets_from_item,
    normalize_grid_mapping,
    rename_dataset,
)


class BaseStacItemAccessor(StacItemAccessor):
    """Provides methods for accessing the data of one general STAC Item"""

    def __init__(self, **storage_options_s3):
        self._https_store = None
        self._s3_store = None
        self._storage_options_s3 = storage_options_s3

    def open_asset(
        self, asset: pystac.Asset, **open_params
    ) -> xr.Dataset | MultiLevelDataset:
        protocol, root, fs_path, storage_options = decode_href(asset.href)
        if protocol == "https":
            store = self._get_store(protocol, root=root)
        elif protocol == "s3":
            storage_options = update_dict(
                self._storage_options_s3,
                storage_options,
                inplace=False,
            )
            store = self._get_store(protocol, root=root, **storage_options)
        else:
            raise DataStoreError(
                f"Neither 's3' nor 'https' could be derived from href {asset.href!r}."
            )
        return store.open_data(fs_path, **open_params)

    def open_item(
        self, item: pystac.Item, **open_params
    ) -> xr.Dataset | CombinedMultiLevelDataset:
        apply_scaling = open_params.pop("apply_scaling", False)
        asset_names = open_params.pop("asset_names", None)
        assets = list_assets_from_item(item, asset_names=asset_names)
        dss = [self.open_asset(asset, **open_params) for asset in assets]
        if isinstance(dss[0], xr.Dataset):
            return self._combiner_function(
                dss,
                item=item,
                assets=assets,
                apply_scaling=apply_scaling,
            )
        else:
            return CombinedMultiLevelDataset(
                dss,
                combiner_function=self._combiner_function,
                combiner_params=dict(
                    item=item,
                    assets=assets,
                    apply_scaling=apply_scaling,
                ),
            )

    def _get_store(self, protocol: str, root: str, **store_params) -> DataStore:
        property_name = f"_{protocol}_store"
        store = getattr(self, property_name, None)

        if store is None:
            store = new_data_store(protocol, root=root, **store_params)
        elif store.root != root:
            LOG.debug(
                "Initializing a new %r data store because root changed from %r to %r.",
                protocol,
                store.root,
                root,
            )
            store = new_data_store(protocol, root=root, **store_params)

        return store

    def _combiner_function(
        self,
        dss: Sequence[xr.Dataset | MultiLevelDataset],
        item: pystac.Item = None,
        assets: Sequence[pystac.Asset] = None,
        apply_scaling: bool = False,
    ) -> xr.Dataset:
        dss = [
            rename_dataset(ds, asset.extra_fields["xcube_stac:asset_id"])
            for (ds, asset) in zip(dss, assets)
        ]
        if apply_scaling:
            raster_version = get_stac_extension(item)
            dss = [
                apply_offset_scaling(ds, asset, raster_version)
                for (ds, asset) in zip(dss, assets)
            ]
        combined_ds = dss[0].copy()
        for ds in dss[1:]:
            combined_ds.update(ds)
        return normalize_grid_mapping(combined_ds)
