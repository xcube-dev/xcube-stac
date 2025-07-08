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
from xcube.core.mldataset import (
    CombinedMultiLevelDataset,
    MultiLevelDataset,
    BaseMultiLevelDataset,
)
from xcube.core.store import DataStore, DataStoreError, new_data_store
from xcube.util.jsonschema import JsonObjectSchema
from xcube_stac.utils import (
    list_assets_from_item,
    normalize_grid_mapping,
    rename_dataset,
    update_dict,
)

from xcube_stac.accessor import StacItemAccessor
from xcube_stac.constants import LOG, SCHEMA_APPLY_SCALING, SCHEMA_ASSET_NAMES
from xcube_stac.href_parse import decode_href
from xcube_stac.stac_extension.raster import apply_offset_scaling, get_stac_extension
from xcube_stac.version import version


class BaseStacItemAccessor(StacItemAccessor):
    """Provides methods for accessing the data of one general STAC item."""

    def __init__(self, catalog: pystac.Catalog, **storage_options_s3):
        self._storage_options_s3 = storage_options_s3
        self._catalog = catalog

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
                catalog=self._catalog,
                assets=assets,
                apply_scaling=apply_scaling,
            )
        else:
            if len(dss) == 1:
                # TODO: no scaling applied. Maybe insert make mldataset and use
                # TODO: combined cut of in combiner function.
                return dss[0]
            return CombinedMultiLevelDataset(
                dss,
                combiner_function=self._combiner_function,
                combiner_params=dict(
                    item=item,
                    catalog=self._catalog,
                    assets=assets,
                    apply_scaling=apply_scaling,
                ),
            )

    def get_open_data_params_schema(
        self, data_id: str = None, opener_id: str = None
    ) -> JsonObjectSchema:
        if opener_id is not None:
            store = new_data_store("https")
            params_schema = store.get_open_data_params_schema(opener_id=opener_id)
            params_properties = params_schema.properties
            params_required = params_schema.required
        else:
            params_properties = {}
            params_required = []

        return JsonObjectSchema(
            properties=dict(
                asset_names=SCHEMA_ASSET_NAMES,
                apply_scaling=SCHEMA_APPLY_SCALING,
                **params_properties,
            ),
            required=[] + params_required,
            additional_properties=True,
        )

    def _get_store(self, protocol: str, root: str, **storage_options) -> DataStore:
        property_name = f"_{protocol}_store"
        store = getattr(self, property_name, None)

        if store is None:
            store = new_data_store(protocol, root=root, storage_options=storage_options)
        elif store.root != root:
            LOG.debug(
                "Initializing a new %r data store because root changed from %r to %r.",
                protocol,
                store.root,
                root,
            )
            store = new_data_store(protocol, root=root, storage_options=storage_options)
        setattr(self, property_name, store)
        return store

    @staticmethod
    def _combiner_function(
        dss: Sequence[xr.Dataset | MultiLevelDataset],
        item: pystac.Item = None,
        catalog: pystac.Catalog = None,
        assets: Sequence[pystac.Asset] = None,
        apply_scaling: bool = False,
    ) -> xr.Dataset:
        if len(dss) > 1:
            dss = [
                rename_dataset(ds, asset.extra_fields["xcube:asset_id"])
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
        combined_ds.attrs = dict(
            stac_catalog_url=catalog.get_self_href(),
            stac_item_id=item.id,
            xcube_stac_version=version,
        )
        return normalize_grid_mapping(combined_ds)


class XcubeStacItemAccessor(BaseStacItemAccessor):
    """Provides methods for accessing the data of one STAC item published
    by xcube Server."""

    def open_asset(
        self, asset: pystac.Asset, **open_params
    ) -> xr.Dataset | MultiLevelDataset:
        data_store_params = asset.extra_fields["xcube:data_store_params"]
        root = data_store_params["root"]
        storage_options = data_store_params["storage_options"]
        fs_path = asset.extra_fields["xcube:open_data_params"]["data_id"]
        storage_options = update_dict(
            self._storage_options_s3,
            storage_options,
            inplace=False,
        )
        store = self._get_store("s3", root=root, **storage_options)
        return store.open_data(fs_path, **open_params)
