from abc import abstractmethod
import os
from typing import Iterable

import numpy as np
import pystac
from xcube.core.store import (
    DATASET_TYPE,
    DataTypeLike,
    DataStoreError,
    MULTI_LEVEL_DATASET_TYPE,
)

from .constants import CDSE_MAP_INSTRUEMNT_FORMAT
from .constants import MAP_FILE_EXTENSION_FORMAT
from .constants import MAP_MIME_TYP_FORMAT
from .constants import MLDATASET_FORMATS
from .constants import LOG
from ._href_parse import decode_href
from ._utils import get_format_from_path
from ._utils import list_assets_from_item
from ._utils import update_dict
from ._utils import is_valid_ml_data_type


class StacAsset:

    def __init__(
        self,
        name: str,
        href: str,
        protocol: str,
        root: str,
        fs_path: str,
        storage_options: dict,
        format_id: str,
    ):
        self._name = name
        self._href = href
        self._protocol = protocol
        self._root = root
        self._fs_path = fs_path
        self._storage_options = storage_options
        self._format_id = format_id
        if format_id in MLDATASET_FORMATS:
            self._data_types = (MULTI_LEVEL_DATASET_TYPE.alias, DATASET_TYPE.alias)
        else:
            self._data_types = DATASET_TYPE.alias

    @property
    def name(self) -> str:
        return self._name

    @property
    def href(self) -> str:
        return self._href

    @property
    def protocol(self) -> str:
        return self._protocol

    @property
    def root(self) -> str:
        return self._root

    @property
    def fs_path(self) -> str:
        return self._fs_path

    @property
    def storage_options(self) -> dict:
        return self._storage_options

    @property
    def format_id(self) -> str:
        return self._format_id

    @property
    def data_types(self) -> tuple[str]:
        return self._data_types

    @classmethod
    @abstractmethod
    def from_pystac_asset(
        cls, asset: pystac.Asset, storage_options: dict
    ) -> "StacAsset":
        """Build xcube-stac specific asset from pystac asset object."""


class GeneralStacAsset(StacAsset):

    @classmethod
    def from_pystac_asset(
        cls, asset: pystac.Asset, storage_options: dict
    ) -> "GeneralStacAsset":
        protocol, root, fs_path, storage_options2 = decode_href(asset.href)
        storage_options = update_dict(storage_options, storage_options2)
        format_id = cls.get_format_id(asset)
        return cls(
            asset.extra_fields["id"],
            asset.href,
            protocol,
            root,
            fs_path,
            storage_options,
            format_id,
        )

    @staticmethod
    def get_format_id(asset: pystac.Asset) -> str:
        if hasattr(asset, "media_type"):
            format_id = MAP_MIME_TYP_FORMAT.get(asset.media_type.split("; ")[0])
        else:
            _, file_extension = os.path.splitext(asset.href)
            format_id = MAP_FILE_EXTENSION_FORMAT.get(file_extension)
        if format_id is None:
            raise DataStoreError(
                f"No format_id found for asset {asset.extra_fields['id']}"
            )
        return format_id


class CdseStacAsset(StacAsset):

    @classmethod
    def from_pystac_asset(
        cls, asset: pystac.Asset, storage_options: dict
    ) -> "CdseStacAsset":
        href = asset.extra_fields["alternate"]["s3"]["href"][1:]
        href_components = href.split("/")
        root = href_components[0]
        instrument = href_components[1]
        fs_path = "/".join(href_components[2:])
        protocol = "s3"
        format_id = CDSE_MAP_INSTRUEMNT_FORMAT[instrument]
        return cls(
            asset.extra_fields["id"],
            asset.href,
            protocol,
            root,
            fs_path,
            storage_options,
            format_id,
        )


class XcubeStacAsset(StacAsset):

    @classmethod
    def from_pystac_asset(
        cls, asset: pystac.Asset, storage_options: dict
    ) -> "XcubeStacAsset":
        protocol = asset.extra_fields["xcube:data_store_id"]
        data_store_params = asset.extra_fields["xcube:data_store_params"]
        root = data_store_params["root"]
        storage_options = data_store_params["storage_options"]
        fs_path = asset.extra_fields["xcube:open_data_params"]["data_id"]
        format_id = get_format_from_path(fs_path)
        return cls(
            asset.extra_fields["id"],
            asset.href,
            protocol,
            root,
            fs_path,
            storage_options,
            format_id,
        )


class StacItem:

    def __init__(self, assets: Iterable[StacAsset], item: pystac.Item):
        self._assets = assets
        self._item = item
        self._format_ids = list(np.unique([asset.format_id for asset in assets]))
        self._protocols = list(np.unique([asset.protocol for asset in assets]))

    @property
    def assets(self) -> Iterable[StacAsset]:
        return self._assets

    @property
    def item(self) -> pystac.Item:
        return self._item

    @property
    def format_ids(self) -> list[str]:
        return self._format_ids

    @property
    def protocols(self) -> list[str]:
        return self._protocols

    @classmethod
    def from_pystac_item(
        cls,
        item: pystac.Item,
        storage_options: dict,
        asset_names: list[str] = None,
        data_type: DataTypeLike = None,
    ) -> "StacItem":
        """Get internal stac item object form pystac item object"""

    @staticmethod
    @abstractmethod
    def xasset() -> StacAsset:
        """Return xcube stac asset class specific to the STAC catalog"""

    @staticmethod
    @abstractmethod
    def supported_protocols() -> list[str]:
        """Return supported protocols."""

    @staticmethod
    @abstractmethod
    def supported_format_ids() -> list[str]:
        """Return supported format IDs."""

    def is_mldataset_available(self):
        return len(self._format_ids) == 0 and self._format_ids[0] in MLDATASET_FORMATS


class GeneralStacItem(StacItem):

    @classmethod
    def from_pystac_item(
        cls,
        item: pystac.Item,
        storage_options: dict,
        asset_names: list[str] = None,
        data_type: DataTypeLike = None,
    ) -> "StacItem":
        assets = list_assets_from_item(item, asset_names)
        return cls(
            [
                cls.xasset().from_pystac_asset(asset, storage_options)
                for asset in assets
            ],
            item,
        )

    @staticmethod
    def xasset():
        return GeneralStacAsset

    @staticmethod
    def supported_protocols():
        return ["s3", "https"]

    @staticmethod
    def supported_format_ids():
        return ["netcdf", "zarr", "geotiff"]


class XcubeStacItem(StacItem):
    @classmethod
    def from_pystac_item(
        cls,
        item: pystac.Item,
        storage_options: dict,
        asset_names: list[str] = None,
        data_type: DataTypeLike = None,
    ) -> "StacItem":
        assets = list_assets_from_item(item, asset_names)
        if asset_names is None:
            if is_valid_ml_data_type(data_type):
                assets = [assets[1]]
            else:
                assets = [assets[0]]
        elif "analytic_multires" in asset_names and "analytic" in asset_names:
            raise DataStoreError(
                "Xcube server publishes data resources as 'dataset' and "
                "'mldataset' under the asset names 'analytic' and "
                "'analytic_multires'. Please select only one asset in "
                "<asset_names> when opening the data."
            )
        return cls(
            [
                cls.xasset().from_pystac_asset(asset, storage_options)
                for asset in assets
            ],
            item,
        )

    @staticmethod
    def xasset():
        return XcubeStacAsset

    @staticmethod
    def supported_protocols():
        return ["s3"]

    @staticmethod
    def supported_format_ids():
        return ["zarr", "levels"]


class CdseStacItem(StacItem):

    @classmethod
    def from_pystac_item(
        cls,
        item: pystac.Item,
        storage_options: dict,
        asset_names: list[str] = None,
        data_type: DataTypeLike = None,
    ) -> "StacItem":
        if (
            asset_names is not None
            and len(asset_names) != 1
            and asset_names[0] == "PRODUCT"
        ):
            asset_names = ["PRODUCT"]
            LOG.warn(
                "In the CDSE Stac Catalog, the asset_names field is consistently "
                "set to ['PRODUCT'] and will be overwritten."
            )
        assets = list_assets_from_item(item, asset_names)
        return cls(
            [
                cls.xasset().from_pystac_asset(asset, storage_options)
                for asset in assets
            ],
            item,
        )

    @staticmethod
    def xasset():
        return CdseStacAsset

    @staticmethod
    def supported_protocols():
        return ["s3"]

    @staticmethod
    def supported_format_ids():
        return ["netcdf", "zarr", "geotiff", "jp2"]
