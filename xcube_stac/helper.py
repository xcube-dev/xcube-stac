from typing import Iterator, Union

import numpy as np
import pystac
import pystac_client.client
from xcube.core.store import DataStoreError
import s3fs

from .accessor import S3DataAccessor
from .accessor import S3Sentinel2DataAccessor
from .constants import MAP_CDSE_COLLECTION_FORMAT
from .constants import MLDATASET_FORMATS
from .constants import STAC_SEARCH_PARAMETERS
from .constants import STAC_SEARCH_PARAMETERS_STACK_MODE
from .constants import STAC_OPEN_PARAMETERS
from .constants import STAC_OPEN_PARAMETERS_STACK_MODE
from .constants import SCHEMA_PROCESSING_LEVEL
from .constants import SCHEMA_COLLECTIONS
from .constants import SCHEMA_SPATIAL_RES
from .sen2.constants import CDSE_SENITNEL_2_BANDS
from .sen2.constants import CDSE_SENTINEL_2_LEVEL_BAND_RESOLUTIONS
from .sen2.constants import CDSE_SENTINEL_2_MIN_RESOLUTIONS
from .sen2.constants import CDSE_SENITNEL_2_SCALE
from .sen2.constants import CDSE_SENITNEL_2_OFFSET_400
from .sen2.constants import CDSE_SENITNEL_2_NO_DATA
from ._href_parse import decode_href
from ._utils import get_format_id
from ._utils import get_format_from_path
from ._utils import is_valid_ml_data_type
from ._utils import list_assets_from_item
from ._utils import search_items
from ._utils import normalize_crs


class Helper:

    def __init__(self):
        self.supported_protocols = ["s3", "https"]
        self.supported_format_ids = ["netcdf", "zarr", "geotiff"]
        self.schema_open_params = STAC_OPEN_PARAMETERS
        self.schema_open_params_stack = STAC_OPEN_PARAMETERS_STACK_MODE
        self.schema_search_params = STAC_SEARCH_PARAMETERS
        self.schema_search_params_stack = STAC_SEARCH_PARAMETERS_STACK_MODE
        self.s3_accessor = S3DataAccessor

    def parse_item(self, item: pystac.Item, **open_params) -> pystac.Item:
        return item

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        assets = list_assets_from_item(
            item,
            asset_names=open_params.get("asset_names"),
            supported_format_ids=self.supported_format_ids,
        )
        data_access_params = {}
        for asset in assets:
            protocol, root, fs_path, storage_options = decode_href(asset.href)
            format_id = get_format_id(asset)
            data_access_params[asset.extra_fields["id"]] = dict(
                name=asset.extra_fields["id"],
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options=storage_options,
                format_id=format_id,
                href=asset.href,
            )
        return data_access_params

    def get_protocols(self, item: pystac.Item, **open_params) -> list[str]:
        params = self.get_data_access_params(item, **open_params)
        return list(np.unique([params[key]["protocol"] for key in params]))

    def get_format_ids(self, item: pystac.Item, **open_params) -> list[str]:
        params = self.get_data_access_params(item, **open_params)
        format_ids = list(np.unique([params[key]["format_id"] for key in params]))
        return [
            format_id
            for format_id in format_ids
            if format_id in self.supported_format_ids
        ]

    def is_mldataset_available(self, item: pystac.Item, **open_params) -> bool:
        format_ids = self.get_format_ids(item, **open_params)
        return all(format_id in MLDATASET_FORMATS for format_id in format_ids)

    def search_items(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        searchable: bool,
        **search_params,
    ) -> Iterator[pystac.Item]:
        return search_items(catalog, searchable, **search_params)


class HelperXcube(Helper):

    def __init__(self):
        super().__init__()
        self.supported_protocols = ["s3"]
        self.supported_format_ids = ["zarr", "levels"]

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        asset_names = open_params.get("asset_names")
        assets = list_assets_from_item(
            item,
            asset_names=asset_names,
            supported_format_ids=self.supported_format_ids,
        )
        opener_id_data_type = open_params.get("opener_id")
        if opener_id_data_type is not None:
            opener_id_data_type = opener_id_data_type.split(":")[0]
        if asset_names is None:
            if is_valid_ml_data_type(open_params.get("data_type")):
                assets = [assets[1]]
            elif is_valid_ml_data_type(opener_id_data_type):
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
        data_access_params = {}
        for asset in assets:
            protocol = asset.extra_fields["xcube:data_store_id"]
            data_store_params = asset.extra_fields["xcube:data_store_params"]
            root = data_store_params["root"]
            storage_options = data_store_params["storage_options"]
            fs_path = asset.extra_fields["xcube:open_data_params"]["data_id"]
            format_id = get_format_from_path(fs_path)
            data_access_params[asset.extra_fields["id"]] = dict(
                name=asset.extra_fields["id"],
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options=storage_options,
                format_id=format_id,
                href=asset.href,
            )
        return data_access_params

    def is_mldataset_available(self, item: pystac.Item, **open_params) -> bool:
        return True


class HelperCdse(Helper):

    def __init__(self, **storage_options_s3):
        super().__init__()
        self.supported_protocols = ["s3"]
        self.supported_format_ids = ["netcdf", "zarr", "geotiff", "jp2"]
        self.schema_open_params = dict(
            **STAC_OPEN_PARAMETERS, spatial_res=SCHEMA_SPATIAL_RES
        )
        open_params_stack = dict(
            **STAC_OPEN_PARAMETERS_STACK_MODE, processing_level=SCHEMA_PROCESSING_LEVEL
        )
        del open_params_stack["query"]
        self.schema_open_params_stack = open_params_stack
        self.schema_search_params = dict(
            **STAC_SEARCH_PARAMETERS_STACK_MODE,
            collections=SCHEMA_COLLECTIONS,
            processing_level=SCHEMA_PROCESSING_LEVEL,
        )
        self._fs = s3fs.S3FileSystem(
            anon=False,
            endpoint_url=storage_options_s3["client_kwargs"]["endpoint_url"],
            key=storage_options_s3["key"],
            secret=storage_options_s3["secret"],
        )
        self.s3_accessor = S3Sentinel2DataAccessor

    def parse_item(self, item: pystac.Item, **open_params) -> pystac.Item:
        processing_level = open_params.pop("processing_level", "L2A")
        open_params["asset_names"] = open_params.get(
            "asset_names", CDSE_SENITNEL_2_BANDS[processing_level]
        )
        href_base = item.assets["PRODUCT"].extra_fields["alternate"]["s3"]["href"][1:]
        res_want = open_params.get("spatial_res", CDSE_SENTINEL_2_MIN_RESOLUTIONS)
        if "crs" in open_params:
            target_crs = normalize_crs(open_params["crs"])
            if target_crs.is_geographic:
                res_want = open_params["spatial_res"] * 111320
        time_end = None
        for asset_name in open_params["asset_names"]:
            res_avail = CDSE_SENTINEL_2_LEVEL_BAND_RESOLUTIONS[processing_level][
                asset_name
            ]
            res_select = res_avail[np.argmin(abs(np.array(res_avail) - res_want))]
            if time_end is None:
                hrefs = self._fs.glob(
                    f"{href_base}/**/*_{asset_name}_{res_select}m.jp2"
                )
                assert len(hrefs) == 1, "No unique jp2 file found"
                href_mod = hrefs[0]
                time_end = hrefs[0].split("/IMG_DATA/")[0][-15:]
            else:
                id_parts = item.id.split("_")
                href_mod = (
                    f"{href_base}/GRANULE/L2A_T{item.properties["tileId"]}_"
                    f"A{item.properties["orbitNumber"]:06}_{time_end}/IMG_DATA/"
                    f"R{res_select}m/T{item.properties["tileId"]}_"
                    f"{id_parts[2]}_{asset_name}_{res_select}m.jp2"
                )
            if float(item.properties["processorVersion"]) >= 4.00:
                offset = CDSE_SENITNEL_2_OFFSET_400[asset_name]
            else:
                offset = 0
            item.assets[asset_name] = pystac.Asset(
                href_mod,
                asset_name,
                media_type="image/jp2",
                roles=["data"],
                extra_fields={
                    "cdse": True,
                    "raster:bands": [
                        dict(
                            nodata=CDSE_SENITNEL_2_NO_DATA,
                            scale=1 / CDSE_SENITNEL_2_SCALE[asset_name],
                            offset=offset / CDSE_SENITNEL_2_SCALE[asset_name],
                        )
                    ],
                },
            )
        # add asset for meta data for angles
        item.assets["granule_metadata"] = pystac.Asset(
            f"{href_base}/GRANULE/MTD_TL.xml",
            "granule_metadata",
            media_type="application/xml",
            roles=["metadata"],
            extra_fields={"cdse": True},
        )
        return item

    def get_data_access_params(self, item: pystac.Item, **open_params) -> dict:
        processing_level = open_params.pop("processing_level", "L2A")
        asset_names = open_params.get(
            "asset_names", CDSE_SENITNEL_2_BANDS[processing_level]
        )
        data_access_params = {}
        for asset_name in asset_names:
            protocol = "s3"
            href_components = item.assets[asset_name].href.split("/")
            root = href_components[0]
            instrument = href_components[1]
            format_id = MAP_CDSE_COLLECTION_FORMAT[instrument]
            fs_path = "/".join(href_components[1:])
            storage_options = {}
            data_access_params[asset_name] = dict(
                name=asset_name,
                protocol=protocol,
                root=root,
                fs_path=fs_path,
                storage_options=storage_options,
                format_id=format_id,
                href=item.assets[asset_name].href,
            )
        return data_access_params

    def get_protocols(self, item: pystac.Item, **open_params) -> list[str]:
        return ["s3"]

    def get_format_ids(self, item: pystac.Item, **open_params) -> list[str]:
        return ["jp2"]

    def is_mldataset_available(self, item: pystac.Item, **open_params) -> bool:
        return True

    def search_items(
        self,
        catalog: Union[pystac.Catalog, pystac_client.client.Client],
        searchable: bool,
        **search_params,
    ) -> Iterator[pystac.Item]:
        processing_level = search_params.pop("processing_level", "L2A")
        if "sortby" not in search_params:
            search_params["sortby"] = "+datetime"
        items = search_items(catalog, searchable, **search_params)
        for item in items:
            if not processing_level[1:] in item.properties["processingLevel"]:
                continue
            yield item
