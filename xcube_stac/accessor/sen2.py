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

from typing import Callable
import time

import boto3
import dask
import numpy as np
import pyproj
import pystac
import rasterio.session
import rioxarray
import xarray as xr
import xmltodict
from xcube.core.chunk import chunk_dataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DataTypeLike

from .._href_parse import decode_href
from .._utils import get_gridmapping
from .._utils import get_spatial_dims
from .._utils import is_valid_ml_data_type
from .._utils import wrapper_resample_in_space
from ..constants import LOG
from ..mldataset.jp2 import Jp2MultiLevelDataset
from ..stack import mosaic_spatial_along_time_take_first
from ..stack import mosaic_spatial_take_first


SENITNEL2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]
SENITNEL2_L2A_BANDS = SENITNEL2_BANDS + ["AOT", "SCL", "WVP"]
SENITNEL2_L2A_BANDS.remove("B10")
SENITNEL2_L2A_BAND_RESOLUTIONS = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
    "AOT": 10,
    "SCL": 20,
    "WVP": 10,
}
SENTINEL2_MIN_RESOLUTIONS = 10
SENTINEL2_REGEX_ASSET_NAME = "^[A-Z]{3}_[0-9]{2}m$"


class S3Sentinel2DataAccessor:
    """Implementation of the data accessor supporting
    the jp2 format of Sentinel-2 data via the AWS S3 protocol.
    """

    def __init__(self, root: str, storage_options: dict):
        self._root = root
        self.session = rasterio.session.AWSSession(
            aws_unsigned=storage_options["anon"],
            endpoint_url=storage_options["client_kwargs"]["endpoint_url"].split("//")[
                1
            ],
            aws_access_key_id=storage_options["key"],
            aws_secret_access_key=storage_options["secret"],
        )
        self.env = rasterio.env.Env(session=self.session, AWS_VIRTUAL_HOSTING=False)
        # keep the rasterio environment open so that the data can be accessed
        # when plotting or writing the data
        self.env = self.env.__enter__()
        # dask multi-threading needs to be turned off, otherwise the GDAL
        # reader for JP2 raises error.
        dask.config.set(scheduler="single-threaded")
        # need boto2 client to read xml meta data remotely
        self.s3_boto = boto3.client(
            "s3",
            endpoint_url=storage_options["client_kwargs"]["endpoint_url"],
            aws_access_key_id=storage_options["key"],
            aws_secret_access_key=storage_options["secret"],
            region_name="default",
        )

    def close(self):
        if self.env is not None:
            LOG.debug("Exit rasterio.env.Env for CDSE data access.")
            self.env.__exit__()
        self.env = None

    def __del__(self):
        self.close()

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
        if opener_id is None:
            opener_id = ""
        if "tile_size" in open_params:
            LOG.info(
                "The parameter tile_size is set to (1024, 1024), which is the "
                "native chunk size of the jp2 files in the Sentinel-2 archive."
            )
        if is_valid_ml_data_type(data_type) or opener_id.split(":")[0] == "mldataset":
            return Jp2MultiLevelDataset(access_params, **open_params)
        else:
            return rioxarray.open_rasterio(
                (
                    f"{access_params['protocol']}://{access_params['root']}/"
                    f"{access_params['fs_path']}"
                ),
                chunks=dict(x=1024, y=1024),
                band_as_variable=True,
            )

    def add_sen2_angles(self, item: pystac.Item, ds: xr.Dataset) -> xr.Dataset:
        return _add_sen2_angles(self._read_meta_data, item, ds)

    def add_sen2_angles_stack(
        self, grouped_items: xr.DataArray, ds: xr.Dataset
    ) -> xr.Dataset:
        return _add_sen2_angles_stack(self._read_meta_data, grouped_items, ds)

    def _read_meta_data(self, item: pystac.Item):
        # read xml file and parse to dict
        href = item.assets["granule_metadata"].href
        protocol, root, fs_path, storage_options = decode_href(href)
        response = self.s3_boto.get_object(Bucket=root, Key=fs_path)
        xml_content = response["Body"].read().decode("utf-8")
        return xmltodict.parse(xml_content)


class FileSentinel2DataAccessor:
    """Implementation of the data accessor supporting
    the jp2 format of Sentinel-2 data via the file protocol,
    used on Creodias VMs.
    """

    def __init__(self, root: str, storage_options: dict):
        self._root = root
        # # dask multi-threading needs to be turned off, otherwise the GDAL
        # # reader for JP2 raises error.
        # dask.config.set(scheduler="single-threaded")

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
        if opener_id is None:
            opener_id = ""
        if "tile_size" in open_params:
            LOG.info(
                "The parameter tile_size is set to (1024, 1024), which is the "
                "native chunk size of the jp2 files in the Sentinel-2 archive."
            )
        if is_valid_ml_data_type(data_type) or opener_id.split(":")[0] == "mldataset":
            return NotImplemented("Multi-level datasets are not implemented.")
        else:
            fs_path = f"/{access_params['root']}/{access_params['fs_path']}"
            attempt = 0
            max_retries = 3
            delay = 2
            while attempt < max_retries:
                try:
                    return rioxarray.open_rasterio(
                        fs_path,
                        chunks=dict(x=1024, y=1024),
                        band_as_variable=True,
                    )
                except Exception as e:
                    LOG.error(
                        f"An error occurred when opening {fs_path!r}: {e}. "
                        f"Retrying in {delay}sec..."
                    )
                    attempt += 1
                    time.sleep(delay)

    def add_sen2_angles(self, item: pystac.Item, ds: xr.Dataset) -> xr.Dataset:
        return _add_sen2_angles(self._read_meta_data, item, ds)

    def add_sen2_angles_stack(
        self, grouped_items: xr.DataArray, ds: xr.Dataset
    ) -> xr.Dataset:
        return _add_sen2_angles_stack(self._read_meta_data, grouped_items, ds)

    def _read_meta_data(self, item: pystac.Item):
        # read xml file and parse to dict
        href = item.assets["granule_metadata"].href
        protocol, root, fs_path, storage_options = decode_href(href)
        with open(f"/{root}/{fs_path}", "r", encoding="utf-8") as file:
            xml_content = file.read()
        return xmltodict.parse(xml_content)


def _add_sen2_angles(
    read_meta_data: Callable[[pystac.Item], dict], item: pystac.Item, ds: xr.Dataset
) -> xr.Dataset:
    xml_dict = read_meta_data(item)

    # read angles from xml file and add to dataset
    band_names = _get_band_names_from_dataset(ds)
    ds_angles = _get_sen2_angles(xml_dict, band_names)
    ds_angles = _rename_spatial_axis(ds_angles)
    for key in ["solar_angle", "viewing_angle"]:
        ds[key] = ds_angles[key]
    return ds


def _add_sen2_angles_stack(
    read_meta_data: Callable[[pystac.Item], dict],
    grouped_items: xr.DataArray,
    ds: xr.Dataset,
) -> xr.Dataset:
    # create target grid mapping, native resolution of 5000 is kept since the
    # angles from the xml meta data, which needs to be done eager
    crs = pyproj.CRS.from_cf(ds.spatial_ref.attrs)
    if crs.is_geographic:
        spatial_res = 5000 / 111320
    else:
        spatial_res = 5000
    y_coord, x_coord = get_spatial_dims(ds)
    bbox = [
        ds[x_coord][0].item(),
        ds[y_coord][0].item(),
        ds[x_coord][-1].item(),
        ds[y_coord][-1].item(),
    ]
    if bbox[3] < bbox[1]:
        y_min, y_max = bbox[3], bbox[1]
        bbox[1], bbox[3] = y_min, y_max
    target_gm = get_gridmapping(bbox, spatial_res, crs)

    # read out angles from all items and mosaic and stack them
    band_names = _get_band_names_from_dataset(ds)
    list_ds_tiles = []
    for tile_id in grouped_items.tile_id.values:
        list_ds_time = []
        idx_remove_dt = []
        for dt_idx, dt in enumerate(grouped_items.time.values):
            list_ds_idx = []
            for idx in grouped_items.idx.values:
                item = grouped_items.sel(tile_id=tile_id, time=dt, idx=idx).item()
                if not item:
                    continue
                try:
                    xml_dict = read_meta_data(item)
                    ds_item = _get_sen2_angles(xml_dict, band_names)
                except Exception as e:
                    LOG.error(
                        f"An error occurred: {e}. Meta data "
                        f"{item.assets['granule_metadata'].href} "
                        f"could not be opened."
                    )
                    continue
                list_ds_idx.append(ds_item)
            if not list_ds_idx:
                idx_remove_dt.append(dt_idx)
                continue
            else:
                ds_time = mosaic_spatial_take_first(list_ds_idx)
                list_ds_time.append(ds_time)
        ds_tile = xr.concat(list_ds_time, dim="time", join="exact")
        np_datetimes_sel = [
            value
            for idx, value in enumerate(grouped_items.time.values)
            if idx not in idx_remove_dt
        ]
        ds_tile = ds_tile.assign_coords(coords=dict(time=np_datetimes_sel))
        list_ds_tiles.append(
            chunk_dataset(
                wrapper_resample_in_space(ds_tile, target_gm),
                chunk_sizes={
                    "time": 1,
                    "angle": -1,
                    "band": -1,
                    target_gm.xy_dim_names[0]: -1,
                    target_gm.xy_dim_names[1]: -1,
                },
            )
        )
    ds_angles = mosaic_spatial_along_time_take_first(
        list_ds_tiles, grouped_items.time.values
    )

    # add the angles to the datacube
    ds_angles = _rename_spatial_axis(ds_angles)
    for key in ["solar_angle", "viewing_angle"]:
        ds[key] = ds_angles[key]
    return ds


def _get_sen2_angles(xml_dict: dict, band_names: list[str]) -> xr.Dataset:
    # read out solar and viewing angles
    geocode = xml_dict["n1:Level-2A_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]
    ULX = float(geocode["Geoposition"][0]["ULX"])
    ULY = float(geocode["Geoposition"][0]["ULY"])
    y = np.arange(ULY, ULY - 5000 * 23, -5000) - 2500
    x = np.arange(ULX, ULX + 5000 * 23, 5000) + 2500

    angles = xml_dict["n1:Level-2A_Tile_ID"]["n1:Geometric_Info"]["Tile_Angles"]
    map_bandid_name = {idx: name for idx, name in enumerate(SENITNEL2_BANDS)}
    band_names = band_names + ["solar"]
    detector_ids = np.unique(
        [
            int(angle["@detectorId"])
            for angle in angles["Viewing_Incidence_Angles_Grids"]
        ]
    )

    da = xr.DataArray(
        np.full(
            (len(band_names), len(detector_ids), 2, len(x), len(y)),
            np.nan,
            dtype=np.float64,
        ),
        dims=["band", "detector_id", "angle", "y", "x"],
        coords=dict(
            band=band_names,
            detector_id=detector_ids,
            angle=["Zenith", "Azimuth"],
            x=x,
            y=y,
        ),
    )

    # Each band has multiple detectors, so we have to go through all of them
    # and save them in a list to later do a nanmean
    for detector_angles in angles["Viewing_Incidence_Angles_Grids"]:
        band_id = int(detector_angles["@bandId"])
        band_name = map_bandid_name[band_id]
        if band_name not in band_names:
            continue
        detector_id = int(detector_angles["@detectorId"])
        for angle in da.angle.values:
            da.loc[band_name, detector_id, angle] = _get_angle_values(
                detector_angles, angle
            )
    # Do the same for the solar angles
    for angle in da.angle.values:
        da.loc["solar", detector_ids[0], angle] = _get_angle_values(
            angles["Sun_Angles_Grid"], angle
        )
    # Apply nanmean along detector ID axis
    da = da.mean(dim="detector_id", skipna=True)
    ds = xr.Dataset()
    ds["solar_angle"] = da.sel(band="solar").drop_vars("band")
    ds["viewing_angle"] = da.isel(band=slice(None, -1))
    crs = pyproj.CRS.from_epsg(geocode["HORIZONTAL_CS_CODE"].replace("EPSG:", ""))
    ds = ds.assign_coords(dict(spatial_ref=xr.DataArray(0, attrs=crs.to_cf())))

    return ds


def _get_angle_values(values_list: dict, angle: str) -> np.ndarray:
    values = values_list[angle]["Values_List"]["VALUES"]
    array = np.array([row.split(" ") for row in values]).astype(float)
    return array


def _get_band_names_from_dataset(ds: xr.Dataset) -> list[str]:
    band_names = [
        str(key).split("_")[0] for key in ds.keys() if str(key).startswith("B")
    ]
    return [name for name in SENITNEL2_BANDS if name in band_names]


def _rename_spatial_axis(ds: xr.Dataset) -> xr.Dataset:
    x_coord, y_coord = get_spatial_dims(ds)
    return ds.rename({y_coord: f"angle_{y_coord}", x_coord: f"angle_{x_coord}"})
