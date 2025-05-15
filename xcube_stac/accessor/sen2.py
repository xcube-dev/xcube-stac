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

from collections import defaultdict

import boto3
import dask
import dask.array as da
import numpy as np
import pyproj
import pystac
import rasterio.session
import rioxarray
import xarray as xr
import xmltodict
from xcube.core.gridmapping import GridMapping
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.resampling import affine_transform_dataset, reproject_dataset
from xcube.core.store import DataTypeLike

from xcube_stac.constants import LOG, TILE_SIZE
from xcube_stac.mldataset.jp2 import Jp2MultiLevelDataset
from xcube_stac.stac_extension.raster import apply_offset_scaling_stack_mode
from xcube_stac.utils import (
    add_nominal_datetime,
    clip_dataset_by_bbox,
    get_gridmapping,
    is_valid_ml_data_type,
    merge_datasets,
    mosaic_spatial_take_first,
    reproject_bbox,
)

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
SENTINEL2_FILL_VALUE = 0
SENITNEL2_L2A_BANDS.remove("B10")
SENTINEL2_BAND_RESOLUTIONS = np.array([10, 20, 60])
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
        if is_valid_ml_data_type(data_type) or opener_id.split(":")[0] == "mldataset":
            return Jp2MultiLevelDataset(access_params, **open_params)
        else:
            return rioxarray.open_rasterio(
                (
                    f"{access_params['protocol']}://{access_params['root']}/"
                    f"{access_params['fs_path']}"
                ),
                chunks={},
                band_as_variable=True,
            )

    def groupby_solar_day(self, items: list[pystac.Item]) -> xr.DataArray:
        items = add_nominal_datetime(items)

        # get dates and tile IDs of the items
        dates = []
        tile_ids = []
        proc_versions = []
        for item in items:
            dates.append(item.properties["datetime_nominal"].date())
            tile_ids.append(item.properties["grid:code"])
            proc_versions.append(self._get_processing_version(item))
        dates = np.unique(dates)
        tile_ids = np.unique(tile_ids)
        proc_versions = np.unique(proc_versions)[::-1]

        # sort items by date and tile ID into a data array
        grouped = xr.DataArray(
            np.empty((len(dates), len(tile_ids), 2, len(proc_versions)), dtype=object),
            dims=("time", "tile_id", "idx", "proc_version"),
            coords=dict(
                time=dates, tile_id=tile_ids, idx=[0, 1], proc_version=proc_versions
            ),
        )
        for idx, item in enumerate(items):
            date = item.properties["datetime_nominal"].date()
            tile_id = item.properties["grid:code"]
            proc_version = self._get_processing_version(item)
            if not grouped.sel(
                time=date, tile_id=tile_id, idx=0, proc_version=proc_version
            ).values:
                grouped.loc[date, tile_id, 0, proc_version] = item
            elif not grouped.sel(
                time=date, tile_id=tile_id, idx=1, proc_version=proc_version
            ).values:
                grouped.loc[date, tile_id, 1, proc_version] = item
            else:
                item0 = grouped.sel(
                    time=date, tile_id=tile_id, idx=0, proc_version=proc_version
                ).item()
                item1 = grouped.sel(
                    time=date, tile_id=tile_id, idx=1, proc_version=proc_version
                ).item()
                LOG.warn(
                    "More that two items found for datetime and tile ID: "
                    f"[{item0.id}, {item1.id}, {item.id}]. Only the first tow items "
                    "are considered."
                )

        # take the latest processing version
        da_arr = grouped.data
        mask = da_arr != None
        proc_version_idx = np.argmax(mask, axis=-1)
        da_arr_select = np.take_along_axis(
            da_arr, proc_version_idx[..., np.newaxis], axis=-1
        )[..., 0]
        grouped = xr.DataArray(
            da_arr_select,
            dims=("time", "tile_id", "idx"),
            coords=dict(time=dates, tile_id=tile_ids, idx=[0, 1]),
        )

        # replace date by datetime form first item
        dts = []
        for date in grouped.time.values:
            next_item = next(
                value
                for value in grouped.sel(time=date, idx=0).values
                if value is not None
            )
            dts.append(
                np.datetime64(
                    next_item.properties["datetime_nominal"].replace(tzinfo=None)
                ).astype("datetime64[ns]")
            )
        grouped = grouped.assign_coords(time=dts)

        return grouped

    def _get_processing_version(self, item: pystac.Item) -> float:
        return float(
            item.properties.get(
                "processing:version",
                item.properties.get("s2:processing_baseline", "1.0"),
            )
        )

    def generate_cube(self, access_params: xr.DataArray, **open_params) -> xr.Dataset:
        utm_tile_id = defaultdict(list)
        for tile_id in access_params.tile_id.values:
            item = next(
                value["item"]
                for value in access_params.sel(tile_id=tile_id, idx=0).values.ravel()
                if value is not None
            )
            crs = item.assets["AOT_10m"].extra_fields["proj:code"]
            utm_tile_id[crs].append(tile_id)

        list_ds_utm = []
        for crs, tile_ids in utm_tile_id.items():
            access_params_sel = access_params.sel(tile_id=tile_ids)
            list_ds_utm.append(self._sort_in_utm(access_params_sel, crs, **open_params))

        ds_final = _merge_utm_zones(list_ds_utm, **open_params)
        if open_params.get("apply_scaling", True):
            ds_final = apply_offset_scaling_stack_mode(ds_final, access_params)
        if open_params.get("angles_sentinel2", False):
            ds_final = self.add_sen2_angles_stack(
                ds_final, access_params, utm_tile_id, **open_params
            )

        return ds_final

    def _sort_in_utm(
        self,
        access_params_sel: xr.DataArray,
        crs_utm: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset:
        items_bbox = _get_bounding_box(access_params_sel)
        final_bbox = reproject_bbox(open_params["bbox"], open_params["crs"], crs_utm)

        list_ds_asset = []
        for asset_name in access_params_sel.asset_name.values:
            asset_ds = _create_empty_dataset(
                access_params_sel, asset_name, items_bbox, final_bbox
            )
            for dt_idx, dt in enumerate(access_params_sel.time.values):
                for tile_id in access_params_sel.tile_id.values:
                    list_ds_idx = []
                    for idx in access_params_sel.idx.values:
                        params = access_params_sel.sel(
                            tile_id=tile_id, asset_name=asset_name, time=dt, idx=idx
                        ).item()
                        if not params:
                            continue
                        try:
                            ds = self.open_data(
                                params, opener_id, data_type, **open_params
                            )
                        except Exception as e:
                            LOG.error(
                                f"An error occurred: {e}. Data could not be opened "
                                f"with parameters {params}"
                            )
                            continue
                        ds = clip_dataset_by_bbox(ds, final_bbox)
                        if any(size == 0 for size in ds.sizes.values()):
                            continue
                        list_ds_idx.append(ds)
                    if not list_ds_idx:
                        continue
                    ds = mosaic_spatial_take_first(list_ds_idx)
                    asset_ds = self._sort_in(asset_ds, asset_name, ds, dt_idx)
            list_ds_asset.append(asset_ds)

        return merge_datasets(list_ds_asset)

    def _sort_in(self, asset_ds, asset_name, ds, dt_idx):
        xmin = asset_ds.indexes["x"].get_loc(ds.x[0].item())
        xmax = asset_ds.indexes["x"].get_loc(ds.x[-1].item())
        ymin = asset_ds.indexes["y"].get_loc(ds.y[0].item())
        ymax = asset_ds.indexes["y"].get_loc(ds.y[-1].item())
        asset_ds[asset_name][dt_idx, ymin : ymax + 1, xmin : xmax + 1] = ds["band_1"]
        return asset_ds

    def get_sen2_angles(self, item: pystac.Item, ds: xr.Dataset) -> xr.Dataset:
        # read xml file and parse to dict
        href = item.assets["granule_metadata"].href
        protocol, remain = href.split("://")
        root = "eodata"
        fs_path = "/".join(remain.split("/")[1:])
        response = self.s3_boto.get_object(Bucket=root, Key=fs_path)
        xml_content = response["Body"].read().decode("utf-8")
        xml_dict = xmltodict.parse(xml_content)

        # read angles from xml file and add to dataset
        band_names = _get_band_names_from_dataset(ds)
        ds_angles = _get_sen2_angles(xml_dict, band_names)

        return ds_angles

    def add_sen2_angles(
        self, item: pystac.Item, ds: xr.Dataset, **open_params
    ) -> xr.Dataset:
        target_gm = GridMapping.from_dataset(ds)
        ds_angles = self.get_sen2_angles(item, ds)
        ds_angles = _resample_dataset_soft(
            ds_angles,
            target_gm,
            fill_value=np.nan,
            interpolation="bilinear",
            **open_params,
        )
        ds = _add_angles(ds, ds_angles)
        return ds

    def add_sen2_angles_stack(
        self,
        ds_final: xr.Dataset,
        access_params: xr.DataArray,
        utm_tile_id: dict,
        **open_params,
    ) -> xr.Dataset:
        target_gm = GridMapping.from_dataset(ds_final)

        list_ds_tiles = []
        for tile_id in access_params.tile_id.values:
            list_ds_time = []
            idx_remove_dt = []
            for dt_idx, dt in enumerate(access_params.time.values):
                list_ds_idx = []
                for idx in access_params.idx.values:
                    params = next(
                        (
                            value
                            for value in access_params.sel(
                                tile_id=tile_id, time=dt, idx=idx
                            ).values.flatten()
                            if value is not None
                        ),
                        None,
                    )
                    if not params:
                        continue
                    try:
                        ds = self.get_sen2_angles(params["item"], ds_final)
                    except Exception as e:
                        LOG.error(
                            f"An error occurred: {e}. Meta data "
                            f"{params['item'].assets['granule_metadata'].href} "
                            f"could not be opened."
                        )
                        continue
                    list_ds_idx.append(ds)
                if not list_ds_idx:
                    idx_remove_dt.append(dt_idx)
                    continue
                else:
                    ds_time = mosaic_spatial_take_first(list_ds_idx)
                    list_ds_time.append(ds_time)
            ds_tile = xr.concat(list_ds_time, dim="time", join="exact")
            np_datetimes_sel = [
                value
                for idx, value in enumerate(access_params.time.values)
                if idx not in idx_remove_dt
            ]
            ds_tile = ds_tile.assign_coords(coords=dict(time=np_datetimes_sel))
            list_ds_tiles.append(
                _resample_dataset_soft(
                    ds_tile,
                    target_gm,
                    fill_value=np.nan,
                    interpolation="bilinear",
                    **open_params,
                )
            )
        ds_angles = mosaic_spatial_take_first(list_ds_tiles)
        ds_final = _add_angles(ds_final, ds_angles)
        return ds_final


def _get_sen2_angles(xml_dict: dict, band_names: list[str]) -> xr.Dataset:
    # read out solar and viewing angles
    geocode = xml_dict["n1:Level-2A_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]
    ulx = float(geocode["Geoposition"][0]["ULX"])
    uly = float(geocode["Geoposition"][0]["ULY"])
    x = ulx + 5000 * np.arange(23)
    y = uly - 5000 * np.arange(23)

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
            (2, len(band_names), len(detector_ids), len(x), len(y)),
            np.nan,
            dtype=np.float32,
        ),
        dims=["angle", "band", "detector_id", "y", "x"],
        coords=dict(
            angle=["Zenith", "Azimuth"],
            band=band_names,
            detector_id=detector_ids,
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
            da.loc[angle, band_name, detector_id] = _get_angle_values(
                detector_angles, angle
            )
    # Do the same for the solar angles
    for angle in da.angle.values:
        da.loc[angle, "solar", detector_ids[0]] = _get_angle_values(
            angles["Sun_Angles_Grid"], angle
        )
    # Apply nanmean along detector ID axis
    da = da.mean(dim="detector_id", skipna=True)
    ds = xr.Dataset()
    for angle in da.angle.values:
        ds[f"solar_angle_{angle.lower()}"] = da.sel(
            band="solar", angle=angle
        ).drop_vars(["band", "angle"])
    for band in da.band.values[:-1]:
        for angle in da.angle.values:
            ds[f"viewing_angle_{angle.lower()}_{band}"] = da.sel(
                band=band, angle=angle
            ).drop_vars(["band", "angle"])
    crs = pyproj.CRS.from_epsg(geocode["HORIZONTAL_CS_CODE"].replace("EPSG:", ""))
    ds = ds.assign_coords(dict(spatial_ref=xr.DataArray(0, attrs=crs.to_cf())))
    ds = ds.chunk()

    return ds


def _add_angles(ds: xr.Dataset, ds_angles: xr.Dataset) -> xr.Dataset:
    ds["solar_angle"] = (
        ds_angles[["solar_angle_zenith", "solar_angle_azimuth"]]
        .to_dataarray(dim="angle")
        .assign_coords(angle=["zenith", "azimuth"])
    )
    ds_temp = xr.Dataset()
    bands = [
        str(k).replace("viewing_angle_zenith_", "")
        for k in ds_angles.keys()
        if "viewing_angle_zenith" in k
    ]
    keys = [k for k in ds_angles.keys() if "viewing_angle_zenith" in k]
    ds_temp["zenith"] = (
        ds_angles[keys].to_dataarray(dim="band").assign_coords(band=bands)
    )
    keys = [k for k in ds_angles.keys() if "viewing_angle_azimuth" in k]
    ds_temp["azimuth"] = (
        ds_angles[keys].to_dataarray(dim="band").assign_coords(band=bands)
    )
    ds["viewing_angle"] = ds_temp[["zenith", "azimuth"]].to_dataarray(dim="angle")

    return ds


def _get_angle_values(values_list: dict, angle: str) -> np.ndarray:
    values = values_list[angle]["Values_List"]["VALUES"]
    array = np.array([row.split(" ") for row in values]).astype(np.float32)
    return array


def _get_band_names_from_dataset(ds: xr.Dataset) -> list[str]:
    band_names = [
        str(key).split("_")[0] for key in ds.keys() if str(key).startswith("B")
    ]
    return [name for name in SENITNEL2_BANDS if name in band_names]


def _merge_utm_zones(
    list_ds_utm: list[xr.Dataset], target_gm: GridMapping = None, **open_params
) -> xr.Dataset:
    resampled_list_ds = []
    for ds in list_ds_utm:
        resampled_list_ds.append(
            _resample_dataset_soft(ds, target_gm=target_gm, **open_params)
        )
        if target_gm is None:
            target_gm = GridMapping.from_dataset(resampled_list_ds[0])
    return mosaic_spatial_take_first(resampled_list_ds)


def _resample_dataset_soft(
    ds: xr.Dataset,
    target_gm: GridMapping,
    fill_value: float | int = None,
    interpolation: str = "nearest",
    **open_params,
) -> xr.Dataset:
    if target_gm is not None:
        crs_final = target_gm.crs
        spatial_res = target_gm.x_res
    else:
        crs_final = pyproj.CRS.from_string(open_params["crs"])
        spatial_res = open_params["spatial_res"]
    crs_data = pyproj.CRS.from_cf(ds.spatial_ref.attrs)

    if crs_final == crs_data:
        if ds.x[1] - ds.x[0] != spatial_res or ds.y[1] - ds.y[0] != -spatial_res:
            if target_gm is None:
                target_gm = get_gridmapping(
                    [ds.x[0].item(), ds.y[-1].item(), ds.x[-1].item(), ds.y[0].item()],
                    spatial_res,
                    crs_data,
                    open_params.get("tile_size", TILE_SIZE),
                )
            ds = affine_transform_dataset(
                ds, target_gm=target_gm, gm_name="spatial_ref"
            )
    else:
        if target_gm is None:
            target_gm = get_gridmapping(
                open_params["bbox"],
                open_params["spatial_res"],
                crs_final,
                open_params.get("tile_size", TILE_SIZE),
            )
        if fill_value is None:
            fill_value = SENTINEL2_FILL_VALUE
        ds = reproject_dataset(
            ds, target_gm=target_gm, fill_value=fill_value, interpolation=interpolation
        )
    return ds


def _get_bounding_box(access_params: xr.DataArray) -> list[float | int]:
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for tile_id in access_params.tile_id.values:
        params = next(
            value
            for value in access_params.sel(tile_id=tile_id).values.flatten()
            if value is not None
        )
        bbox = params["item"].assets[params["name"]].extra_fields["proj:bbox"]
        if xmin > bbox[0]:
            xmin = bbox[0]
        if ymin > bbox[1]:
            ymin = bbox[1]
        if xmax < bbox[2]:
            xmax = bbox[2]
        if ymax < bbox[3]:
            ymax = bbox[3]
    return [xmin, ymin, xmax, ymax]


def _create_empty_dataset(
    access_params: xr.DataArray,
    asset_name: str,
    items_bbox: list[float | int],
    final_bbox: list[float | int],
    spatial_res: int | float = None,
) -> xr.Dataset:
    params = next(
        value
        for value in access_params.sel(asset_name=asset_name).values.flatten()
        if value is not None
    )
    if spatial_res is None:
        spatial_res = params["item"].assets[params["name"]].extra_fields["gsd"]
    crs = params["item"].assets[params["name"]].extra_fields["proj:code"]
    half_res = spatial_res / 2

    y_start = items_bbox[3] - spatial_res * (
        (items_bbox[3] - final_bbox[3]) // spatial_res
    )
    y_end = items_bbox[1] + spatial_res * (
        (final_bbox[1] - items_bbox[1]) // spatial_res
    )
    y = np.arange(y_start - half_res, y_end, -spatial_res)
    x_end = items_bbox[2] - spatial_res * (
        (items_bbox[2] - final_bbox[2]) // spatial_res
    )
    x_start = items_bbox[0] + spatial_res * (
        (final_bbox[0] - items_bbox[0]) // spatial_res
    )
    x = np.arange(x_start + half_res, x_end, spatial_res)

    empty_data = da.zeros(
        (access_params.sizes["time"], len(y), len(x)),
        dtype=np.uint16,
        chunks=(1, TILE_SIZE, TILE_SIZE),
    )
    ds = xr.Dataset(
        {params["name_origin"]: (("time", "y", "x"), empty_data)},
        coords={"x": x, "y": y, "time": access_params.time},
    )
    ds = ds.assign_coords(
        spatial_ref=xr.DataArray(0, attrs=pyproj.CRS.from_string(crs).to_cf())
    )
    return ds
