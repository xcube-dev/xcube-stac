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
    get_spatial_dims,
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
        """Open Sentinel-2 tiles (stored in jp2 fomrat) as a dataset or multilevel
        dataset using the specified access parameters.

        This method supports opening single-resolution datasets with `rioxarray`
        or multi-resolution datasets using a custom `Jp2MultiLevelDataset` class,
        depending on the specified data type or opener ID.

        Args:
            access_params: Dictionary containing the necessary information to locate
                and open the dataset, including keys such as 'protocol', 'root',
                and 'fs_path'.
            opener_id: Optional string identifier indicating the opener type. If not
                provided, defaults to an empty string. Used to determine if a
                multi-resolution dataset should be returned.
            data_type: Optional data type indicator used to infer whether a
                `MultiLevelDataset` is expected.
            **open_params: Additional opening parameters forwarded to the dataset
                opener.

        Returns:
            A data or multilevel dataset, depending on the input parameters.

        Notes:
            - If `data_type` is a valid multi-level dataset type or `opener_id`
              starts with `"mldataset"`, a `Jp2MultiLevelDataset` is returned.
            - Otherwise, the dataset is opened using `rioxarray.open_rasterio`.
        """
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
        """Group STAC items by solar day, tile ID, and processing version.

        This method organizes a list of Sentinel-2 STAC items into an `xarray.DataArray`
        with dimensions `(time, tile_id, idx)`, where:
        - `time` corresponds to the solar acquisition date (ignoring time of day),
        - `tile_id` is the Sentinel-2 MGRS tile code,
        - `idx` accounts for up to two acquisitions per tile (e.g., due to multiple
          observations for the same tile),
        - The most recent processing version is selected if multiple exist.

        Args:
            items: List of STAC items to group. Each item must have:
                - `properties["datetime_nominal"]`: Nominal acquisition datetime.
                - `properties["grid:code"]`: Tile ID (MGRS grid).
                - A processing version recognizable by `_get_processing_version`.

        Returns:
            A 3D DataArray of shape (time, tile_id, idx) containing STAC items.
            Time coordinate values are actual datetimes (not just dates), derived
            from the nominal acquisition datetime of the first item per date/tile
            combination.

        Notes:
            - Only up to two items per (date, tile_id) and processing version
              are considered.
            - If more than two items exist for the same (date, tile_id, proc_version),
              a warning is logged and only the first two are used.
            - Among multiple processing versions, only the latest is retained.
        """
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
                    f"[{item0.id}, {item1.id}, {item.id}]. Only the first two items "
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

        # replace date by datetime from first item
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
        """Extract the processing version from a STAC item.

        This method attempts to retrieve the processing version of a Sentinel-2 item
        from either the `processing:version` field or, if absent, falls back to
        `s2:processing_baseline`. If both are missing, a default of "1.0" is used.

        Args:
            item: A STAC item containing metadata about a Sentinel-2 observation.

        Returns:
            The processing version of the item.
        """
        return float(
            item.properties.get(
                "processing:version",
                item.properties.get("s2:processing_baseline", "1.0"),
            )
        )

    def generate_cube(self, access_params: xr.DataArray, **open_params) -> xr.Dataset:
        """Generate a spatiotemporal data cube from access parameters.

        This function takes grouped access parameters and generates a unified xarray
        dataset, mosaicking and stacking the data across spatial tiles and time.
        Optionally, scaling is applied and solar and viewing angle are calculated.

        Args:
            access_params: A 4D data array indexed by tile_id, asset_name, time,
                and idx, where each element contains data access metadata.
            **open_params: Optional keyword arguments for data opening and processing:
                - apply_scaling (bool): Whether to apply radiometric scaling.
                    Default is True.
                - angles_sentinel2 (bool): Whether to include Sentinel-2 solar angles.
                    Default is False.
                - bbox (list): Bounding box used for spatial subsetting.
                - crs (str): Coordinate reference system of the bounding box.
                - opener_id, data_type, etc.: Passed to the data-opening function.

        Returns:
            A dataset representing the spatiotemporal cube.
        """
        # Group the tile IDs by UTM zones
        utm_tile_id = defaultdict(list)
        for tile_id in access_params.tile_id.values:
            item = next(
                value["item"]
                for value in access_params.sel(tile_id=tile_id, idx=0).values.ravel()
                if value is not None
            )
            crs = item.assets["AOT_10m"].extra_fields["proj:code"]
            utm_tile_id[crs].append(tile_id)

        # Insert the tile data per UTM zone
        list_ds_utm = []
        for crs, tile_ids in utm_tile_id.items():
            access_params_sel = access_params.sel(tile_id=tile_ids)
            list_ds_utm.append(
                self._insert_tile_data_in_utm_zone(
                    access_params_sel, crs, **open_params
                )
            )

        # Reproject datasets from different UTM zones to a common grid reference system
        # and merge them into a single unified dataset for seamless spatial analysis.
        ds_final = _merge_utm_zones(list_ds_utm, **open_params)

        if open_params.get("apply_scaling", True):
            ds_final = apply_offset_scaling_stack_mode(ds_final, access_params)
        if open_params.get("angles_sentinel2", False):
            ds_final = self.add_sen2_angles_stack(ds_final, access_params)

        return ds_final

    def _insert_tile_data_in_utm_zone(
        self,
        access_params_sel: xr.DataArray,
        crs_utm: str,
        opener_id: str = None,
        data_type: DataTypeLike = None,
        **open_params,
    ) -> xr.Dataset:
        """Load and insert tile data for a specific UTM zone into a unified dataset.

        This method processes the assets from selected STAC items within a particular
        UTM zone. It opens the data for each asset and time step, clips it to the
        target bounding box (reprojected to the UTM CRS), mosaics overlapping spatial
        datasets by taking the first valid pixel, and inserts the processed data into
        an aggregated dataset for that UTM zone.

        Args:
            access_params: An xarray DataArray containing access information for assets,
                indexed by tile_id, asset_name, time, and idx.
            crs_utm: The target UTM coordinate reference system identifier.
            opener_id: Optional string identifier to specify the data opener to use.
            data_type: Optional data type hint for the data opening function.
            **open_params: Additional parameters to control data opening and processing,
                including 'bbox' (bounding box) and 'crs' (coordinate reference system).

        Returns:
            A dataset containing mosaicked and stacked 3d datacubes
            (time, spatial_y, spatial_x) for all assets within the specified UTM zone.
        """
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
                    ds = mosaic_spatial_take_first(
                        list_ds_idx, fill_value=SENTINEL2_FILL_VALUE
                    )
                    asset_ds = self._insert_tile_data(asset_ds, asset_name, ds, dt_idx)
            list_ds_asset.append(asset_ds)

        return merge_datasets(list_ds_asset)

    def _insert_tile_data(self, asset_ds, asset_name, ds, dt_idx):
        """Insert spatial data from a smaller dataset into a larger asset dataset at
        the correct spatial indices.

        This method locates the spatial coordinates of the input dataset `ds` within
        the larger `asset_ds` along the 'x' and 'y' dimensions, then copies the data
        from `ds["band_1"]` into the corresponding slice of `asset_ds` for the
        specified time index `dt_idx`.

        Args:
            asset_ds: The larger dataset representing the aggregated asset data.
            asset_name: The name of variable within `asset_ds` to insert data into.
            ds: The smaller xarray Dataset containing data to be inserted.
            dt_idx: The time index in `asset_ds` at which to insert the data.

        Returns:
            The updated `asset_ds` with data from `ds` inserted at the appropriate
            spatial location.
        """
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
        """Extract Sentinel-2 solar and viewing angle information from the granule
        metadata and add it to the dataset `ds`.

        This method downloads and parses the granule metadata XML file associated with
        the given STAC item, then extracts solar and viewing angle information for
        the relevant bands.

        Args:
            item: A STAC item containing a 'granule_metadata' asset with the
                XML metadata.
            ds: A dataset containing Sentinel-2 data bands, used to determine which
                bands' angles to extract.

        Returns:
            An updated version of `ds`, with solar and viewing angle data added
            corresponding to the bands.
        """
        target_gm = get_angle_target_gm(ds)
        ds_angles = self.get_sen2_angles(item, ds)
        ds_angles = _resample_dataset_soft(
            ds_angles,
            target_gm,
            fill_value=np.nan,
            interpolation="bilinear",
        )
        ds = _add_angles(ds, ds_angles)
        return ds

    def add_sen2_angles_stack(
        self,
        ds_final: xr.Dataset,
        access_params: xr.DataArray,
    ) -> xr.Dataset:
        """Add Sentinel-2 solar and viewing angle information from multiple STAC items
        as a mosaicked and stacked dataset.

        This method processes angle metadata for each tile from the provided
        `access_params`. It retrieves angle datasets for each tile,
        mosaics them spatially, resamples to match the target grid mapping, and
        finally adds the combined angle data to the input dataset `ds_final`.

        Args:
            ds_final: The main dataset containing Sentinel-2 data to which angle data
                will be added.
            access_params: A DataArray containing access parameters for STAC items.

        Returns:
            An updated `ds_final` which includes solar and viewing angle information
            stacked and mosaicked, so that it aligns with the Sentinel-2 spectral data.
        """
        target_gm = get_angle_target_gm(ds_final)
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
                            ).values.ravel()
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
            ds_tile = xr.concat(list_ds_time, dim="time", join="exact").chunk(-1)
            np_datetimes_sel = [
                value
                for idx, value in enumerate(access_params.time.values)
                if idx not in idx_remove_dt
            ]
            ds_tile = ds_tile.assign_coords(coords=dict(time=np_datetimes_sel))
            ds_tile = _resample_dataset_soft(
                ds_tile,
                target_gm,
                fill_value=np.nan,
                interpolation="bilinear",
            )
            ds_tile = ds_tile.chunk(dict(time=1))
            if len(idx_remove_dt) > 0:
                ds_tile = _fill_nan_slices(
                    ds_tile, access_params.time.values, idx_remove_dt
                )
            list_ds_tiles.append(ds_tile)

        ds_angles = mosaic_spatial_take_first(list_ds_tiles)
        ds_final = _add_angles(ds_final, ds_angles)
        return ds_final


def get_angle_target_gm(ds_final: xr.Dataset) -> GridMapping:
    """Determine the target grid mapping for angle datasets based on the input dataset.

    This function computes an appropriate bounding box and spatial resolution
    for resampling angle data to align with the spatial reference of the given
    Sentinel-2 dataset. It handles both geographic (latitude/longitude) and
    projected coordinate reference systems, adjusting resolution accordingly.

    Args:
        ds_final: The dataset whose spatial reference and coordinates define the
            target grid mapping.

    Returns:
        A GridMapping object representing the bounding box, resolution, and CRS
        suitable for resampling angle data to match `ds_final`.

    Notes:
        The native resolution of solar and viewing angle data in Sentinel-2 products
        is 5000 meters. This resolution is preserved in the resulting target grid
        mapping used during datacube generation.
    """
    crs = pyproj.CRS.from_cf(ds_final.spatial_ref.attrs)
    y_coord, x_coord = get_spatial_dims(ds_final)
    if crs.is_geographic:
        y_res = 5000 / 111320
        y_center = (ds_final[x_coord][0].item() + ds_final[x_coord][-1].item()) / 2
        x_res = 5000 / (111320 * np.cos(np.radians(y_center)))
    else:
        y_res = 5000
        x_res = 5000

    bbox = [
        ds_final[x_coord][0].item(),
        ds_final[y_coord][0].item(),
        ds_final[x_coord][-1].item(),
        ds_final[y_coord][-1].item(),
    ]
    if bbox[3] < bbox[1]:
        y_min, y_max = bbox[3], bbox[1]
        bbox[1], bbox[3] = y_min, y_max
    return get_gridmapping(bbox, (x_res, y_res), crs)


def _fill_nan_slices(ds: xr.Dataset, times: np.array, idx_nan: list[int]) -> xr.Dataset:
    """Insert NaN-filled time slices into a dataset at specified indices along
    the time axis.

    This function takes a dataset and a list of time indices (`idx_nan`) where
    NaN-filled slices should be inserted. It constructs a dataset by
    concatenating segments of the original dataset and NaN-filled slices such that
    the resulting Dataset includes placeholders at the specified positions.

    Parameters:
        ds: The input dataset with a "time" dimension.
        times: array of datetime-like values corresponding to the full time range,
            including positions for NaNs.
        idx_nan: Indices in `times` where NaN slices should be inserted.

    Returns:
        A new dataset with NaN-filled slices inserted at the specified indices,
        maintaining alignment with `times`.
    """
    ds_nan = _create_nan_slice(ds)
    list_ds = []
    if idx_nan[0] > 0:
        list_ds.append(ds.isel(time=slice(None, idx_nan[0])))
    for i, idx in enumerate(idx_nan):
        list_ds.append(ds_nan.assign_coords(coords=dict(time=[times[idx]])))
        if i < len(idx_nan) - 1:
            list_ds.append(ds.isel(time=slice(idx - i, idx_nan[i + 1] - i - 1)))
    if idx_nan[-1] < len(times) - 1:
        list_ds.append(ds.isel(time=slice(idx_nan[-1] - (len(idx_nan) - 1), None)))
    return xr.concat(list_ds, dim="time", join="exact")


def _create_nan_slice(ds: xr.Dataset) -> xr.Dataset:
    """Create a NaN-filled slice of the input dataset for a single time step.

    This function generates a new dataset with the same structure, dimensions,
    coordinates, and attributes as the first time step of the input dataset,
    but replaces all data values with NaNs. This is useful for inserting placeholder
    time steps into time-series datasets.

    Parameters:
        ds: The input dataset with a "time" dimension.

    Returns:
        A dataset with one time step where all variable values are NaN,
        matching the shape and metadata of the original dataset.
    """
    nan_ds = xr.Dataset()
    for var_name, array in ds.data_vars.items():
        array = array.isel(time=slice(0, 1))
        nan_data = da.full(
            array.shape, np.nan, dtype=array.dtype, chunks=array.data.chunksize
        )
        nan_ds[var_name] = xr.DataArray(
            nan_data, dims=array.dims, coords=array.coords, attrs=array.attrs
        )
    return nan_ds


def _get_sen2_angles(xml_dict: dict, band_names: list[str]) -> xr.Dataset:
    """Extract solar and viewing angle information from a Sentinel-2 metadata
    dictionary derived from the XML metadata file.

    This function parses geometric metadata from a Sentinel-2 L2A product XML dictionary
    to compute solar and viewing zenith/azimuth angles over a 23x23 grid of 5000-meter
    spacing (native grid of Sentinel-2 angles). Viewing angles are computed per
    detector and band, then averaged across detectors. Solar angles are included as a
    separate band.

    Parameters:
        xml_dict: Parsed XML metadata from a Sentinel-2 L2A SAFE product.
        band_names: List of Sentinel-2 band names to extract angles for.

    Returns:
        A dataset containing the following variables:
            - solar_angle_zenith, solar_angle_azimuth
            - viewing_angle_zenith_{band}, viewing_angle_azimuth_{band}
        Each variable is a 2D grid with coordinates x and y.

    Notes:
        - The grid resolution is 5000 meters, 23x23 in size.
        - Angles are averaged over detector IDs.
        - The dataset includes a spatial reference system as a coordinate.
    """
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

    array = xr.DataArray(
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
        for angle in array.angle.values:
            array.loc[angle, band_name, detector_id] = _get_angle_values(
                detector_angles, angle
            )
    # Do the same for the solar angles
    for angle in array.angle.values:
        array.loc[angle, "solar", detector_ids[0]] = _get_angle_values(
            angles["Sun_Angles_Grid"], angle
        )
    # Apply nanmean along detector ID axis
    array = array.mean(dim="detector_id", skipna=True)
    ds = xr.Dataset()
    for angle in array.angle.values:
        ds[f"solar_angle_{angle.lower()}"] = array.sel(
            band="solar", angle=angle
        ).drop_vars(["band", "angle"])
    for band in array.band.values[:-1]:
        for angle in array.angle.values:
            ds[f"viewing_angle_{angle.lower()}_{band}"] = array.sel(
                band=band, angle=angle
            ).drop_vars(["band", "angle"])
    crs = pyproj.CRS.from_epsg(geocode["HORIZONTAL_CS_CODE"].replace("EPSG:", ""))
    ds = ds.assign_coords(dict(spatial_ref=xr.DataArray(0, attrs=crs.to_cf())))
    ds = ds.chunk()

    return ds


def _add_angles(ds: xr.Dataset, ds_angles: xr.Dataset) -> xr.Dataset:
    """Add solar and viewing angle information to the datacube.

    This function integrates angle data (solar and viewing) from a separate angle
    dataset, generated by _get_sen2_angles, into the final datacube by combining
    relevant variables into structured DataArrays.

    Parameters:
        ds: The primary dataset containing the spectral data to which angle
            information will be added.
        ds_angles: A dataset containing solar and viewing angle variables,
            generated by _get_sen2_angles.

    Returns:
        The input dataset `ds` with two additional variables:
            - 'solar_angle': A DataArray of shape (angle, angle_y, angle_x).
            - 'viewing_angle': A DataArray of shape (angle, band, angle_y, angle_x).

    Notes:
        - Renames spatial coordinates in `ds_angles` to avoid conflicts.
        - The angle dimension contains 'zenith' and 'azimuth'
    """
    x_coord, y_coord = get_spatial_dims(ds_angles)
    ds_angles = ds_angles.rename(
        {y_coord: f"angle_{y_coord}", x_coord: f"angle_{x_coord}"}
    )
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
    ds = ds.chunk(dict(angle=-1, band=-1))

    return ds


def _get_angle_values(values_list: dict, angle: str) -> np.ndarray:
    """Extract a 2D array of angle values from a nested Sentinel-2 metadata dictionary.

    This function reads angle values (e.g., 'Zenith' or 'Azimuth') from a structured
    dictionary, typically parsed from a Sentinel-2 XML file. The values are expected
    to be strings representing rows of space-separated numbers, which are converted
    into a NumPy array.

    Parameters:
        values_list: A nested dictionary containing angle metadata.
        angle: The key indicating which angle to extract (e.g., "Zenith", "Azimuth").

    Returns:
        A 2D NumPy array of float32 values representing the specified angle grid.
    """
    values = values_list[angle]["Values_List"]["VALUES"]
    array = np.array([row.split(" ") for row in values]).astype(np.float32)
    return array


def _get_band_names_from_dataset(ds: xr.Dataset) -> list[str]:
    """Extract valid Sentinel-2 band names from the dataset variable names.

    This function scans the keys of the input dataset and collects those that
    start with a 'B' (e.g., 'B02', 'B08A'). It returns only the names that match
    known Sentinel-2 band identifiers defined in the global `SENITNEL2_BANDS` list.

    Parameters:
        ds: The input dataset containing variables named with band prefixes.

    Returns:
        A list of valid Sentinel-2 band names found in the dataset.
    """
    band_names = [
        str(key).split("_")[0] for key in ds.keys() if str(key).startswith("B")
    ]
    return [name for name in SENITNEL2_BANDS if name in band_names]


def _merge_utm_zones(list_ds_utm: list[xr.Dataset], **open_params) -> xr.Dataset:
    """Merge multiple Sentinel-2 tiles from different UTM zones into a single dataset.

    This function takes a list of Sentinel-2 datasets (each in a different UTM zone),
    resamples them to a common grid defined by a target CRS and spatial resolution,
    and mosaics them into a single output using a "take first" strategy for overlaps.

    Parameters:
        list_ds_utm: A list of xarray Datasets, one for each UTM zone.
        open_params: Dictionary of parameters required for constructing the target grid,
            including:
            - crs: Target coordinate reference system (string or EPSG code).
            - spatial_res: Spatial resolution as a single value or tuple (x_res, y_res).
            - bbox: Bounding box for the output grid (minx, miny, maxx, maxy).
            - tile_size (optional): Tile size for the target grid.

    Returns:
        A single xarray Dataset reprojected to the target CRS and resolution,
        containing merged data from all input UTM zones.

    Notes:
        - If one input dataset already matches the target CRS and resolution,
          its grid mapping is reused unless resolution mismatches are found.
        - Overlapping regions are resolved by selecting the first non-NaN value.
    """
    # get correct target gridmapping
    crss = [pyproj.CRS.from_cf(ds["spatial_ref"].attrs) for ds in list_ds_utm]
    target_crs = pyproj.CRS.from_string(open_params["crs"])
    crss_equal = [target_crs == crs for crs in crss]
    if any(crss_equal):
        true_index = crss_equal.index(True)
        ds = list_ds_utm[true_index]
        target_gm = GridMapping.from_dataset(ds)
        spatial_res = open_params["spatial_res"]
        if not isinstance(spatial_res, tuple):
            spatial_res = (spatial_res, spatial_res)
        if (
            ds.x[1] - ds.x[0] != spatial_res[0]
            or abs(ds.y[1] - ds.y[0]) != spatial_res[1]
        ):
            target_gm = get_gridmapping(
                open_params["bbox"],
                open_params["spatial_res"],
                open_params["crs"],
                open_params.get("tile_size", TILE_SIZE),
            )
    else:
        target_gm = get_gridmapping(
            open_params["bbox"],
            open_params["spatial_res"],
            open_params["crs"],
            open_params.get("tile_size", TILE_SIZE),
        )

    resampled_list_ds = []
    for ds in list_ds_utm:
        resampled_list_ds.append(_resample_dataset_soft(ds, target_gm))
    return mosaic_spatial_take_first(resampled_list_ds, fill_value=SENTINEL2_FILL_VALUE)


def _resample_dataset_soft(
    ds: xr.Dataset,
    target_gm: GridMapping,
    fill_value: float | int = None,
    interpolation: str = "nearest",
) -> xr.Dataset:
    """Resample a dataset to a target grid mapping, using either affine transform
    or reprojection.

    If the source and target grid mappings are close, the dataset is returned unchanged.
    If they share the same CRS but differ spatially, an affine transformation is applied.
    Otherwise, the dataset is reprojected to the target grid, with optional fill value
    and interpolation.

    Parameters:
        ds: The source xarray Dataset to be resampled.
        target_gm: The target grid mapping
        fill_value: Value to fill in areas without data during reprojection.
            Defaults to 0 (no-data value in the Sentinel-2 product).
        interpolation: Interpolation method to use during reprojection
            (see reproject_dataset function in xcube).

    Returns:
        The resampled dataset aligned with the target grid mapping.
    """
    source_gm = GridMapping.from_dataset(ds)
    if source_gm.is_close(target_gm):
        return ds
    if target_gm.crs == source_gm.crs:
        var_configs = {}
        for var in ds.data_vars:
            var_configs[var] = dict(recover_nan=True)
        ds = affine_transform_dataset(
            ds,
            source_gm=source_gm,
            target_gm=target_gm,
            gm_name="spatial_ref",
            var_configs=var_configs,
        )
    else:
        if fill_value is None:
            fill_value = SENTINEL2_FILL_VALUE
        ds = reproject_dataset(
            ds,
            source_gm=source_gm,
            target_gm=target_gm,
            fill_value=fill_value,
            interpolation=interpolation,
        )
    return ds


def _get_bounding_box(access_params: xr.DataArray) -> list[float | int]:
    """Compute the overall bounding box that covers all tiles in the given access
    parameters.

    Iterates through each tile in `access_params` to extract the bounding box
    from its metadata and calculates the minimum bounding rectangle encompassing all
    tiles.

    Parameters:
        access_params: An array containing tile metadata, with coordinates
            including 'tile_id'.

    Returns:
        A list with four elements [xmin, ymin, xmax, ymax] representing the
        bounding box that encloses all tiles.
    """
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for tile_id in access_params.tile_id.values:
        params = next(
            value
            for value in access_params.sel(tile_id=tile_id).values.ravel()
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
    items_bbox: list[float | int] | tuple[float | int],
    final_bbox: list[float | int] | tuple[float | int],
    spatial_res: int | float = None,
) -> xr.Dataset:
    """Create an empty xarray Dataset aligned to a specified spatial grid and
    bounding box.

    This function initializes a dataset filled with zeros (no-data value os Sentinel-2
    product), matching the native pixel distribution of the native Sentinel-2 product,
    the spatial extent and resolution defined by the input bounding
    boxes and parameters extracted from `access_params`. The dataset includes
    coordinates for time, y, and x dimensions, and an appropriate spatial reference
    coordinate.

    Parameters:
        access_params: An xarray DataArray containing metadata for available assets,
            indexed by time, tile ID and asset names.
        asset_name: The name of the asset to use for extracting spatial metadata
            such as coordinate reference system and ground sampling distance.
        items_bbox: Bounding box (xmin, ymin, xmax, ymax) covering the input tiles.
        final_bbox: Desired final bounding box (xmin, ymin, xmax, ymax) to align the
            dataset to.
        spatial_res: Optional spatial resolution. If None, it is extracted from asset
            metadata.

    Returns:
        A dataset with dimensions (time, y, x), initialized with zeros,
        and properly aligned spatial coordinates and CRS information.
    """
    params = next(
        value
        for value in access_params.sel(asset_name=asset_name).values.ravel()
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
