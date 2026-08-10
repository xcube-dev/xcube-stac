# The MIT License (MIT)
# Copyright (c) 2024-2026 by the xcube development team and contributors
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

import datetime
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyproj
import pystac
import xarray as xr

from xcube_stac.accessors.hls import (
    Sen2HlsStacArdcAccessor,
    Sen2HlsStacItemAccessor,
    _merge_utm_zones,
    fix_utm_hemisphere,
)


class HlsStacItemAccessorTest(unittest.TestCase):

    @staticmethod
    def make_item(bbox, proj_code):
        return pystac.Item(
            id="test-item",
            geometry=None,
            bbox=bbox,
            datetime=datetime.datetime.fromisoformat("20200101T11:23:23"),
            properties={"proj:epsg": proj_code},
        )

    def test_fix_utm_hemisphere_northern(self):
        item = self.make_item(
            bbox=[500000, 10, 600000, 20],  # center latitude = 15
            proj_code=32733,  # incorrectly southern
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:epsg"], 32633)

    def test_fix_utm_hemisphere_southern(self):
        item = self.make_item(
            bbox=[500000, -20, 600000, -10],  # center latitude = -15
            proj_code=32633,  # incorrectly northern
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:epsg"], 32733)

    def test_fix_utm_hemisphere_equator_is_northern(self):
        item = self.make_item(
            bbox=[500000, -1, 600000, 1],  # center latitude = 0
            proj_code=32733,
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:epsg"], 32633)

    def test_fix_utm_hemisphere_keeps_correct_code(self):
        item = self.make_item(
            bbox=[500000, 10, 600000, 20],
            proj_code=32633,
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:epsg"], 32633)

    def test_fix_utm_hemisphere_preserves_zone(self):
        item = self.make_item(
            bbox=[500000, -10, 600000, -5],
            proj_code=32621,
        )
        result = fix_utm_hemisphere([item])
        self.assertEqual(result[0].properties["proj:epsg"], 32721)


class HlsStacCoverageTest(unittest.TestCase):
    def setUp(self):
        self.catalog = pystac.Catalog(
            id="test-catalog",
            description="Test Catalog",
            href="https://example.com/catalog.json",
        )
        self.item_accessor = Sen2HlsStacItemAccessor(self.catalog)
        self.ardc_accessor = Sen2HlsStacArdcAccessor(self.catalog)

    @staticmethod
    def make_asset(name):
        asset = pystac.Asset(
            href=f"https://example.com/{name}.tif",
            media_type="image/tiff; application=geotiff",
            roles=["data"],
            title=name,
        )
        asset.extra_fields["raster:bands"] = [
            {"scale": 0.1, "offset": 1.0, "nodata": 0}
        ]
        return asset

    @classmethod
    def make_item(
        cls,
        item_id="HLS.L30.T32TNS.2020124T101010.v2.0",
        asset_names=("B01",),
        proj_code="EPSG:32632",
        bbox=(0.0, 0.0, 1.0, 1.0),
        dt="2020-12-04T10:15:00",
    ):
        item = pystac.Item(
            id=item_id,
            geometry=None,
            bbox=list(bbox),
            datetime=datetime.datetime.fromisoformat(dt),
            properties={"proj:code": proj_code},
        )
        item.stac_extensions = [
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
        ]
        for asset_name in asset_names:
            item.add_asset(asset_name, cls.make_asset(asset_name))
        return item

    @staticmethod
    def make_data_array(var_names=("B01",), values=None, x=(0.0, 1.0), y=(1.0, 0.0)):
        if values is None:
            values = np.array([[1, 2], [3, 4]], dtype=np.float32)
        if not isinstance(var_names, tuple):
            var_names = tuple(var_names)
        data_vars = {}
        for var_name in var_names:
            data = values
            if var_name == "Fmask":
                data = np.array([[255, 1], [1, 255]], dtype=np.uint8)
            data_vars[var_name] = (("y", "x"), data)
        return xr.Dataset(
            data_vars,
            coords=dict(
                x=np.array(x),
                y=np.array(y),
                spatial_ref=0,
            ),
        )

    @staticmethod
    def make_time_cube_dataset(
        var_names=("Fmask",),
        time="2020-12-04T10:15:00",
        x=(0.0, 1.0),
        y=(1.0, 0.0),
    ):
        if not isinstance(var_names, tuple):
            var_names = tuple(var_names)
        data_vars = {}
        for var_name in var_names:
            if var_name == "Fmask":
                data = np.array([[[255, 1], [1, 255]]], dtype=np.uint8)
            else:
                data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
            data_vars[var_name] = (("time", "y", "x"), data)
        return xr.Dataset(
            data_vars,
            coords=dict(
                time=np.array([time], dtype="datetime64[ns]"),
                x=np.array(x),
                y=np.array(y),
                spatial_ref=0,
            ),
        )

    def test_apply_offset_scaling_scales_and_masks(self):
        ds = self.make_data_array(("B01",), values=np.array([[1, 0], [3, 4]]))
        ds["B01"].attrs = {
            "_FillValue": 0,
            "scale_factor": 0.5,
            "add_offset": 1.0,
        }

        result = Sen2HlsStacItemAccessor._apply_offset_scaling(ds)

        np.testing.assert_allclose(
            result["B01"].values,
            np.array([[1.5, np.nan], [2.5, 3.0]], dtype=np.float32),
            equal_nan=True,
        )

    def test_combiner_function_resamples_with_geographic_crs(self):
        raw_ds = self.make_data_array(("B01",))
        item = self.make_item(asset_names=("B01",))
        source_gm = Mock()
        source_gm.xy_bbox = (10.0, 20.0, 30.0, 40.0)
        source_gm.crs = pyproj.CRS.from_epsg(32632)
        source_gm.xy_res = 30.0
        target_gm = Mock()
        geographic_crs = pyproj.CRS.from_epsg(4326)

        with (
            patch(
                "xcube_stac.accessors.hls.GridMapping.from_dataset",
                return_value=source_gm,
            ) as from_dataset_mock,
            patch(
                "xcube_stac.accessors.hls.reproject_bbox",
                return_value=(1.0, 2.0, 3.0, 4.0),
            ) as reproject_bbox_mock,
            patch(
                "xcube_stac.accessors.hls.resolution_meters_to_degrees",
                return_value=0.25,
            ) as res_to_deg_mock,
            patch(
                "xcube_stac.accessors.hls.GridMapping.regular_from_bbox",
                return_value=target_gm,
            ) as regular_from_bbox_mock,
            patch(
                "xcube_stac.accessors.hls.resample_in_space",
                side_effect=lambda ds, **kwargs: ds,
            ) as resample_mock,
        ):
            result = self.item_accessor._combiner_function(
                [raw_ds],
                item,
                self.catalog,
                assets=[item.assets["B01"]],
                apply_scaling=False,
                crs=geographic_crs,
            )

        self.assertIn("B01", result)
        from_dataset_mock.assert_called_once()
        reproject_bbox_mock.assert_called_once_with(
            source_gm.xy_bbox, source_gm.crs, geographic_crs
        )
        res_to_deg_mock.assert_called_once_with(source_gm.xy_res, 30.0)
        self.assertEqual(regular_from_bbox_mock.call_count, 1)
        self.assertEqual(
            regular_from_bbox_mock.call_args.kwargs["bbox"], (1.0, 2.0, 3.0, 4.0)
        )
        self.assertEqual(regular_from_bbox_mock.call_args.kwargs["xy_res"], 0.25)
        self.assertEqual(regular_from_bbox_mock.call_args.kwargs["crs"], geographic_crs)
        self.assertEqual(
            tuple(regular_from_bbox_mock.call_args.kwargs["tile_size"]),
            (2048, 2048),
        )
        resample_mock.assert_called_once()

    def test_group_items_keeps_duplicate_items_per_tile_and_date(self):
        items = [
            self.make_item(
                item_id="HLS.L30.T32TNS.2020124T101010.v2.0",
                dt="2020-12-04T10:10:10",
            ),
            self.make_item(
                item_id="HLS.L30.T32TNS.2020124T111111.v2.0",
                dt="2020-12-04T11:11:11",
            ),
            self.make_item(
                item_id="HLS.L30.T32TMS.2020125T101010.v2.0",
                dt="2020-12-05T10:10:10",
            ),
        ]

        grouped = self.ardc_accessor._group_items(items)

        self.assertEqual(grouped.dims, ("time", "tile_id"))
        self.assertEqual(list(grouped.tile_id.values), ["32TMS", "32TNS"])
        self.assertEqual(len(grouped.time.values), 2)
        self.assertEqual(
            len(grouped.sel(time=grouped.time.values[0], tile_id="32TNS").item()),
            2,
        )
        self.assertEqual(
            grouped.sel(time=grouped.time.values[1], tile_id="32TMS").item(),
            [items[2]],
        )

    def test_generate_utm_cube_handles_fmask_and_empty_time_slices(self):
        grouped_data = np.empty((2, 1), dtype=object)
        grouped_data[0, 0] = [self.make_item()]
        grouped_data[1, 0] = []
        grouped_items = xr.DataArray(
            grouped_data,
            dims=("time", "tile_id"),
            coords=dict(
                time=np.array(
                    ["2020-12-04T10:15:00", "2020-12-05T10:15:00"],
                    dtype="datetime64[ns]",
                ),
                tile_id=["32TNS"],
            ),
        )
        raw_ds = self.make_data_array(("Fmask", "B01"))
        bbox = (0.0, 0.0, 1.0, 1.0)
        crs_utm = pyproj.CRS.from_epsg(32632)

        with (
            patch(
                "xcube_stac.accessors.hls.reproject_bbox",
                return_value=bbox,
            ) as reproject_bbox_mock,
            patch.object(
                self.ardc_accessor, "open_item", return_value=raw_ds
            ) as open_item_mock,
            patch(
                "xcube_stac.accessors.hls.mosaic_spatial_take_first",
                side_effect=lambda dss, var_ref, fill_value: dss[0],
            ) as mosaic_mock,
        ):
            result_single = self.ardc_accessor._generate_utm_cube(
                grouped_items,
                crs_utm,
                bbox=bbox,
                crs=crs_utm,
                asset_names=["Fmask"],
                apply_scaling=False,
            )
            result_pair = self.ardc_accessor._generate_utm_cube(
                grouped_items,
                crs_utm,
                bbox=bbox,
                crs=crs_utm,
                asset_names=["Fmask", "B01"],
                apply_scaling=False,
            )

        self.assertEqual(result_single.sizes["time"], 2)
        self.assertTrue(np.all(result_single["Fmask"].isel(time=1).values == 255))
        self.assertIn("B01", result_pair)
        self.assertEqual(mosaic_mock.call_args_list[0].args[1], "Fmask")
        self.assertEqual(mosaic_mock.call_args_list[0].args[2], 255)
        self.assertEqual(mosaic_mock.call_args_list[1].args[1], "B01")
        self.assertTrue(np.isnan(mosaic_mock.call_args_list[1].args[2]))
        self.assertEqual(open_item_mock.call_count, 2)
        self.assertEqual(reproject_bbox_mock.call_count, 2)

    def test_merge_utm_zones_uses_matching_grid_and_fmask_fill_value(self):
        bbox = (0.0, 0.0, 1.0, 1.0)
        target_crs = pyproj.CRS.from_epsg(32632)
        source_gm = Mock()
        source_gm.xy_var_names = ("x", "y")
        target_gm = Mock()
        target_gm.xy_var_names = ("x", "y")

        def run_merge(var_names):
            ds = self.make_time_cube_dataset(
                var_names=var_names, x=(0.0, 5.0), y=(5.0, 0.0)
            )
            with (
                patch(
                    "xcube_stac.accessors.hls.pyproj.CRS.from_cf",
                    side_effect=[target_crs],
                ),
                patch(
                    "xcube_stac.accessors.hls.GridMapping.from_dataset",
                    return_value=source_gm,
                ) as from_dataset_mock,
                patch(
                    "xcube_stac.accessors.hls.GridMapping.regular_from_bbox",
                    return_value=target_gm,
                ) as regular_from_bbox_mock,
                patch(
                    "xcube_stac.accessors.hls.resample_in_space",
                    side_effect=lambda ds, **kwargs: ds,
                ) as resample_mock,
                patch(
                    "xcube_stac.accessors.hls.mosaic_spatial_take_first",
                    side_effect=lambda dss, var_ref, fill_value: dss[0],
                ) as mosaic_mock,
            ):
                result = _merge_utm_zones(
                    [ds],
                    bbox=bbox,
                    crs=str(target_crs),
                    spatial_res=30.0,
                )

            self.assertIn(var_names[0], result)
            self.assertEqual(from_dataset_mock.call_count, 1)
            self.assertEqual(regular_from_bbox_mock.call_count, 1)
            self.assertEqual(resample_mock.call_count, 1)
            self.assertEqual(mosaic_mock.call_count, 1)
            return mosaic_mock.call_args

        single_args = run_merge(("Fmask",))
        self.assertEqual(single_args.args[1], "Fmask")
        self.assertEqual(single_args.args[2], 255)

        paired_args = run_merge(("Fmask", "B01"))
        self.assertEqual(paired_args.args[1], "B01")
        self.assertTrue(np.isnan(paired_args.args[2]))
