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
from unittest.mock import patch

import numpy as np
import pyproj
import pystac
import xarray as xr

from xcube_stac.accessors.landsat import (
    _CHUNK_SIZE,
    LandsatC2L2StacArdcAccessor,
    LandsatC2L2StacItemAccessor,
)


class LandsatC2L2StacItemAccessorTest(unittest.TestCase):
    def setUp(self):
        self.catalog = pystac.Catalog(
            id="test-catalog",
            description="Test Catalog",
            href="https://example.com/catalog.json",
        )
        self.accessor = LandsatC2L2StacItemAccessor(self.catalog)
        self.ardc_accessor = LandsatC2L2StacArdcAccessor(self.catalog)

    @staticmethod
    def make_asset(name):
        asset = pystac.Asset(
            href=f"https://example.com/{name}.tif",
            media_type="image/tiff; application=geotiff",
            roles=["data"],
            title=name,
        )
        asset.extra_fields["raster:bands"] = [
            {"scale": 0.1, "offset": 273.15, "nodata": 0}
        ]
        return asset

    @classmethod
    def make_item(cls, asset_names=("lwir11",)):
        item = pystac.Item(
            id="LC08_L2SP_047027_20201204_02_T1",
            geometry=None,
            bbox=[0, 0, 1, 1],
            datetime=datetime.datetime.fromisoformat("2020-12-04T10:15:00"),
            properties={},
        )
        item.stac_extensions = [
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
        ]
        for asset_name in asset_names:
            item.add_asset(asset_name, cls.make_asset(asset_name))
        return item

    def test_open_item_applies_raster_scaling(self):
        raw_lwir11 = xr.Dataset(
            {
                "band_1": xr.DataArray(
                    np.array([[1, 2], [0, 4]], dtype=np.uint16), dims=("y", "x")
                )
            }
        )
        raw_red = xr.Dataset(
            {
                "band_1": xr.DataArray(
                    np.array([[5, 6], [7, 8]], dtype=np.uint16), dims=("y", "x")
                )
            }
        )
        item = self.make_item(("lwir11", "red"))

        with patch.object(
            self.accessor,
            "open_asset",
            side_effect=[raw_lwir11, raw_red],
        ):
            ds = self.accessor.open_item(item, asset_names=["lwir11", "red"])

        self.assertIn("lwir11", ds)
        self.assertIn("red", ds)
        self.assertEqual(ds["lwir11"].attrs["units"], "K")
        np.testing.assert_allclose(
            ds["lwir11"].values,
            np.array([[273.25, 273.35], [np.nan, 273.55]]),
            equal_nan=True,
        )

    def test_open_item_resamples_with_geographic_crs(self):
        raw_ds = xr.Dataset(
            {
                "band_1": xr.DataArray(
                    np.array([[1, 2], [3, 4]], dtype=np.uint16), dims=("y", "x")
                )
            },
            coords={"x": [0.0, 1.0], "y": [1.0, 0.0]},
        )
        raw_ds["band_1"].attrs = {
            "_FillValue": 0,
            "scale_factor": 1.0,
            "add_offset": 0.0,
        }
        item = self.make_item(("red",))
        source_gm = type(
            "GridMappingStub",
            (),
            {
                "xy_bbox": (10.0, 20.0, 30.0, 40.0),
                "crs": pyproj.CRS.from_epsg(32633),
                "xy_res": 30.0,
            },
        )()
        target_gm = object()
        geographic_crs = pyproj.CRS.from_epsg(4326)

        with (
            patch(
                "xcube_stac.accessors.landsat.GridMapping.from_dataset",
                return_value=source_gm,
            ) as from_dataset_mock,
            patch(
                "xcube_stac.accessors.landsat.reproject_bbox",
                return_value=(1.0, 2.0, 3.0, 4.0),
            ) as reproject_bbox_mock,
            patch(
                "xcube_stac.accessors.landsat.resolution_meters_to_degrees",
                return_value=0.25,
            ) as res_to_deg_mock,
            patch(
                "xcube_stac.accessors.landsat.GridMapping.regular_from_bbox",
                return_value=target_gm,
            ) as regular_from_bbox_mock,
            patch(
                "xcube_stac.accessors.landsat.resample_in_space",
                side_effect=lambda ds, **kwargs: ds,
            ) as resample_mock,
        ):
            ds = self.accessor._combiner_function(
                [raw_ds],
                item,
                self.catalog,
                assets=[item.assets["red"]],
                apply_scaling=False,
                crs=geographic_crs,
            )

        self.assertIn("red", ds)
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
            tuple(_CHUNK_SIZE.values()),
        )
        resample_mock.assert_called_once()

    def test_open_item_resamples_uses_source_crs_when_not_provided(self):
        raw_ds = xr.Dataset(
            {
                "band_1": xr.DataArray(
                    np.array([[1, 2], [3, 4]], dtype=np.uint16), dims=("y", "x")
                )
            },
            coords={"x": [0.0, 10.0], "y": [10.0, 0.0]},
        )
        raw_ds["band_1"].attrs = {
            "_FillValue": 0,
            "scale_factor": 1.0,
            "add_offset": 0.0,
        }
        item = self.make_item(("red",))
        source_gm = type(
            "GridMappingStub",
            (),
            {
                "xy_bbox": (10.0, 20.0, 30.0, 40.0),
                "crs": pyproj.CRS.from_epsg(32633),
                "xy_res": 10.0,
            },
        )()
        target_gm = object()

        with (
            patch(
                "xcube_stac.accessors.landsat.GridMapping.from_dataset",
                return_value=source_gm,
            ),
            patch(
                "xcube_stac.accessors.landsat.GridMapping.regular_from_bbox",
                return_value=target_gm,
            ) as regular_from_bbox_mock,
            patch(
                "xcube_stac.accessors.landsat.resample_in_space",
                side_effect=lambda ds, **kwargs: ds,
            ) as resample_mock,
        ):
            ds = self.accessor._combiner_function(
                [raw_ds],
                item,
                self.catalog,
                assets=[item.assets["red"]],
                apply_scaling=False,
                bbox=(1.0, 2.0, 3.0, 4.0),
                spatial_res=10.0,
            )

        self.assertIn("red", ds)
        self.assertEqual(regular_from_bbox_mock.call_count, 1)
        self.assertEqual(
            regular_from_bbox_mock.call_args.kwargs["bbox"], (1.0, 2.0, 3.0, 4.0)
        )
        self.assertEqual(regular_from_bbox_mock.call_args.kwargs["xy_res"], 10.0)
        self.assertEqual(regular_from_bbox_mock.call_args.kwargs["crs"], source_gm.crs)
        self.assertEqual(
            tuple(regular_from_bbox_mock.call_args.kwargs["tile_size"]),
            tuple(_CHUNK_SIZE.values()),
        )
        resample_mock.assert_called_once()

    def test_open_ardc_pops_stac_item_id_and_adds_attributes(self):
        grouped_data = np.empty((1, 1), dtype=object)
        grouped_data[0, 0] = []
        grouped_items = xr.DataArray(
            grouped_data,
            dims=("time", "tile_id"),
            coords={
                "time": np.array(["2020-12-04"], dtype="datetime64[ns]"),
                "tile_id": ["047027"],
            },
        )
        ds = xr.Dataset(attrs={"stac_item_id": "should-be-removed"})
        returned = xr.Dataset(attrs={"stac_item_id": "should-be-removed"})

        with (
            patch.object(
                self.ardc_accessor, "_group_items", return_value=grouped_items
            ) as group_items_mock,
            patch.object(
                self.ardc_accessor, "_generate_cube", return_value=ds
            ) as generate_cube_mock,
            patch(
                "xcube_stac.accessors.landsat.add_attributes",
                return_value=returned,
            ) as add_attributes_mock,
        ):
            result = self.ardc_accessor.open_ardc(
                "landsat-data",
                [self.make_item(("lwir11",))],
                bbox=(0.0, 0.0, 1.0, 1.0),
                crs="EPSG:32633",
                spatial_res=30.0,
            )

        self.assertIs(result, returned)
        self.assertNotIn("stac_item_id", ds.attrs)
        group_items_mock.assert_called_once()
        generate_cube_mock.assert_called_once()
        add_attributes_mock.assert_called_once()

    def test_group_items_groups_by_date_and_wrs_path_row(self):
        items = []
        for date, path, row in [
            ("2020-12-04T10:15:00", 47, 27),
            ("2020-12-04T11:15:00", 47, 27),
            ("2020-12-05T10:15:00", 48, 28),
        ]:
            item = pystac.Item(
                id=f"LC08_L2SP_{path:03d}{row:03d}_20201204_02_T1_{date[-8:]}",
                geometry=None,
                bbox=[0, 0, 1, 1],
                datetime=datetime.datetime.fromisoformat(date),
                properties={
                    "landsat:wrs_path": path,
                    "landsat:wrs_row": row,
                },
            )
            items.append(item)

        grouped = self.ardc_accessor._group_items(items)

        self.assertEqual(grouped.dims, ("time", "tile_id"))
        self.assertEqual(list(grouped.tile_id.values), ["047027", "048028"])
        self.assertEqual(len(grouped.time.values), 2)
        self.assertEqual(
            len(grouped.sel(time=grouped.time.values[0], tile_id="047027").item()),
            2,
        )
        self.assertEqual(
            grouped.sel(time=grouped.time.values[1], tile_id="048028").item(),
            [items[2]],
        )
