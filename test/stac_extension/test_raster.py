# The MIT License (MIT)
# Copyright (c) 2024 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
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

import unittest
import datetime

import pystac
import numpy as np
import xarray as xr

from xcube_stac.stac_extension.raster import apply_offset_scaling


def create_raster_stac_item_v1() -> pystac.Item:
    item_id = "example-item"
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]
        ],
    }
    bbox = [100.0, 0.0, 101.0, 1.0]
    dt = datetime.datetime(2023, 1, 1, 0, 0, 0)
    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=dt,
        properties={},
        stac_extensions=["https://stac-extensions.github.io/raster/v1.1.0/schema.json"],
    )

    for asset_name in ["B01", "B02"]:
        asset_href = f"https://example.com/data/{asset_name}.tif"
        asset = pystac.Asset(href=asset_href, media_type="image/tiff", roles=["data"])
        asset.extra_fields["raster:bands"] = [
            dict(
                nodata=0,
                scale=0.1,
                offset=-0.05,
                spatial_resolution=10.0,
                unit="meters",
            )
        ]
        item.add_asset(asset_name, asset)
    return item


def create_raster_stac_item_v2() -> pystac.Item:
    item_id = "example-item"
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]
        ],
    }
    bbox = [100.0, 0.0, 101.0, 1.0]
    dt = datetime.datetime(2023, 1, 1, 0, 0, 0)
    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=dt,
        properties={},
        stac_extensions=["https://stac-extensions.github.io/raster/v2.0.0/schema.json"],
    )

    for asset_name in ["B01", "B02"]:
        asset_href = f"https://example.com/data/{asset_name}.tif"
        asset = pystac.Asset(href=asset_href, media_type="image/tiff", roles=["data"])
        asset.extra_fields["nodata"] = 0
        asset.extra_fields["raster:scale"] = 0.1
        asset.extra_fields["raster:offset"] = -0.05
        item.add_asset(asset_name, asset)
    return item


class RasterTest(unittest.TestCase):

    def test_apply_offset_scaling(self):
        item_v1 = create_raster_stac_item_v1()
        item_v2 = create_raster_stac_item_v2()
        ds_v1 = xr.Dataset()
        ds_v1["B01"] = xr.DataArray(
            data=np.array([[0, 3, 3], [1, 1, 1], [2, 2, 2]]),
            dims=("y", "x"),
            coords=dict(y=[5000, 5010, 5020], x=[7430, 7440, 7450]),
        )
        ds_v1["B02"] = xr.DataArray(
            data=np.array([[3, 3, 3], [1, 1, 1], [2, 0, 2]]),
            dims=("y", "x"),
            coords=dict(y=[5000, 5010, 5020], x=[7430, 7440, 7450]),
        )
        ds_v2 = ds_v1.copy()
        ds_mod_v1 = apply_offset_scaling(ds_v1, item_v1, "B01")
        ds_mod_v2 = apply_offset_scaling(ds_v2, item_v2, "B01")
        ds_mod_expected = xr.Dataset()
        ds_mod_expected["B01"] = xr.DataArray(
            data=np.array(
                [[np.nan, 0.25, 0.25], [0.05, 0.05, 0.05], [0.15, 0.15, 0.15]]
            ),
            dims=("y", "x"),
            coords=dict(y=[5000, 5010, 5020], x=[7430, 7440, 7450]),
        )
        ds_mod_expected["B02"] = xr.DataArray(
            data=np.array([[3, 3, 3], [1, 1, 1], [2, 0, 2]]),
            dims=("y", "x"),
            coords=dict(y=[5000, 5010, 5020], x=[7430, 7440, 7450]),
        )
        xr.testing.assert_allclose(ds_mod_expected, ds_mod_v1)
        xr.testing.assert_allclose(ds_mod_expected, ds_mod_v2)

        ds_mod2_v1 = apply_offset_scaling(ds_mod_v1, item_v1, "B02")
        ds_mod2_v2 = apply_offset_scaling(ds_mod_v2, item_v2, "B02")
        ds_mod_expected2 = xr.Dataset()
        ds_mod_expected2["B01"] = xr.DataArray(
            data=np.array(
                [[np.nan, 0.25, 0.25], [0.05, 0.05, 0.05], [0.15, 0.15, 0.15]]
            ),
            dims=("y", "x"),
            coords=dict(y=[5000, 5010, 5020], x=[7430, 7440, 7450]),
        )
        ds_mod_expected2["B02"] = xr.DataArray(
            data=np.array(
                [[0.25, 0.25, 0.25], [0.05, 0.05, 0.05], [0.15, np.nan, 0.15]]
            ),
            dims=("y", "x"),
            coords=dict(y=[5000, 5010, 5020], x=[7430, 7440, 7450]),
        )
        xr.testing.assert_allclose(ds_mod_expected2, ds_mod2_v1)
        xr.testing.assert_allclose(ds_mod_expected2, ds_mod2_v2)

        with self.assertLogs("xcube.stac", level="WARNING") as cm:
            ds_mod_v1 = apply_offset_scaling(ds_v1, item_v1, "B01", raster_version="v3")
        xr.testing.assert_allclose(ds_v1, ds_mod_v1)
        self.assertEqual(1, len(cm.output))
        msg = (
            "WARNING:xcube.stac:Stac extension raster exists only for version 'v1' "
            "and 'v2', not for 'v3'. No scaling is applied."
        )
        self.assertEqual(msg, str(cm.output[-1]))

        item_v1.stac_extensions = []
        with self.assertLogs("xcube.stac", level="WARNING") as cm:
            ds__mod_v1 = apply_offset_scaling(ds_v1, item_v1, asset_name="B01")
        xr.testing.assert_allclose(ds_v1, ds__mod_v1)
        self.assertEqual(1, len(cm.output))
        msg = (
            "WARNING:xcube.stac:The item 'example-item' is not conform to "
            "the stac-extension 'raster'. No scaling is applied."
        )
        self.assertEqual(msg, str(cm.output[-1]))
