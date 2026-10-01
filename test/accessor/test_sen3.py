import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import dask.array as da
import numpy as np
import pystac
import xarray as xr

from xcube_stac.accessors.sen3 import (
    Sen3CdseStacItemAccessor,
    Sen3LstNrtCdseStacItemAccessor,
    Sen3LstNtcCdseStacItemAccessor,
    Sen3LstNtcPlanetaryComputerStacItemAccessor,
    Sen3PlanetaryComputerStacItemAccessor,
    _apply_scaling,
    _clean_masks,
    _group_items,
    orthorectify_geolocation,
)

from ..sampledata import (
    sentinel_3_lst_data,
    sentinel_3_lst_flag_data,
    sentinel_3_lst_geolocation_data,
    sentinel_3_syn_cloud_data,
    sentinel_3_syn_data,
    sentinel_3_syn_geolocation_data,
)


class Sen3LstAccessorTest(unittest.TestCase):
    def setUp(self):
        catalog = pystac.Catalog(id="test", description="test")
        self.accessor = Sen3LstNtcCdseStacItemAccessor(catalog)
        self.item = pystac.Item(
            id="test-item",
            geometry=None,
            bbox=None,
            datetime=datetime(2020, 1, 1, tzinfo=timezone.utc),
            properties={},
            assets={
                "LST_in": pystac.Asset(href="lst"),
                "flags_in": pystac.Asset(href="flags"),
                "geodetic_in": pystac.Asset(href="geolocation"),
            },
        )

    def test_open_item_selects_and_scales_uncertainty(self):
        with patch.object(
            self.accessor,
            "open_asset",
            side_effect=[
                sentinel_3_lst_data(),
                sentinel_3_lst_geolocation_data(),
            ],
        ):
            dataset = self.accessor.open_item(
                self.item,
                asset_names=["LST_uncertainty"],
                apply_geo_orthorectification=False,
                apply_rectification=False,
            )

        self.assertEqual(["LST_uncertainty"], list(dataset.data_vars))
        self.assertAlmostEqual(
            0.0002,
            float(dataset["LST_uncertainty"].isel(band=0, x=0, y=0).compute()),
        )

    def test_open_data_schema_contains_uncertainty(self):
        schema = self.accessor.get_open_data_params_schema()
        asset_names = schema.properties["asset_names"]
        self.assertIn("LST_uncertainty", asset_names.items.enum)

    def test_open_item_selects_flags_and_uncertainty(self):
        with patch.object(
            self.accessor,
            "open_asset",
            side_effect=[
                sentinel_3_lst_data(),
                sentinel_3_lst_flag_data(),
                sentinel_3_lst_geolocation_data(),
            ],
        ):
            dataset = self.accessor.open_item(
                self.item,
                asset_names=["LST_uncertainty", "confidence_in"],
                apply_geo_orthorectification=False,
                apply_rectification=False,
            )

        self.assertCountEqual(
            ["LST_uncertainty", "confidence_in"], list(dataset.data_vars)
        )

    def test_open_item_can_read_flags_without_lst(self):
        with patch.object(
            self.accessor,
            "open_asset",
            side_effect=[sentinel_3_lst_flag_data(), sentinel_3_lst_geolocation_data()],
        ):
            dataset = self.accessor.open_item(
                self.item,
                asset_names=["confidence_in"],
                apply_geo_orthorectification=False,
                apply_rectification=False,
            )

        self.assertEqual(["confidence_in"], list(dataset.data_vars))


class Sen3SynAccessorTest(unittest.TestCase):
    def setUp(self):
        self.accessor = Sen3CdseStacItemAccessor(
            pystac.Catalog(id="test", description="test")
        )
        self.item = pystac.Item(
            id="test-item",
            geometry=None,
            bbox=None,
            datetime=datetime(2020, 1, 1, tzinfo=timezone.utc),
            properties={},
            assets={
                "syn_Oa01_reflectance": pystac.Asset(
                    href="band", media_type="application/netcdf"
                ),
                "flags": pystac.Asset(href="flags", media_type="application/netcdf"),
                "geolocation": pystac.Asset(
                    href="geolocation", media_type="application/netcdf"
                ),
            },
        )

    def test_open_item_selects_band_and_flag_without_error_band(self):
        with patch.object(
            self.accessor,
            "open_asset",
            side_effect=[
                sentinel_3_syn_data(),
                sentinel_3_syn_cloud_data(),
                sentinel_3_syn_geolocation_data(),
            ],
        ):
            dataset = self.accessor.open_item(
                self.item,
                asset_names=["SDR_Oa01", "CLOUD_flags"],
                add_error_bands=False,
                apply_rectification=False,
            )

        self.assertCountEqual(["SDR_Oa01", "CLOUD_flags"], dataset.data_vars)
        self.assertEqual("test-item", dataset.attrs["stac_item_id"])
        self.assertIn("scale_factor", dataset["SDR_Oa01"].attrs)

    def test_open_asset_unwraps_rasterio_list(self):
        asset = pystac.Asset(href="band", media_type="application/netcdf")
        with patch(
            "xcube_stac.accessors.sen3.rioxarray.open_rasterio",
            return_value=[sentinel_3_lst_data(), sentinel_3_lst_data()],
        ):
            dataset = Sen3LstNtcCdseStacItemAccessor(
                pystac.Catalog(id="test", description="test")
            ).open_asset(asset)

        self.assertIn("LST", dataset.data_vars)


class Sen3HelperTest(unittest.TestCase):
    def test_clean_masks_adds_fill_value_flag(self):
        flags = xr.Dataset(
            {
                "flags": xr.DataArray(
                    np.zeros((2, 2), dtype=np.uint16),
                    dims=("y", "x"),
                    attrs={
                        "flag_masks": np.array([1, 2], dtype=np.uint16),
                        "flag_meanings": "a b",
                        "_FillValue": 65535,
                        "scale_factor": 1,
                        "add_offset": 0,
                    },
                )
            }
        )

        result = _clean_masks(flags)
        self.assertNotIn("_FillValue", result["flags"].attrs)
        self.assertIn("fill_value", result["flags"].attrs)
        self.assertNotIn("scale_factor", result["flags"].attrs)

    def test_apply_scaling_masks_fill_and_applies_offset(self):
        dataset = xr.Dataset(
            {
                "value": xr.DataArray(
                    np.array([1, -9999], dtype=np.int16),
                    dims="x",
                    attrs={
                        "_FillValue": -9999,
                        "scale_factor": 2,
                        "add_offset": 10,
                    },
                )
            }
        )

        result = _apply_scaling(dataset)

        np.testing.assert_allclose(result["value"].values, [12, np.nan], equal_nan=True)

    def test_group_items_sorts_by_date_and_orbit(self):
        def item(item_id, timestamp, orbit):
            return pystac.Item(
                id=item_id,
                geometry=None,
                bbox=[0, 0, 1, 1],
                datetime=timestamp,
                properties={"sat:orbit_state": orbit},
            )

        items = [
            item(
                "ascending", datetime(2020, 1, 1, 12, tzinfo=timezone.utc), "ascending"
            ),
            item(
                "descending", datetime(2020, 1, 1, 6, tzinfo=timezone.utc), "descending"
            ),
            item(
                "next-day", datetime(2020, 1, 2, 6, tzinfo=timezone.utc), "descending"
            ),
        ]

        grouped = _group_items(items)

        self.assertEqual(3, grouped.sizes["time"])
        self.assertEqual(
            "descending", grouped[0].item()[0].properties["sat:orbit_state"]
        )
        self.assertEqual(
            "ascending", grouped[1].item()[0].properties["sat:orbit_state"]
        )

    def test_orthorectify_geolocation_corrects_latitude(self):
        shape = (2, 2)
        coords = {
            "lat": (("y", "x"), da.full(shape, 50.0, chunks=shape)),
            "lon": (("y", "x"), da.from_array([[0.0, 1.0], [0.0, 1.0]], chunks=shape)),
        }
        dataset = xr.Dataset(
            {"elev": (("y", "x"), da.full(shape, 1000.0, chunks=shape))},
            coords=coords,
        )
        angles = xr.Dataset(
            {
                "sat_zenith_tn": (("y", "x"), da.full(shape, 45.0, chunks=shape)),
                "sat_azimuth_tn": (("y", "x"), da.zeros(shape, chunks=shape)),
            },
            coords={
                "lon": (
                    ("y", "x"),
                    da.from_array([[0.0, 1.0], [0.0, 1.0]], chunks=shape),
                )
            },
        )

        result = orthorectify_geolocation(dataset, angles).compute()

        self.assertTrue(np.all(result.lat.values < dataset.lat.values))
        np.testing.assert_allclose(result.lon.values, dataset.lon.values)


class Sen3PlanetaryComputerTest(unittest.TestCase):
    def test_pc_signed_detection(self):
        signed = pystac.Item(
            id="signed",
            geometry=None,
            bbox=None,
            datetime=datetime(2020, 1, 1, tzinfo=timezone.utc),
            properties={},
            assets={"data": pystac.Asset(href="https://example.test/data?sig=abc")},
        )
        unsigned = pystac.Item(
            id="unsigned",
            geometry=None,
            bbox=None,
            datetime=datetime(2020, 1, 1, tzinfo=timezone.utc),
            properties={},
            assets={"data": pystac.Asset(href="https://example.test/data")},
        )

        self.assertTrue(Sen3PlanetaryComputerStacItemAccessor._is_pc_signed(signed))
        self.assertFalse(Sen3PlanetaryComputerStacItemAccessor._is_pc_signed(unsigned))

    def test_pc_lst_accessor_uses_pc_asset_names(self):
        accessor = Sen3LstNtcPlanetaryComputerStacItemAccessor(
            pystac.Catalog(id="test", description="test")
        )

        self.assertEqual("lst-in", accessor._lst_name)
        self.assertIn(
            "LST_uncertainty",
            accessor.get_open_data_params_schema().properties["asset_names"].items.enum,
        )

    def test_nrt_schema_uses_nrt_flag_names(self):
        accessor = Sen3LstNrtCdseStacItemAccessor(
            pystac.Catalog(id="test", description="test")
        )
        enum = (
            accessor.get_open_data_params_schema().properties["asset_names"].items.enum
        )

        self.assertIn("counter_water_in", enum)
        self.assertNotIn("bayes_orphan_in", enum)
