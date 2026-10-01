import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pystac

from xcube_stac.accessors.sen3 import Sen3LstNtcCdseStacItemAccessor

from ..sampledata import (
    sentinel_3_lst_data,
    sentinel_3_lst_geolocation_data,
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
