## Changes in 1.0.1 (under development)
 
## Changes in 1.0.0

* Restructured the store architecture: the keyword argument `stack_mode` has been
  removed. A new data store, `stac-cdse-ardc`, has been introduced, enabling the
  creation of 3D analysis-ready data cubes from multiple STAC items (observation
  tiles). The README has been updated to provide a concise user guide.
* Added support for analysis-ready data cubes in `stac-cdse-ardc` for **Sentinel-2
  L1C** (`data_id="sentinel-2-l1c"`).
* Added support for analysis-ready data cubes in `stac-cdse-ardc` for **Sentinel-3
  Synergy** products (`data_id="sentinel-3-syn-2-syn-ntc"`).


## Changes in 0.4.0

* Fixed a bug in `stac-cdse` where assigning the nearest resolution of the spectral 
  band to the requested resolution was not working correctly.

## Changes in 0.3.0

* For the xcube-cdse data store, Sentinel-2 pixel data are now first sorted into 
  common native UTM grids per zone, instead of being reprojected individually to the
  user defined target grid. After sorting, resampling and reprojection are applied 
  as needed.
* xcube’s new, faster `reproject_dataset()` method (https://github.com/xcube-dev/xcube/pull/1152)
  is used when reprojection to a different CRS is required.

## Changes in 0.2.0

* CDSE Sentinel-2 Viewing Angle Dataset: The dimension of the viewing angle dataset 
  has been updated to `(time, angle, band, angle_y, angle_x)`.  
* The package `_utils` has been made public and is now called `utils`.
* All data opener-specific open parameters are now consolidated into a single keyword
  argument. This argument holds a dictionary where keys follow the structure
  `<data_type>:<format_id>`.
* Fixed a bug in `"stac-cdse"` for `data_id="sentinel-2-l2a"`, where the filter 
  intended to exclude items with invalid bounding boxes in STAC item was
  not functioning correctly.
* Fixed a bug in `"stac-cdse"` for `data_id="sentinel-2-l2a"` where tiles crossing UTM
  zones were handled inconsistently due to different naming of the CRS variable
  stemming from xcube, depending on whether reprojection or simple resampling was
  applied.


## Changes in 0.1.0

* Initial version of STAC Data Store.
