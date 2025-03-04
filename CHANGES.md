## Changes in 0.2.0


* CDSE Sentinel-2 Viewing Angle Dataset: The dimension of the viewing angle dataset 
  has been updated to `(time, angle, band, angle_y, angle_x)`.  
* The package `_utils` has been made public and is now called `utils`.
* All data opener-specific open parameters are now consolidated into a single keyword
  argument. This argument holds a dictionary where keys follow the structure
  `<data_type>:<format_id>`.
* Fixed a bug in `"xcube-stac"` for `data_id="sentinel-2-l2a"`, where the filter 
  intended to exclude items with invalid bounding boxes in STAC metadata was
  not functioning correctly.
* Fixed a bug in `"xcube-stac"` for `data_id="sentinel-2-l2a"`: when crossing the utm
  zone, one tile will be reprojected while the other tile is only resampled. There was 
  an error with difference naming of the crs variable, which has been fixed now. 
* Fixed a bug in `"xcube-stac"` for `data_id="sentinel-2-l2a"` where the naming of the
  CRS variable tiles crossing UTM zones were handled inconsistently, depending on if
  reprojection or simple resampling has been applied.
* Fixed a bug in `"xcube-stac"` for `data_id="sentinel-2-l2a"` where tiles crossing UTM
  zones were handled inconsistently due to different naming of the CRS variable
  stemming from xcube, depending on whether reprojection or simple resampling was
  applied.



## Changes in 0.1.0

* Initial version of STAC Data Store.
