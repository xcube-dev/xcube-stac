## Changes in 0.2.0 (under development)


* CDSE Sentinel-2 Viewing Angle Dataset: The dimension of the viewing angle dataset 
  has been updated to `(time, angle, band, angle_y, angle_x)`.  
* The package `_utils` has been made public and is now called `utils`.
* All data opener-specific open parameters are now consolidated into a single keyword
  argument. This argument holds a dictionary where keys follow the structure
  `<data_type>:<format_id>`.  


## Changes in 0.1.0

* Initial version of STAC Data Store.
