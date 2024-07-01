## About this xcube server STAC example

When running the xcube server, a STAC catalog is built dynamically, where each data
sets assigned in the configuration file `config.yml` is linked to a STAC item. To run
the xcube serve follow the section below. To access the data via the STAC using the
`xcube-stac` plugin, refer to the notebook `examples/xcube_server_stac_s3.ipynb`. 

### Running the xcube Server using the configuration stored in `config.yml`
we start the [xcube server](https://xcube.readthedocs.io/en/latest/examples/xcube_serve.html#running-the-server),
by directing into the `xcube-stac` directory and running the
following command line in the terminal: 

```bash
"xcube serve --verbose -c examples/xcube_server/config.yml"
```

### Test data

The following data is used by the configuration `config.yml`:

* `cube-1-250-250.zarr`: data cute in zarr format with
  chunking (time, lat, lon) = (1, 250, 250)

* `cube-1-250-250.levels`: Multi-level/multi-resolution (image pyramid) 
  version of `cube-1-250-250.zarr`.

* `sample-geotiff.tif`: A simple GeoTIFF

* `sample-cog.tif`: A Cloud-optimized GeoTIFF with 3 overview levels downloaded from 
  https://rb.gy/h8qz14. This image is one of many free GeoTIFFs available from
  [Sentinel-2](https://registry.opendata.aws/sentinel-2-l2a-cogs/).

### Access to SpatioTemporalAssetCatalog (STAC)

When running the xcube server, the STAC catalog is published at 
http://localhost:8080/ogc. Individual items can be accessed at
http://localhost:8080/ogc/collections/datacubes/items/{dataset_identifier},
where the dataset_identifier is assigned in the configuration file in the 
`Identifier` field for the respective dataset.  

