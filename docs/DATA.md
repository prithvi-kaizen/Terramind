# Data

The project uses **GeoCH4PlumeNet**, available at [Zenodo record 16813369](https://zenodo.org/records/16813369), DOI `10.5281/zenodo.16813369`. The record describes 1,000 samples with simulated plumes, 13-band 512×512 Sentinel-2 L1C GeoTIFFs, masks, and an original 70/10/20 train/validation/test split. Labels derive from simulated plume transmissivity. They are not field measurements or analyst-drawn masks. The dataset contains modified Copernicus Sentinel data [2024].

The downloaded training corpus is not included here. Retrieve the archives and metadata CSV from the record and retain the original split directories for future evaluation. Do not merge the official test set into training.

## Historical notebook layout

The imported notebooks expect this flat layout:

```text
data/raw/
  Locations_and_tiles.csv
  0___0_S2B_20240326T165849_R1000_nonorm_bands20m.tiff
  ... image tiles ...
  label/
    0___0_S2B_20240326T165849_R1000_bin_label.tiff
    ... matching masks ...
```

Use only the historical training/validation subset when preparing that layout. The notebooks identify validation tiles by location IDs 8 and 9; saved output records 804 total tiles, including 619 for training and 185 for validation. This is a project-specific split, not the published benchmark split.

Read the CSV with `sep=';'` and `decimal=','`. Match each image and mask by their shared filename prefix. The loader assumes the following band order, based on the notebook's corrected inspection:

```text
B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
```

Consequently, B11 and B12 are array indices 11 and 12 (Rasterio bands 12 and 13). Verify actual band descriptions and dataset documentation before processing tiles from another provider.

## Label interpretation

The notebook converts soft masks to binary labels using `mask > 0.5`. It computes the 33rd and 66th percentiles of training-mask pixel counts for its three severity classes. The filename field `R1000` must not be inferred to be a model emission estimate. The saved exploration shows only that rate value in the historical subset.

The training notebook reports no zero-plume training tiles. A negative-background evaluation is needed before reporting leak-detection false-alarm performance. Pixel-level background within positive tiles is different from a genuinely plume-free scene.

## Included artifacts

`sample_input/` contains one imported tile. `results/` contains its historical predicted mask and alert. These support a small demo, not a training-data distribution. Artifact hashes and original paths are recorded in [PROVENANCE.md](PROVENANCE.md).

The dataset record does not display a named license in the retrieved license field. Confirm its terms with the publisher before redistributing additional data. The repository does not assign new terms to these artifacts.
