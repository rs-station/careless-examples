# careless-examples
Examples accompanying the [`careless`](https://github.com/rs-station/careless) [paper](https://www.nature.com/articles/s41467-022-35280-8). 
These examples have been tested against `careless` release version 0.2.0. 

Examples accompanying our [2025 Science Advances paper](https://www.science.org/doi/full/10.1126/sciadv.adj2921) on the use of multivariate priors with `careless` can be found at the [dw-examples repo](https://github.com/Hekstra-Lab/dw-examples).

## Installation
If you haven't already, follow the instructions on the [readme](https://github.com/rs-station/careless) to install `careless`.
To download the examples, run
```bash
git clone https://github.com/rs-station/careless-examples
```
or [download](https://github.com/rs-station/careless-examples/archive/main.zip) the examples as a `.zip` archive.

## Examples
Examples from the careless manuscript:
- [Room temp SAD phasing for lysozyme](HEWLSSAD.md)
- [Time resolved differences in photoactive yellow protein](PYPTRX.md)
- [Merging serial crystallography from a free electron laser](XFEL.md)


Additional examples:
- For a barebones implementation of the careless model, have a look at [careless_zero](CARELESS_ZERO.md)
- [Boosting SAD signal with transfer learning](TRANSFER_ANOM.md)
- [Using a bivariate prior to exploit correlations between Friedel mates](DOUBLE_WILSON.md)
- A walkthrough of the `careless.stats` module for [computing and visualizing crystallography statistics](STATS.md) 
