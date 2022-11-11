### Using a bivariate prior to exploit correlations between Friedel mates

In the example [“Merging serial crystallography from a free electron laser“](https://github.com/rs-station/careless-examples/blob/main/XFEL.md), we see that Careless can extract anomalous signal from XFEL data. 
By default, Careless uses univariate priors (the Wilson distribution) for each structure factor amplitude. 
In practice, however, there are many instances where we expect pairs of structure factor amplitudes to be highly similar. 
In the example [“Boosting SAD signal with transfer learning”](https://github.com/rs-station/careless-examples/blob/main/TRANSFER_ANOM.md), we took advantage of this by preprocessing Friedel mates under the assumption that their values were, in fact, identical. 
We can relax this assumption by, instead, assuming a certain degree of correlation between their true values.

We have implemented a capability to use multivariate priors in Careless. 
Their numerical implementation is an active area of research, and there may still be room for improvement. 
Here, we illustrate the potential of such multivariate priors using anomalous signal from XFEL data as an example.

### Split the Friedel mates into separate files
The software requires that related sets of reflection observations be presented as separate mtz files. 
In our case, this means we need to split the unmerged MTZ file into two files, one for each half of reciprocal space. 

As currently implemented, we first split the unmerged intensities into two MTZs, 
In the `careless-examples/thermolysin_xfel` folder, run 

python friedelize.py unmerged.mtz

to create two files, 
 - `friedel_plus.mtz`
 - `friedel_minus.mtz`

### Merge with the Double-Wilson prior
We then call Careless, with a few modifications relative to the [XFEL example](https://github.com/rs-station/careless-examples/blob/main/XFEL.md). Specifically, 

 1) we need to specify the statistical dependencies between the input MTZs:
`--double-wilson-parents=None,0` 
This is a comma-separated list of integers with one entry per MTZ. Each entry tells careless the parent distribution upon which an MTZ depends. None indicates a data set with no parent distribution which means it will be processed with Wilson's prior.  In principle, this list can be extended, for example for time-resolved crystallographic applications with many time points. 
 2) `--double-wilson-r=0.,0.998` specifies the correlation of the first dataset to ‘None’, and the expected correlation of the second MTZ to the first (here, 0.998). 
 3) `--mc-samples=10` improves accuracy of the evaluation of the loss function. 
 4) `--studentt-likelihood-dof=32` sets up use of a robust loss function like in the first example (Room temp SAD phasing for lysozyme). 
 5) `--refine-uncertainties` uses the error model from Aimless to refine the likelihood function. 

With these considerations, we call Careless as follows:

```bash
careless mono \
  --separate-files \
  --mc-samples=10 \
  --mlp-layers=10 \
  --image-layers=2 \
  --dmin=1.8 \
  --iterations=30_000 \
  --double-wilson-parents=None,0 \
  --double-wilson-r=0.,0.998 \
  --refine-uncertainties \
  --studentt-likelihood-dof=32  \
  dHKL,xobs,yobs,ewald_offset \
  friedel_plus.mtz \
  friedel_minus.mtz \
  merge_dw/thermolysin
```

### Analyze output

This Careless command will output two MTZs each corresponding to half of reciprocal space. We must reassemble the outputs into a single MTZ for downstream analysis in PHENIX. This can be accomplished by

```bash
python unfriedelize.py merge_dw/thermolysin_0.mtz merge_dw/thermolysin_1.mtz merged_anom.mtz
```

Now run refinement in PHENIX using

```bash
mkdir phenix_omit
phenix.refine refine_omit.eff merged_anom.mtz
```

The anomalous omit map generated is stored in the `ANOM` and `PHANOM` columns of `phenix_omit/thermolysin_refine_1.mtz`


This approach results in substantial improvements. Using Careless `v0.2.0`, we obtained the following anomalous peak heights:

|Site | Careless | Careless (EO) | CCTBX | Careless (bivariate) |
| -- | -- | -- | -- | -- |
| Zn-317 | 11.58 | 20.42 | 20.07 | 30.97 |
| Ca-318 | <3.5  |  5.42 |  6.17 |  8.48 |
| Ca-319 | <3.5  |  3.61 |  3.81 |  6.06 |
| Ca-320 | 4.17  |  7.32 |  7.05 |  9.22 |
| Ca-321 | <3.5  |  5.49 |  7.25 |  7.99 |


