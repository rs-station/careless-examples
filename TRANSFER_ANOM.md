### Boosting Anomalous Signal with Transfer Learning

Ever since `careless` v0.2.1, it is possible to load scale model weights from previous runs. This makes it possible to boost certain signals by exploiting broken symmetries in diffraction data. This example demonstrates the concept using lysozyme sulfur anomalous data. 

The basic concept is to merge in two steps
 1) Merge without the `--anomalous` flag. The scale function weights from this step will be optimized to consistently estimate Friedel mates. 
 2) Merge using `--anomalous` flag and the scale weights from (1). Keeping the scaling model frozen, estimate structure factor amplitudes for both Friedel mates.

### Step 1: Merging without anomalous
In `careless-examples/hewl_ssad`, run the command

```bash
mkdir no_anom
careless mono  \
    --studentt-likelihood-dof=16 \
    --mc-samples=20 \
    --mlp-layers=10 \
    --image-layers=2 \
    "BATCH,dHKL,Hobs,Kobs,Lobs,XDET,YDET,BG,SIGBG,LP,QE,FRACTIONCALC" \
    unmerged.mtz \
    no_anom/hewl
```
to merge both halves of reciprocal space together. This command will save the optimal neural network weights to `no_anom/hewl_scale`. 
These weights are optimized to make consistent estimates across Friedel mates. Therefore, when we use them later, 
they can be expected to make sure that the two centrosymmetrically related halves of reciprocal space are on the same scale. 

### Step 2: Transfer learning with anomalous
In `careless-examples/hewl_ssad`, run the command
```bash
mkdir anom
careless mono  \
    --freeze-scale \
    --scale-file=no_anom/hewl_scale \
    --anomalous \
    --studentt-likelihood-dof=16 \
    --mc-samples=20 \
    --mlp-layers=10 \
    --image-layers=2 \
    "BATCH,dHKL,Hobs,Kobs,Lobs,XDET,YDET,BG,SIGBG,LP,QE,FRACTIONCALC" \
    unmerged.mtz \
    anom/hewl
```
to merge the Friedel mates separately using the scales from (1). The `--freeze-scale` argument is used to keep the weights
constant during merging. 

### Analysis of output
To generate the anomalous omit map referenced in the appendix, use the following command:

```
mkdir phenix_omit
phenix.refine refine_omit.eff anom/hewl_0.mtz
```

This will create refinement results in `phenix_omit` which you can inspect with coot. 
