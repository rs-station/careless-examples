# Using `careless.stats`

This notebook describes all the machine learning and data crossvalidation metrics that `careless` can compute. As an example, we describe the metrics from the analysis of anomalous signal from a thermolysin serial femtosecond crystallography dataset, as done in [Hekstra, Wang et al. 2025](https://www.biorxiv.org/content/10.1101/2024.07.22.604476v1). This notebook heavily draws on explanations from [Aldama, Dalton, and Hekstra 2023](https://journals.iucr.org/d/issues/2023/09/00/qi5002/index.html), which is good prerequisite reading for this notebook; this notebook also breaks down the common validation scripts found in `dw-examples`. 

# Table of Contents
1. [Elementary crystallographic statistics](#Elementarycrystallographicstatistics)
2. [Validation statistics](#Validationstatistics)
3. [Machine learning metrics](#Machinelearningmetrics)


## Elementary crystallographic statistics <a name="Elementarycrystallographicstatistics"></a>

### Completeness and multiplicity
`careless.stats` also computes standard crystallography statistics, such as completeness and $I/\sigma(I)$. 

```
>careless.completeness ./thermolysin_dw/thl_1p8A_grid_both.mtz -o ./thermolysin_dw/completeness.csv -b 10
```
![image](./thermolysin_dw/completeness.png)

Now, in python: 
```
pd.read_csv("./thermolysin_dw/completeness.csv")
```
![image](./thermolysin_dw/pd_completeness.png)

The completeness of the last resolution bin is 0.88, and the overall completeness is 0.984. 

```careless.isigi ./thermolysin_dw/thl_1p8A_grid_xval_[0-1].mtz --intensity-key I --uncertainty-key SigI -i ./thermolysin_dw/isigi.png -b 10```

![image](./thermolysin_dw/isigi.png)



## Validation statistics <a name="Validationstatistics"></a>

### $CC_\text{pred}$


Careless provides several validation statistics that may help validate the quality of the merged data. The first is the $CC_\text{pred}$, the correlation coefficient between the intensities predicted by the model and the observed intensities. A typical $CC_\text{pred}$ depends on the data quality and declines as a function of resolution bin, as is the case for most examples in [Hekstra, Wang et al. 2025](https://www.biorxiv.org/content/10.1101/2024.07.22.604476v1). The $CC_\text{pred}$ is computed for a training set and a test set ($CC_\text{pred, train}$ and ($CC_\text{pred, test}$). A large gap between the two (>0.05) indicates that there is overfitting.

```
careless.ccpred ./thermolysin_dw/thl_1p8A_grid_predictions_[0-1].mtz -o ./thermolysin_dw/ccpred_res.csv \
--image ./thermolysin_dw/ccpred_careless.png -b 10 -m spearman -l 0.5 1.1 \
```

The raw plot from the above shell command is reproduced below. 

![image](./thermolysin_dw/ccpred_careless.png)

The train-test gap is minimal until the last resolution bin. 

### $CC_\text{1/2}$

Careless can provide estimates of the $CC_\text{1/2}$, the half-data-set correlation coefficient. We recommend using this measure to determine the resolution cutoff ($CC_\text{1/2}$ > 0.3). This measure is determined from the cross-validation datasets, the number of which are specified using the `--half-dataset-repeats` flag.

```
careless.cchalf ./thermolysin_dw/*xval_both.mtz -o ./thermolysin_dw/cchalf_res.csv \
 -i ./thermolysin_dw/cchalf_res.png -b 10 -m spearman -l 0. 1. \
 ```

![image](./thermolysin_dw/cchalf_res.png)

$CC_\text{1/2}$ is very well-estimated by half datasets (small error bars over the half-dataset repeats) and > 0.3, even in the last resolution bin. 

### $CC_\text{anom}$

For anomalous datasets, we may estimate the half-dataset correlation of the anomalous difference signal. 

```
careless.ccanom ./thermolysin_dw/*xval_both.mtz -o ./thermolysin_dw/ccanom_res.csv \
--image ./thermolysin_dw/ccanom_res.png -b 10 -l 0. 0.3
```

The $CC_\text{anom}$ is not expected to be large, and expected to decrease as a function of resolution. We observe some high-resolution tailing in datasets merged with a bivariate prior on the input structure factors. This is due to numerical issues that have been fixed not during the processing of this dataset but by `careless 0.4.9`. 

### $R_\text{split}$

The $R_\text{split}$ is a crossvalidation metric for serial crystallography data and is described in White et al. 2012. The data are split in half and then the degree to which the half-datasets agree is quantified per resolution bin. 
```
careless.rsplit ./thermolysin_dw/*xval_both.mtz -o ./thermolysin_dw/rsplit_res.csv --image ./thermolysin_dw/rsplit_res.png -b 10 -l 0 0.3
```
![image](./thermolysin_dw/rsplit_res.png)

A lower $R_\text{split}$ is better. 

### Image $CC_\text{pred}$
```
careless.image_ccpred ./thermolysin_dw/thl_1p8A_grid_predictions_[0-1].mtz -o ./thermolysin_dw/imagecc_res.csv \
--image ./thermolysin_dw/imagecc_careless.png -m weighted -l 0. 1.1
```

![image](./thermolysin_dw/imagecc_careless.png)

The above plot displays the CCpred for each image. Images with low CCpred are either misindexed or their scales are poorly modeled. 


## Machine learning metrics <a name="Machinelearningmetrics"></a>

`careless` is a package for scaling and merging crystallographic data using variational inference. Every time I process a dataset using `careless`, I will learn the distribution of the structure factor amplitudes and the scales for each miller index, given the intensity of each observed reflection in the input dataset. The scale function is implemented as a neural network, whose parameters are optimized by minimizing an objective function ('loss'). 

```
import careless.stats
import matplotlib.pyplot as plt
import pandas as pd

training_metrics = pd.read_csv("./thermolysin_dw/thl_1p8A_grid_history.csv")
training_metrics.plot(x="step",y="loss", loglog=True)
plt.savefig("./thermolysin_dw/loss.png")
plt.show()

plt.clf()

fig,ax = plt.subplots()
training_metrics.plot(x="step",y="NLL", ax=ax,loglog=True)
training_metrics.plot(x="step",y="NLL_val", ax=ax, loglog=True)
training_metrics.plot(x="step",y="F KLDiv", ax=ax, loglog=True)
plt.savefig("./thermolysin_dw/loss_components.png")
plt.show()
```
![image](./thermolysin_dw/loss.png)


We see that the loss goes down over the 30000 steps of training. This loss is smooth. If the loss is jagged, then I advise increasing `--mc-samples` for smoother training. Training takes about 17.5 minutes as according to the below log:

``` 
> head -c 16k ./thermolysin_dw/myoutput_19756592.err

Training:   0%|                                                                                               | 0/30000 [00:00<?, ?it/s]WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1707949336.689264 1893395 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
Training:   0%| | 92/30000 [00:15<17:11, 29.01it/s, Grad Norm=1.08e+07, loss=3.59e+07, F KLDiv=1.28e+06, NLL=3.47e+07, NLL_val=3.48e+07]
```

This is the careless training log, which, at each step, reports all the machine learning metrics. We will go through most of these metrics now. 

Our objective function is the standard ELBO, whose negative we minimize:
$${{{{{{{-\rm{ELBO}}}}}}}}\left(q\right)={-{\mathbb{E}}}_{q}\left[\log p(I|F,{{\Sigma }})\right]+{D}_{KL}\left({q}_{F}\parallel p(F)\right) $$

The first term on the right-hand side is the negative log likelihood (NLL), which represents the degree to which the structure factors faithfully represent the experimental intensities. The secod term is the Kullback-Liebler divergence between the variational distribution on the structure factor amplitudes, and a prior distribution. This term measures the similarity of two distributions that is minimal (zero) when the two are identical. We now plot the NLL, KL divergence (KLDiv), and the NLL_val, which is the NLL on a held-out validation set of reflections (the fraction of the data for which we can specify by `--test-fraction`. I typically choose `--test-fraction=0.1`. 

![image](./thermolysin_dw/loss_components.png)

The NLL and NLL_val are almost identical, indicating minimal overfitting, and both go down, indicating that the the neural network is learning the scales and structure factor amplitudes. The KLDiv actually goes down, as the structured prior introduced by the double-Wilson distribution is initially distinct from the factorized prior. 

All these statistics can also be visualized with the command line: 

`careless.plot_history -s ./thermolysin_dw/thl_1p8A_grid_history.csv -o ./thermolysin_dw/thl_plot_history.png`

![image](./thermolysin_dw/thl_plot_history.png)
