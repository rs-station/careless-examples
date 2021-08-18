import reciprocalspaceship as rs
import torch
from torch import distributions as td
from tqdm import trange
import numpy as np

inFN = "../hewl_ssad/unmerged.mtz"
outFN = "careless_zero_out.mtz"


metadata_keys = [
    'dHKL',
    'XDET',
    'YDET',
    'BATCH',
]

intensity_key = 'IPR'
sigma_intensity_key = 'SIGIPR'

mtz = rs.read_mtz(inFN).compute_dHKL().reset_index()
mtz = mtz[~mtz.label_absences()['ABSENT']]
mtz.loc[:,['Hobs','Kobs','Lobs']] = mtz.loc[:,['H', 'K', 'L']].to_numpy('float32')
mtz.hkl_to_asu(inplace=True)
mtz['miller_id'] = mtz.groupby(['H', 'K', 'L']).ngroup()

miller_id = mtz['miller_id'].to_numpy()
centric = mtz.label_centrics().groupby('miller_id').first()['CENTRIC'].to_numpy()
multiplicity = mtz.compute_multiplicity().groupby('miller_id').first()['EPSILON'].to_numpy(np.float32)
metadata = mtz[metadata_keys].to_numpy(np.float32)
intensities = mtz[intensity_key].to_numpy(np.float32)
uncertainties = mtz[sigma_intensity_key].to_numpy(np.float32)

# Convert everything to torch
multiplicity = torch.FloatTensor(multiplicity)
metadata = torch.FloatTensor(metadata)
intensities = torch.FloatTensor(intensities)
uncertainties = torch.FloatTensor(uncertainties)
centric = torch.BoolTensor(centric)
miller_id = torch.LongTensor(miller_id)

steps=10000
n_layers = 20
mc_samples = 3

class FoldedNormalPosterior(torch.nn.Module):
    def __init__(self, loc_init, scale_init, loc_bijector=None, scale_bijector=None):
        super().__init__()
        self._loc   = torch.nn.Parameter(  loc_init, requires_grad=True)
        self._scale = torch.nn.Parameter(scale_init, requires_grad=True)

        if loc_bijector is None:
            loc_bijector = torch.nn.Softplus()
        self.loc_bijector = loc_bijector
        
        if scale_bijector is None:
            scale_bijector = torch.nn.Softplus()
        self.scale_bijector = scale_bijector

    @property
    def loc(self):
        return self.loc_bijector(self._loc)

    @property
    def scale(self):
        return self.scale_bijector(self._scale)

    @property
    def base_dist(self):
        return td.Normal(self.loc, self.scale)

    @property
    def dist(self):
        from pyro.distributions import FoldedDistribution
        return FoldedDistribution(self.base_dist)

    @property
    def mean(self):
        return self.dist.mean

    @property
    def stddev(self):
        return self.dist.stddev

    def rsample(self, *args, **kwargs):
        return self.dist.rsample(*args, **kwargs)

    def log_prob(self, value):
        return self.dist.log_prob(value)

class WilsonPrior(torch.nn.Module):
    def __init__(self, centric, multiplicity):
        super().__init__()
        self.centric      = torch.nn.Parameter(     centric, requires_grad=False)
        self.multiplicity = torch.nn.Parameter(multiplicity, requires_grad=False)

    @property
    def p_acentric(self):
        return td.Weibull(2., torch.sqrt(self.multiplicity))

    @property
    def p_centric(self):
        return td.HalfNormal(self.multiplicity)

    def log_prob(self, value):
        return torch.where(
            self.centric, 
            self.p_centric.log_prob(value), 
            self.p_acentric.log_prob(value)
            )

    @property
    def mean(self):
        return torch.where(
            self.centric, 
            self.p_centric.mean, 
            self.p_acentric.mean,
            )

    @property
    def stddev(self):
        return torch.where(
            self.centric, 
            self.p_centric.stddev, 
            self.p_acentric.stddev,
            )

class VariationalMergingModel(torch.nn.Module):
    def __init__(self, surrogate_posterior, prior, n_layers=20):
        super().__init__()
        self.surrogate_posterior = surrogate_posterior
        self.prior = prior

        #Construct scale function
        n,d = metadata.shape
        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Linear(d, d))
            layers.append(torch.nn.LeakyReLU(0.1))
        layers.append(torch.nn.Linear(d, 2))
        self.NN = torch.nn.Sequential(*layers)

    def likelihood(self, intensities, uncertainties):
        return td.Normal(loc=intensities, scale=uncertainties)

    def minus_elbo(self, miller_id, metadata, intensities, uncertainties, mc_samples=1):
        q = self.surrogate_posterior
        z = q.rsample((mc_samples,))
        F = z[...,miller_id]
        loc, scale = self.NN(metadata).T
        scale = td.transform_to(td.constraints.positive)(scale)
        Sigma = td.Normal(loc, scale).rsample((mc_samples,))
        log_likelihood = self.likelihood(intensities, uncertainties).log_prob(F * F * Sigma).sum()
        log_p_z = self.prior.log_prob(z)
        log_q_z = q.log_prob(z)
        kl_div = (log_q_z - log_p_z).sum()
        return -log_likelihood + kl_div

    def forward(self, miller_id, metadata, intensities, uncertainties, mc_samples=1):
        return self.minus_elbo(miller_id, metadata, intensities, mc_samples)

#Construct variational distributions
p = WilsonPrior(centric, multiplicity)
q = FoldedNormalPosterior(p.mean, p.stddev)

model = VariationalMergingModel(q, p, n_layers)

#Move everything to the gpu
model = model.cuda()
miller_id = miller_id.cuda()
metadata = metadata.cuda()
intensities = intensities.cuda()
uncertainties = uncertainties.cuda()

#Train the model
optimizer = torch.optim.Adam(model.parameters())

for i in trange(steps):
    optimizer.zero_grad()
    loss = model(miller_id, metadata, intensities, uncertainties, mc_samples)
    loss.backward()
    optimizer.step()

#Export the results
F,SigF = q.mean.detach().numpy(), q.stddev.detach().numpy()

output = rs.DataSet({
    'H' : mtz.groupby('miller_id').first()['Hasu'].astype('H'),
    'K' : mtz.groupby('miller_id').first()['Kasu'].astype('H'),
    'L' : mtz.groupby('miller_id').first()['Lasu'].astype('H'),
    'F' : rs.DataSeries(q.mean().numpy(), dtype='F'),
    'SIGF' : rs.DataSeries(q.stddev().numpy(), dtype='Q'),
    }, 
    cell=mtz.cell, 
    spacegroup=mtz.spacegroup
).set_index(['H', 'K', 'L'])
output.write_mtz(outFN)
