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


class VariationalMergingModel(torch.nn.Module):
    def __init__(self, centric, multiplicity, miller_id, metadata, intensities, uncertainties, n_layers=20):
        super().__init__()
        self.centric       = torch.nn.Parameter(centric       , requires_grad=False)
        self.multiplicity  = torch.nn.Parameter(multiplicity  , requires_grad=False)
        self.miller_id     = torch.nn.Parameter(miller_id     , requires_grad=False)
        self.metadata      = torch.nn.Parameter(metadata      , requires_grad=False)
        self.intensities   = torch.nn.Parameter(intensities   , requires_grad=False)
        self.uncertainties = torch.nn.Parameter(uncertainties , requires_grad=False)

        #Construct scale function
        n,d = metadata.shape
        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Linear(d, d))
            layers.append(torch.nn.LeakyReLU(0.1))
        layers.append(torch.nn.Linear(d, 2))
        self.NN = torch.nn.Sequential(*layers)

        #Construct variational distributions
        loc_init   = torch.where(centric, self.p_centric.mean, self.p_acentric.mean)
        scale_init = torch.where(centric, self.p_centric.stddev, self.p_acentric.stddev)

        self.loc   = torch.nn.Parameter(  loc_init, requires_grad=True)
        self.scale = torch.nn.Parameter(scale_init, requires_grad=True)

    @property
    def p_acentric(self):
        return td.Weibull(2., torch.sqrt(self.multiplicity))

    @property
    def p_centric(self):
        return td.HalfNormal(self.multiplicity)

    @property
    def likelihood(self):
        return td.Normal(loc=self.intensities, scale=self.uncertainties)

    @property
    def q(self):
        base_q = td.Normal(
            td.transform_to(td.constraints.positive)(self.loc),
            self.scale,
        )
        from pyro.distributions import FoldedDistribution
        q = FoldedDistribution(base_q)
        return q

    def minus_elbo(self, mc_samples):
        q = self.q
        z = q.rsample((mc_samples,))
        F = z[...,self.miller_id]
        loc, scale = self.NN(self.metadata).T
        scale = td.transform_to(td.constraints.positive)(scale)
        Sigma = td.Normal(loc, scale).rsample((mc_samples,))
        log_likelihood = self.likelihood.log_prob(F * F * Sigma).sum()
        log_p_z = torch.where(self.centric, self.p_centric.log_prob(z), self.p_acentric.log_prob(z))
        log_q_z = q.log_prob(z)
        kl_div = (log_q_z - log_p_z).sum()
        return -log_likelihood + kl_div

    def forward(self, mc_samples):
        return self.minus_elbo(mc_samples)

model = VariationalMergingModel(centric, multiplicity, miller_id, metadata, intensities, uncertainties, n_layers)
model = model.cuda()

#Train the model
optimizer = torch.optim.Adam(model.parameters())

for i in trange(steps):
    optimizer.zero_grad()
    loss = model(mc_samples)
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
