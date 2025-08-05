from model import *
from matplotlib import pyplot as plt
from IPython import embed

# Parameters
nrefls=3
multiplicity = 10
mean_snr = 10.
mlp_layers = 10
mlp_width = 10
training_steps=10_000
mc_samples=1_000

error_model = HalfNormalSNR(mean_snr)
#error_model = ExponentialSNR(mean_snr)
prior = tfd.Weibull(2., 1.)
metadata_prior = tfd.Uniform(-1., 1.)
scale_func = lambda x: 2. - np.abs(np.cos(x*np.pi))
simulator = Simulator(prior, metadata_prior, scale_func, error_model)
simulator.simulate(nrefls, multiplicity)

surrogate_posterior = VariationalStructureFactor.from_loc_and_scale(
    prior.mean() * np.ones(nrefls),
    prior.stddev() * np.ones(nrefls)/100.,
)
scale_model = ScaleModel.from_dims(mlp_layers, mlp_width)
likelihood = tfd.Normal(simulator.Iobs, simulator.SigIobs)


merging_model = MergingModel(surrogate_posterior, scale_model, prior, likelihood)
data = (simulator.Metadata, simulator.Iobs, simulator.SigIobs)

merging_model.train(data, training_steps, mc_samples)

c1 = "#1b9e77"
c2 = "#d95f02"
c3 = "#7570b3"
cmap_name = 'cividis'

colors = [c1, c2, c3]

def array_to_colors(x, cmap_name='cividis'):
    c = np.argsort(np.argsort(x))
    c = c - c.min()
    c = c / c.max()
    cmap = plt.get_cmap(cmap_name)
    return cmap(c)

if nrefls > 3:
    colors = array_to_colors(np.arange(nrefls))

from itertools import cycle
colorcycle = cycle(colors)


def plot_true_structure_factors(simulator, colorcycle):
    for i,F in enumerate(simulator.F):
        color = next(colorcycle)
        plt.plot([i-0.25, i+0.25], [F, F], '-', color=color)

    plt.xlabel("Miller Index")
    plt.ylabel("Structure Factor")

def plot_true_scale(simulator, xpoints=1_000):
    x = np.linspace(-1., 1., xpoints)
    plt.plot(x, simulator.scale_func(x), '--k')

    plt.xlabel("Metadata")
    plt.ylabel("Scale")

def plot_noisy_observations(simulator, colorcycle):
    for x,y,yerr in zip(
        simulator.Metadata.numpy(),
        simulator.Iobs.numpy(),
        simulator.SigIobs.numpy(),
        ):
        color = next(colorcycle)
        plt.errorbar(x, y, yerr=yerr, ls='none', c=color)
        plt.errorbar(x, y, yerr=yerr, ls='none', c=color)

    plt.xlabel("Metadata ($M$)")
    plt.ylabel("Intensity ($I$)")

def plot_true_intensities(simulator, colorcycle, xpoints=1_000):
    x = np.linspace(-1., 1., xpoints)
    y = simulator.scale_func(x)
    for i,(F,M,Itrue) in enumerate(zip(simulator.F, simulator.Metadata, simulator.Itrue)):
        color = next(colorcycle)
        plt.plot(x, y*F*F, '--', color=color)
        plt.plot(M, Itrue, '.', color=color)

    plt.xlabel("Metadata")
    plt.ylabel("Intensity")

def plot_inferred_structure_factors(surrogate_posterior, colorcycle, nsamples=10_000):
    samples = surrogate_posterior.sample(nsamples)
    for i,z  in enumerate(samples.numpy().T):
        color = next(colorcycle)
        violin_parts = plt.violinplot(z, [i])
        for pc in violin_parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            violin_parts['cmaxes'].set_color(color)
            violin_parts['cmins'].set_color(color)
            violin_parts['cbars'].set_color(color)

    plt.xlabel("Miller Index ($h$)")
    plt.ylabel("Structure Factor ($F$)")

def plot_inferred_scale(scale_model, nsamples=10_000, xpoints=1_000):
    x = np.linspace(-1., 1., xpoints)
    q = scale_model(x[:,None])
    z = tf.squeeze(q.sample(nsamples))
    low,mid,high = np.percentile(z, [2.5, 50., 97.5], axis=0)
    plt.fill_between(x, low, high, color='grey')
    plt.plot(x, mid, '--k')

    plt.xlabel("Metadata ($M$)")
    plt.ylabel("Scale ($\Sigma$)")

def plot_inferred_intensities(merging_model, simulator, colorcycle, nsamples=10_000, xpoints=1_000):
    z_f = merging_model.surrogate_posterior.sample(nsamples)

    x = np.linspace(-1., 1., xpoints)
    q_s = merging_model.scale_model(x[:,None])
    z_s = tf.squeeze(q_s.sample(nsamples))

    low,mid,high = np.percentile(z_f[:, None, :] * z_f[:, None, :] * z_s[:, :, None], [2.5, 50.0, 97.5], axis=0)

    for l,m,h in zip(low.T,mid.T,high.T):
        color = next(colorcycle)
        plt.fill_between(x, l, h, color=color, alpha=0.5)

    metadata = simulator.Metadata
    q_s = merging_model.scale_model(metadata[...,None])
    z_s = tf.squeeze(q_s.sample(nsamples))
    mid = np.median(z_f[..., None] * z_f[..., None] * z_s, axis=0)

    for x,median in zip(metadata, mid):
        color = next(colorcycle)
        plt.plot(x, median, '.', color=color)

    plt.xlabel("Metadata ($M$)")
    plt.ylabel("Intensity ($F^2\Sigma$)")

plt.figure()
plot_true_structure_factors(simulator, colorcycle)

plt.figure()
plot_true_scale(simulator)

plt.figure()
plot_noisy_observations(simulator, colorcycle)

plt.figure()
plot_true_intensities(simulator, colorcycle)

plt.figure()
plot_inferred_structure_factors(surrogate_posterior, colorcycle)

plt.figure()
plot_inferred_scale(scale_model)

plt.figure()
plot_inferred_intensities(merging_model, simulator, colorcycle)

plt.show()
#embed(colors='linux')
