import numpy as np
from scipy.stats.distributions import weibull_min

import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import util as tfu


class VariationalStructureFactor(tfd.TruncatedNormal):
    @classmethod
    def from_loc_and_scale(cls, loc, scale, zero=1e-32, infinity=1e10):
        """
        Instantiate a learnable truncated normal distribution over structure factors.
        """
        loc = tfu.TransformedVariable(
            loc,
            tfb.Softplus()
        )
        scale = tfu.TransformedVariable(
            scale,
            tfb.Softplus()
        )
        return cls(loc, scale, zero, infinity)


class ExponentialSNR:
    """
    An error model where the signal to noise ratio is
    exponentially distributed with mean given by 
    the parameter mean_snr.
    """
    def __init__(self, mean_snr):
        self.mean_snr = mean_snr

    def __call__(self, intensity):
        scale = tf.math.reciprocal(self.mean_snr)
        isigi = tfd.Exponential(scale * np.ones_like(intensity)).sample()
        sigi = intensity / isigi
        return sigi
 

class HalfNormalSNR:
    """
    An error model where the signal to noise ratio is
    halfnormally distributed with mean given by 
    the parameter mean_snr.
    """
    def __init__(self, mean_snr):
        self.mean_snr = mean_snr

    def __call__(self, intensity):
        scale = np.sqrt(np.pi / 2) * self.mean_snr
        isigi = tfd.HalfNormal(scale * np.ones_like(intensity)).sample()
        sigi = intensity / isigi
        return sigi
        
class Simulator:
    F = None
    Metadata = None
    Sigma = None
    Itrue = None
    Iobs = None
    SigIobs = None
    def __init__(self, prior, metadata_prior, scale_func, error_model):
        """
        Parameters
        ----------
        prior : tfd.Distribution
            A prior distribution on structure factors
        metadata_prior : tfd.Distribution
            A prior distribution on metadata
        scale_func : callable
            A function that accepts metadata and returns scale values
        error_model: callable
            A function that accepts intensities and returns sigmas.
        """
        self.prior = prior
        self.metadata_prior = metadata_prior
        self.scale_func = scale_func
        self.error_model = error_model

    def simulate(self, nrefls, multiplicity):
        self.F = tf.sort(self.prior.sample(nrefls))
        self.Metadata = self.metadata_prior.sample((nrefls, multiplicity))
        self.Sigma = self.scale_func(self.Metadata)
        self.Itrue = (self.F * self.F)[:,None] * self.Sigma
        self.SigIobs = self.error_model(self.Itrue)
        self.Iobs = tfd.Normal(self.Itrue, self.SigIobs).sample()
        

class ScaleModel(tfk.models.Sequential):
    @classmethod
    def from_dims(cls, mlp_layers, mlp_width):
        leakiness = 0.3
        layers = [
            tfk.layers.Dense(mlp_width),
        ]
        for i in range(mlp_layers):
            layers.append(tfk.layers.Dense(
                mlp_width, 
                kernel_initializer='identity', 
                use_bias=True, 
                activation=tfk.layers.LeakyReLU(leakiness)
            ))
        layers.append(tfk.layers.Dense(
            2, 
            kernel_initializer='identity', 
            use_bias=True, 
            activation='linear'
        ))
        layers.append(tfk.layers.Dense(tfp.layers.IndependentNormal.params_size(1)))
        layers.append(tfp.layers.IndependentNormal(1))
        return cls(layers)

class MergingModel(tfk.models.Model):
    def __init__(self, surrogate_posterior, scale_model, prior, likelihood=None):
        super().__init__()
        self.surrogate_posterior = surrogate_posterior
        self.scale_model = scale_model
        self.prior = prior
        self.likelihood = likelihood

    def nll_kl_samples(self, inputs, mc_samples=1_000):
        """
        inputs : (Metadata, Iobs, SigIobs)
            A tuple of arrays / tensors.
        """
        Metadata, Iobs, SigIobs = inputs

        z_f = self.surrogate_posterior.sample(mc_samples)
        kl_div = tf.reduce_sum(self.surrogate_posterior.log_prob(z_f) - self.prior.log_prob(z_f))
        q_Sigma = self.scale_model(Metadata[...,None])
        z_Sigma = q_Sigma.sample(mc_samples)
        Ipred = (z_f * z_f)[...,None] * tf.squeeze(z_Sigma, axis=-1)
        if self.likelihood is None:
            ll = -tf.reduce_sum(
                tfd.Normal(Iobs, SigIobs).log_prob(Ipred)
            )
        else:
            ll = -tf.reduce_sum(
                self.likelihood.log_prob(Ipred)
            )
        return ll, kl_div, Ipred

    def loss_function(self, inputs, mc_samples=1_000):
        ll, kl_div, _ = self.nll_kl_samples(inputs, mc_samples)
        return ll + kl_div

    def call(self, inputs, mc_samples=1_000):
        """
        inputs : (Metadata, Iobs, SigIobs)
            A tuple of arrays / tensors.
        """
        ll, kl_div, Ipred = self.nll_kl_samples(inputs, mc_samples)
        self.add_loss(kl_div)
        self.add_metric(kl_div, "KL")
        self.add_loss(ll)
        self.add_metric(ll, "NLL")
        return Ipred

    @tf.function
    def train_step(self, inputs, steps, mc_samples, optimizer, variables=None):
        with tf.GradientTape() as tape:
            loss = self.loss_function(inputs, mc_samples)

        if variables is None:
            variables = self.trainable_variables

        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    def train(self, data, steps, mc_samples=1_000, optimizer=None):
        if optimizer is None:
            optimizer = tfk.optimizers.Adam() 

        from tqdm import trange
        for i in (pbar := trange(steps)):
            loss = self.train_step(data, steps, mc_samples, optimizer)
            loss = float(loss)
            pbar.set_description(f"ELBO: {-loss:0.2e}")


if __name__=="__main__":

    output_directory = "toy_model_data"
    disable_gpus = False

# Number of reflections to sample
    nrefls=3
# Number of observations per reflection
    multiplicity=10

    mean_snr = 10.
    mlp_layers = 20
    mlp_width = 10
    metadata_range = (-1., 1.)
    xpoints = 1_000
    training_steps = 10_000
    mc_samples=1_000
    nsamples=10_000 #For posterior samples
    lr = 1e-3
    sigma_prior = None #tfd.Normal(0, 1.)

    import os
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    def scale_func(metadata):
        """ takes metadata and returns Sigma """
        x = metadata * np.pi
        return 1.36 - np.abs(np.cos(x))

    def error_model(intensity, mean_snr):
        from scipy.stats.distributions import halfnorm,expon
        scale = np.sqrt(np.pi / 2.) * mean_snr
        isigi = halfnorm.rvs(scale=scale * np.ones_like(intensity))
        sigi = intensity / isigi
        return sigi


    metadata = np.random.uniform(*metadata_range, size=(nrefls, multiplicity))
    F = weibull_min.rvs(2., size=nrefls) #Draw from acentric wilson prior
    F = np.sort(F)
    Sigma = scale_func(metadata)

    miller_id = np.repeat(np.arange(nrefls), multiplicity)
    Itrue = scale_func(metadata.ravel()) * np.square(F[miller_id])
    SigIobs = error_model(Itrue, mean_snr)
    Iobs = np.random.normal(Itrue, SigIobs)
    print(f"Signal to noise quintiles: {np.percentile((Iobs / SigIobs), [0.0, 25.0, 50.0, 75.0, 100.0])}")
    print(f"Mean: {np.mean(Iobs/SigIobs)}")



    def disable_gpus():
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    if disable_gpus:
        disable_gpus()

    metadata = metadata.flatten()[:,None]
    p = tfd.Weibull(2.*np.ones(nrefls, dtype='float32'), np.ones(nrefls, dtype='float32'))

    likelihood = tfd.Normal(Iobs.astype('float32'), SigIobs.astype('float32'))

    loc_init = p.mean()
    scale_init = p.stddev() / 100.
    zero,infinity = 1e-32,1e10
    q = tfd.TruncatedNormal(
        loc   = tfu.TransformedVariable(  loc_init, tfb.Softplus()), 
        scale = tfu.TransformedVariable(scale_init, tfb.Softplus()), 
        low = zero,
        high = infinity,
    )


#Construct scale function
    n,d = metadata.shape
    leakiness = 0.3
    NN = tfk.models.Sequential()
    NN.add(tfk.Input(d))
    NN.add(tfk.layers.Dense(mlp_width))
    for i in range(mlp_layers):
        NN.add(tfk.layers.Dense(
            mlp_width, 
            kernel_initializer='identity', 
            use_bias=True, 
            activation=tfk.layers.LeakyReLU(leakiness)
        ))
    NN.add(tfk.layers.Dense(
        2, 
        kernel_initializer='identity', 
        use_bias=True, 
        activation='linear'
    ))

#Evaluate the elbo
    @tf.function
    def minus_elbo():
        z = q.sample(mc_samples)
        F = tf.gather(z, miller_id, axis=1)
        mu, sig = tf.unstack(NN(metadata), axis=-1)

        #mu = tf.math.softplus(mu)
        sig = tf.math.softplus(sig)
        q_sigma = tfd.Normal(mu, sig)
        z_Sigma = q_sigma.sample(mc_samples)

        log_likelihood = tf.reduce_sum(likelihood.log_prob(F * F * z_Sigma)) 
        log_p_z = p.log_prob(z)
        log_q_z = q.log_prob(z)
        kl_div = tf.reduce_sum(log_q_z - log_p_z)
        if sigma_prior is not None:
            sigma_kl = tf.reduce_sum(q_sigma.log_prob(z_Sigma) - sigma_prior.log_prob(z_Sigma))
            kl_div += sigma_kl

        return -log_likelihood + kl_div

    optimizer = tfk.optimizers.Adam(lr)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = minus_elbo()
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return float(loss)

    from tqdm import trange
    losses = []
    variables = list(q.trainable_variables) + NN.trainable_variables
    for i in (pbar := trange(training_steps)):
        try:
            loss = train_step()
            losses.append(float(loss))
            pbar.set_description(f"ELBO: {float(loss):0.2e}")
        except KeyboardInterrupt:
            break


#Reshape outputs to group by refl id
    metadata = metadata.reshape((nrefls, multiplicity))
    Itrue = Itrue.reshape((nrefls, multiplicity))
    Iobs =  Iobs.reshape((nrefls, multiplicity))
    SigIobs =  SigIobs.reshape((nrefls, multiplicity))


