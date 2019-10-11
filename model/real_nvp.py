import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd


class ShiftAndLogScale(tf.Module):
    def __init__(self, output_units):
        super().__init__()

        self.output_units = output_units
        self.mlp = K.Sequential([
            K.layers.Dense(512, activation='relu'),
            K.layers.Dense(512, activation='relu'),
            K.layers.Dense(output_units * 2),
        ])

    @tf.function
    def __call__(self, x, output_units):
        assert output_units == self.output_units

        x = self.mlp(x)
        x = tf.split(x, 2, axis=-1)
        return x


class RealNVP(tfb.Chain):
    def __init__(self, n_layers, n_masked, n_units):
        def make_layer(i):
            fn = ShiftAndLogScale(n_units - n_masked)
            chain = [
                tfb.RealNVP(
                    num_masked=n_masked,
                    shift_and_log_scale_fn=fn,
                ),
                tfb.BatchNormalization(),
            ]
            if i % 2 == 0:
                perm = lambda: tfb.Permute(permutation=[1, 0])
                chain = [perm(), *chain, perm()]
            return tfb.Chain(chain)

        super().__init__([make_layer(i) for i in range(n_layers)])
