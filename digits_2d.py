#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

tfb = tfp.bijectors
tfd = tfp.distributions
#%%
mnist = (tfds.load(
    name="mnist",
    split=tfds.Split.TRAIN,
).map(lambda x: {
    'image': x['image'] / 255,
    'label': x['label']
}))
img = next(iter(mnist))['image'].numpy().squeeze()

plt.imshow(img)


#%%
class Digits2D:
    def __init__(self, img, length=100, buff_size=100):
        self.img = img
        self.counter = 0
        self.length = length
        self.buff_size = buff_size
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = np.random.rand(self.buff_size, 2)

    def generate(self):
        h, w = img.shape
        while True:
            for x, y in self.buffer:

                rx = (w * x - 0.5).round().astype(int)
                ry = (h * (1 - y) - 0.5).round().astype(int)

                if img[ry, rx] > 0.3:
                    self.counter += 1
                    yield (x, y)

                if self.counter >= self.length:
                    return

            self.reset_buffer()


def sample_plot(samples):
    plt.figure(figsize=(4, 4))
    plt.scatter(*(samples.T), s=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


sample_plot(np.array(list(Digits2D(img, length=1000).generate())))


#%%
class ShiftAndLogScale(tf.Module):
    def __init__(self, output_units):
        super().__init__()

        self.output_units = output_units
        self.mlp = K.Sequential([
            K.layers.Dense(512, activation='relu'),
            K.layers.Dense(512, activation='relu'),
            K.layers.Dense(output_units * 2),
        ])

    def __call__(self, x, output_units):
        assert output_units == self.output_units

        x = self.mlp(x)
        x = tf.split(x, 2, axis=-1)
        return x


#%%

bijectors = []
n_layers = 10
for i in range(n_layers):
    shift_and_log_scale_fn = tfb.real_nvp_default_template(
        hidden_layers=[32, 32],
        activation=tf.nn.relu,
    )
    bijectors.append(
        tfb.RealNVP(
            num_masked=1,
            shift_and_log_scale_fn=ShiftAndLogScale(1),
        ))
    bijectors.append(tfp.bijectors.Permute(permutation=[1, 0]))

dist_mnd = tfd.MultivariateNormalDiag(loc=[0, 0], scale_diag=[1, 1])
nvp = tfb.Chain(bijectors)

flow = tfd.TransformedDistribution(
    distribution=dist_mnd,
    bijector=nvp,
)

#%%

dataset = tf.data.Dataset.from_generator(
    Digits2D(img, length=320000).generate,
    output_types=(float),
    output_shapes=(2, ),
)

optimizer = tf.optimizers.Adam(learning_rate=0.0001)
avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)

for batch in dataset.batch(32):
    with tf.GradientTape() as tape:
        log_prob = flow.log_prob(batch)
        loss = -tf.reduce_mean(log_prob)
    grads = tape.gradient(loss, flow.trainable_variables)
    optimizer.apply_gradients(zip(grads, flow.trainable_variables))
    avg_loss.update_state(loss)

    if tf.equal(optimizer.iterations % 10, 0):
        print(
            f'Step {optimizer.iterations.numpy()}',
            f'Loss {avg_loss.result():.6f}',
        )
        avg_loss.reset_states()

    if optimizer.iterations % 100 == 0:
        samples = flow.sample(1000).numpy()
        sample_plot(samples)
