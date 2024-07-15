__version__ = '1.0.1'

import io
import os
import pathlib
import pickle
from functools import partial as bind
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class InceptionV3(nn.Module):

  num_classes: int = 0

  @nn.compact
  def __call__(self, x, train=True, rng=jax.random.PRNGKey(0)):
    avg_pool = bind(nn.avg_pool, count_include_pad=False)
    x = Conv2D(32, 3, 2, name='Conv2d_1a_3x3')(x, train)
    x = Conv2D(32, 3, name='Conv2d_2a_3x3')(x, train)
    x = Conv2D(64, 3, pad='same', name='Conv2d_2b_3x3')(x, train)
    x = nn.max_pool(x, (3, 3), (2, 2))
    x = Conv2D(80, 1, name='Conv2d_3b_1x1')(x, train)
    x = Conv2D(192, 3, name='Conv2d_4a_3x3')(x, train)
    x = nn.max_pool(x, (3, 3), (2, 2))
    x = InceptionA(32, name='Mixed_5b')(x, train)
    x = InceptionA(64, name='Mixed_5c')(x, train)
    x = InceptionA(64, name='Mixed_5d')(x, train)
    x = InceptionB(name='Mixed_6a')(x, train)
    x = InceptionC(128, name='Mixed_6b')(x, train)
    x = InceptionC(160, name='Mixed_6c')(x, train)
    x = InceptionC(160, name='Mixed_6d')(x, train)
    x = InceptionC(192, name='Mixed_6e')(x, train)
    x = InceptionD(name='Mixed_7a')(x, train)
    x = InceptionE(avg_pool, name='Mixed_7b')(x, train)
    x = InceptionE(nn.max_pool, name='Mixed_7c')(x, train)
    x = x.mean((1, 2), keepdims=True)
    if self.num_classes:
      x = nn.Dropout(rate=0.5)(x, deterministic=not train, rng=rng)
      x = x.reshape((x.shape[0], -1))
      x = nn.Dense(self.num_classes, name='fc')(x)
    return x


class InceptionA(nn.Module):

  pool_depth: int = 0

  @nn.compact
  def __call__(self, x, train=True):
    a = Conv2D(64, 1, name='branch1x1')(x, train)
    b = Conv2D(48, 1, name='branch5x5_1')(x, train)
    b = Conv2D(64, 5, pad='same', name='branch5x5_2')(b, train)
    c = Conv2D(64, 1, name='branch3x3dbl_1')(x, train)
    c = Conv2D(96, 3, pad='same', name='branch3x3dbl_2')(c, train)
    c = Conv2D(96, 3, pad='same', name='branch3x3dbl_3')(c, train)
    d = nn.avg_pool(x, (3, 3), padding='same', count_include_pad=False)
    d = Conv2D(self.pool_depth, 1, name='branch_pool')(d, train)
    return jnp.concatenate((a, b, c, d), axis=-1)


class InceptionB(nn.Module):

  @nn.compact
  def __call__(self, x, train=True):
    a = Conv2D(384, 3, 2, name='branch3x3')(x, train)
    b = Conv2D(64, 1, name='branch3x3dbl_1')(x, train)
    b = Conv2D(96, 3, pad='same', name='branch3x3dbl_2')(b, train)
    b = Conv2D(96, 3, 2, name='branch3x3dbl_3')(b, train)
    c = nn.max_pool(x, (3, 3), (2, 2))
    return jnp.concatenate((a, b, c), axis=-1)


class InceptionC(nn.Module):

  depth: int = 0

  @nn.compact
  def __call__(self, x, train=True):
    a = Conv2D(192, 1, name='branch1x1')(x, train)
    b = Conv2D(self.depth, 1, name='branch7x7_1')(x, train)
    b = Conv2D(self.depth, (1, 7), pad='same', name='branch7x7_2')(b, train)
    b = Conv2D(192, (7, 1), pad='same', name='branch7x7_3')(b, train)
    c = Conv2D(self.depth, 1, name='branch7x7dbl_1')(x, train)
    c = Conv2D(self.depth, (7, 1), pad='same', name='branch7x7dbl_2')(c, train)
    c = Conv2D(self.depth, (1, 7), pad='same', name='branch7x7dbl_3')(c, train)
    c = Conv2D(self.depth, (7, 1), pad='same', name='branch7x7dbl_4')(c, train)
    c = Conv2D(192, (1, 7), pad='same', name='branch7x7dbl_5')(c, train)
    d = nn.avg_pool(x, (3, 3), padding='same', count_include_pad=False)
    d = Conv2D(192, 1, name='branch_pool')(d, train)
    return jnp.concatenate((a, b, c, d), axis=-1)


class InceptionD(nn.Module):

  @nn.compact
  def __call__(self, x, train=True):
    a = Conv2D(192, 1, name='branch3x3_1')(x, train)
    a = Conv2D(320, 3, 2, name='branch3x3_2')(a, train)
    b = Conv2D(192, 1, name='branch7x7x3_1')(x, train)
    b = Conv2D(192, (1, 7), pad='same', name='branch7x7x3_2')(b, train)
    b = Conv2D(192, (7, 1), pad='same', name='branch7x7x3_3')(b, train)
    b = Conv2D(192, 3, 2, name='branch7x7x3_4')(b, train)
    c = nn.max_pool(x, (3, 3), (2, 2))
    return jnp.concatenate((a, b, c), axis=-1)


class InceptionE(nn.Module):

  pool: Callable

  @nn.compact
  def __call__(self, x, train=True):
    a = Conv2D(320, 1, name='branch1x1')(x, train)
    b = Conv2D(384, 1, name='branch3x3_1')(x, train)
    b1 = Conv2D(384, (1, 3), pad='same', name='branch3x3_2a')(b, train)
    b2 = Conv2D(384, (3, 1), pad='same', name='branch3x3_2b')(b, train)
    b = jnp.concatenate((b1, b2), axis=-1)
    c = Conv2D(448, 1, name='branch3x3dbl_1')(x, train)
    c = Conv2D(384, 3, pad='same', name='branch3x3dbl_2')(c, train)
    c1 = Conv2D(384, (1, 3), pad='same', name='branch3x3dbl_3a')(c, train)
    c2 = Conv2D(384, (3, 1), pad='same', name='branch3x3dbl_3b')(c, train)
    c = jnp.concatenate((c1, c2), axis=-1)
    d = self.pool(x, (3, 3), padding='same')
    d = Conv2D(192, 1, name='branch_pool')(d, train)
    return jnp.concatenate((a, b, c, d), axis=-1)


class Conv2D(nn.Module):

  depth: int
  kernel: Any = 3
  stride: Any = 1
  pad: Any = 'valid'

  @nn.compact
  def __call__(self, x, train=True):
    kernel = self.kernel
    if isinstance(kernel, int):
      kernel = (kernel,) * 2
    args = (self.depth, kernel, self.stride, self.pad)
    x = nn.Conv(*args, use_bias=False, name='conv')(x)
    x = nn.BatchNorm(
        use_running_average=(not train),
        epsilon=1e-3, name='bn')(x)
    x = jax.nn.relu(x)
    return x


class FID:

  def __init__(
      self, weights, reference=None, resize=299,
      check_shapes=False, weights_device=None):
    if isinstance(weights, str):
      weights = pathlib.Path(weights)
    with weights.open('rb') as f:
      loaded = pickle.loads(f.read())
    if reference is not None:
      if isinstance(reference, str):
        reference = pathlib.Path(reference)
      with reference.open('rb') as f:
        reference = np.load(io.BytesIO(f.read()))
        self.ref = (reference['mu'], reference['sigma'])
    else:
      self.ref = None
    model = InceptionV3()
    loaded = self._convert(loaded)
    if check_shapes:
      print('Checking FID weights...')
      reference = model.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
      pairs = jax.tree.map(lambda x, y: (x, y), loaded, reference)
      for path, (x1, x2) in jax.tree_util.tree_leaves_with_path(
          pairs, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2):
        name = tuple(segment.key for segment in path)
        assert x1.shape == x2.shape, (name, x1.shape, x2.shape)
    self.params = jax.device_put(loaded, weights_device)
    self.apply = bind(jax.jit(bind(model.apply, train=False)), self.params)
    self.resize = resize

  def compute_acts(self, imgs):
    assert imgs.dtype == jnp.uint8, imgs.dtype
    assert imgs.ndim == 4, imgs.shape
    imgs = imgs.astype(jnp.float32) / 255
    imgs = imgs * 2 - 1
    if self.resize:
      imgs = jax.image.resize(
          imgs, ((len(imgs)), self.resize, self.resize, 3),
          method='bilinear', antialias=False)
    acts = self.apply(imgs)
    acts = jnp.squeeze(acts, (1, 2))
    return acts

  def compute_stats(self, acts):
    if isinstance(acts, (list, tuple)):
      acts = jnp.concatenate(acts, 0)
    assert acts.shape[1:] == (2048,)
    mu = acts.mean(0)
    sigma = jnp.cov(acts, rowvar=False)
    return (mu, sigma)

  def compute_score(self, stats, ref=None):
    mu1, sigma1 = stats
    mu2, sigma2 = (self.ref if ref is None else ref)
    diff = mu1 - mu2
    offset = jnp.eye(sigma1.shape[0]) * 1e-6
    covmean = jax.scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = jnp.real(covmean)
    fid = diff @ diff + jnp.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

  def _convert(self, params):
    params = {'batch_stats': {}, 'params': params}
    params['params'].pop('fc')  # FID doesn't use the output layer.
    # Move BN stats into separate collection for Flax.
    is_layer = lambda x: isinstance(x, dict) and 'bn' in x
    for path, layer in jax.tree_util.tree_leaves_with_path(
        params['params'], is_leaf=is_layer):
      target = params['batch_stats']
      for segment in path:
        if segment.key not in target:
          target[segment.key] = {}
        target = target[segment.key]
      mean = layer['bn'].pop('mean')
      var = layer['bn'].pop('var')
      target['bn'] = {'mean': mean, 'var': var}
    return params


if __name__ == '__main__':
  import tqdm

  # Resources:
  # https://github.com/matthias-wright/jax-fid
  # https://github.com/openai/guided-diffusion/tree/main/evaluations
  urls = [
      ('https://www.dropbox.com/s/xt6zvlvt22dcwck/' +
       'inception_v3_weights_fid.pickle?dl=1'),
      ('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/' +
       'ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz'),
      ('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/' +
       'ref_batches/imagenet/256/admnet_guided_upsampled_imagenet256.npz'),
  ]
  for url in urls:
    if not pathlib.Path(url.rsplit('/', 1)[-1]).exists():
      os.system(f'wget "{url}"')

  weights = './inception_v3_weights_fid.pickle?dl=1'
  reference = './VIRTUAL_imagenet256_labeled.npz'
  fid = FID(weights, reference)

  filename = 'admnet_guided_upsampled_imagenet256.npz'
  samples = jax.device_put(np.load(filename)['arr_0'])

  # samples = samples[:3000]  # 15.77

  batch = 64
  acts = []
  for i in tqdm.trange(0, len(samples), batch):
    acts.append(fid.compute_acts(samples[i: i + batch]))
  assert sum(len(x) for x in acts) == len(samples)

  print('Computing score...')
  stats = fid.compute_stats(acts)
  score = fid.compute_score(stats)

  print(score)  # 3.9371355
