[![PyPI](https://img.shields.io/pypi/v/fidjax.svg)](https://pypi.python.org/pypi/fidjax/#history)

# FID JAX

Clean implementation of the [Frechet Inception Distance][paper] in JAX.

- Reproduces OpenAI's TensorFlow implementation.
- Pure JAX implementation runs on CPU/GPU/TPU and inside JIT.
- Can load weights from GCS using `pathlib` API.
- Clean and simple code.

[paper]: https://arxiv.org/pdf/1706.08500
[openai]: https://github.com/openai/guided-diffusion/tree/main/evaluations

## Instructions

1️⃣ FID JAX is a [single file][file], so you can just copy it to your project
directory. Or you can install the package:

```sh
pip install fidjax
```

[file]: https://github.com/danijar/fidjax/blob/main/fid.py

2️⃣ Download the Inception weights (credits to [Matthias Wright][jaxfid]):

```sh
wget https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1
```

3️⃣ Download the ImageNet reference stats of the desired resolution ([generate
your own](#custom-datasets) for other datasets):

```sh
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz
```

4️⃣ Compute activations, statistics, and scores in JAX:

```python
import fidjax
import numpy as np

weights = './inception_v3_weights_fid.pickle?dl=1'
reference = './VIRTUAL_imagenet128_labeled.npz'
fid = fidjax.FID(weights, reference)

fid_total = 50000
fid_batch = 1000
acts = []
for range(fid_total // fid_batch):
  samples = ...  # (B, H, W, 3) jnp.uint8
  acts.append(fid.compute_acts(samples))
stats = fid.compute_stats(acts)
score = fid.compute_score(stats)

print(float(score))  # FID
```

[jaxfid]: https://github.com/matthias-wright/jax-fid

## Accuracy

| Dataset | Model | FID JAX | OpenAI TF |
| :------ | :---- | :-----: | :-------: |
| ImageNet 256 | ADM (guided, upsampled) | 3.937 | 3.943 |

## Tutorials

### Using Cloud Storage

Point to the files via a `pathlib.Path` implementation that support your
Cloud storage. For example for GCS:

```python
import elements  # pip install elements
import fidjax

weights = elements.Path('gs://bucket/fid/inception_v3_weights_fid.pickle')
reference = elements.Path('gs://bucket/fid/VIRTUAL_imagenet128_labeled.npz')

fid = fidjax.FID(weights, reference)
```

### Custom Datasets

Generate reference statistics for custom datasets:

```python
import fidjax
import numpy as np

weights = './inception_v3_weights_fid.pickle?dl=1'
fid = fidjax.FID(weights)

acts = fid.compute_acts(images)
mu, sigma = fid.compute_stats(acts)

np.savez('reference.npz', {'mu': mu, 'sigma': sigma})
```

## Resources

- [OpenAI standard implementation in TensorFlow][openai]
- [PyTorch port](https://github.com/mseitzer/pytorch-fid)
- [JAX port][jaxfid]
- [JAX port (single-file)](https://github.com/kvfrans/jax-fid-parallel)

## Questions

Please file an [issue on Github](https://github.com/danijar/fidjax/issues).
