# Generative Models on MNIST and CIFAR-10

Course assignment 1 (811, AI in Multimedia), by Rachita Dewanjee. PyTorch.

Four problems, from a simple classifier up to latent image generators:
a CNN classifier, a binary-latent autoencoder, autoregressive models over latent tokens,
and two latent generative models on CIFAR-10 (masked discrete diffusion and a Diffusion Transformer).
All targets were met.

## Results

| Problem | Dataset | Model | Result | Target |
|---|---|---|---|---|
| 1 Classification | MNIST | CNN | test accuracy **99.05%** | > 90% |
| 2 Binary autoencoder | MNIST | conv AE, binary latent 3×4×4 (48 bits) | PSNR **29.96 dB** | > 25 dB |
| 3.1 Sequence → token | MNIST tokens | GRU | **95.4%** training sequences | > 80% |
| 3.2 Sequence → sequence | MNIST tokens | causal Transformer | **94.8%** training sequences | > 80% |
| 3.3 Sequence → histogram | MNIST tokens | GRU | **100.0%** training sequences | > 99% |
| 4.1 Continuous AE | CIFAR-10 | conv AE, latent 5×8×8 | PSNR **29.19 dB** | > 25 dB |
| 4.1 Discrete AE | CIFAR-10 | VQ-VAE, latent 10×8×8 indices | PSNR **32.47 dB** | > 25 dB |
| 4.2 Discrete diffusion | CIFAR-10 | masked diffusion (Transformer) on VQ-VAE tokens | FID **33.6** | < 40 |
| 4.3 Continuous Transformer | CIFAR-10 | Diffusion Transformer (DiT) on AE latents | FID **22.2** | < 40 |

## Notebooks

| Notebook | What it does |
|---|---|
| [`P1.ipynb`](notebooks/P1.ipynb) | CNN digit classifier with data augmentation |
| [`P2.ipynb`](notebooks/P2.ipynb) | Autoencoder that compresses each digit to a 48-bit binary latent |
| [`P3.ipynb`](notebooks/P3.ipynb) | Turns the binary latents into 16-token sequences; GRU next-token, causal Transformer, and GRU histogram models |
| [`P4_a.ipynb`](notebooks/P4_a.ipynb) | CIFAR-10 continuous autoencoder and VQ-VAE (10 codebooks, EMA updates, dead-code restart) |
| [`P4_b_discretediffusion.ipynb`](notebooks/P4_b_discretediffusion.ipynb) | Masked discrete diffusion over VQ-VAE tokens with a class-conditional Transformer |
| [`P4_b_eval.ipynb`](notebooks/P4_b_eval.ipynb) | Sampler tuning and FID evaluation of the trained masked-diffusion model |
| [`P4_c_transformer.ipynb`](notebooks/P4_c_transformer.ipynb) | Diffusion Transformer (DiT) on continuous latents, cosine schedule, DDIM with classifier-free guidance |
| [`p4-c-eval-v2.ipynb`](notebooks/p4-c-eval-v2.ipynb) | DiT evaluation: sampler settings and a perceptually sharpened decoder to bring FID below 40 |

## Running
The notebooks were run on Google Colab and Kaggle GPUs. Paths such as `/content/drive/...` and
`/kaggle/...` point to those environments; change them to run elsewhere. The Problem 3 and
Problem 4.2/4.3 notebooks load the models trained in the earlier notebooks.

Requirements: Python 3, PyTorch, torchvision, NumPy, Matplotlib.
