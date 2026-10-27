# Week 19 Graded Mini Project: Fashion-MNIST Generative Modelling

This project trains four generative components on Fashion-MNIST:
- **Denoising Autoencoder (DAE):** compresses images to a 32-dim code and reconstructs them from noisy inputs.
- **Variational Autoencoder (VAE):** learns a probabilistic latent space that can be sampled, interpolated and traversed.
- **Conditional GAN (cGAN):** generates images of a requested clothing class.
- **StyleGAN-lite (AdaIN):** mixes the shape of one image with the style of another.

## Files

| File | What it is |
|---|---|
| `Week 19_Graded Mini Project_Jain.ipynb` | The notebook: Tasks 1-5 with all outputs, observations and the analysis |
| `results/` | Figures 1-7 as PNGs, plus training curves, extra figures and `metrics.json` |
| `Week 19_Graded Mini Project_Jain.pdf` | Report: every task answered with code snippets, figures, metrics and observations |
| `Week 19_Graded Mini Project_Jain_Analysis.pdf` | 1-2 page analysis: AE vs VAE, VAE vs cGAN, why labels help, takeaways, practical notes |

Figures in `results/`:

| Figure | File |
|---|---|
| 1 | `fig1_dae_clean_noisy_denoised.png` (and `fig1b_dae_failure_cases.png`) |
| 2 | `fig2_vae_random_samples.png` |
| 3 | `fig3_vae_interpolations.png` |
| 4 | `fig4_vae_traversals.png` (and `fig4b_vae_latent_pca.png`, `fig4c_ae_vs_vae.png`) |
| 5 | `fig5_cgan_per_class.png` |
| 6 | `fig6_cgan_diversity.png` |
| 7 | `fig7_adain_content_style_mixed.png` (and `fig7b_adain_alpha.png`) |

## How to run

**Google Colab (recommended, GPU):**
1. Upload the notebook.
2. Choose **Runtime → Change runtime type → GPU (T4)**.
3. Choose **Runtime → Run all**.

Fashion-MNIST downloads automatically. Training takes about 10 minutes on a T4, and the first cell prints `device: cuda`.

**Locally:**

```bash
pip install torch torchvision matplotlib tqdm scikit-learn jupyter
jupyter nbconvert --to notebook --execute --inplace "Week 19_Graded Mini Project_Jain.ipynb"
```

The notebook picks the best device available: `cuda`, then Apple-silicon `mps`, then `cpu`. Seeds are fixed at 42. GPU kernels are not bit-for-bit deterministic, so numbers can differ slightly between runs and devices.
