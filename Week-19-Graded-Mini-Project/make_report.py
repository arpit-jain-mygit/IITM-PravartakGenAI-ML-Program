"""
Build the two Week 19 PDFs from the executed notebook and results/ (repo tooling - not part of the submission zip):

  Week 19_Graded Mini Project_Jain.pdf           every task answered: code snippets, figures, metrics, observations
  Week 19_Graded Mini Project_Jain_Analysis.pdf  the 1-2 page analysis (comparisons, takeaways, practical notes)

    python make_report.py

Code snippets are copied from the notebook cells, numbers from results/metrics.json and observations from
observations.json, so the PDFs always match the notebook.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import nbformat
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from report_helpers import Body, new_document, section

HERE = Path(__file__).parent
R = HERE / "results"
NAME = "Week 19_Graded Mini Project_Jain"
M = json.load(open(R / "metrics.json"))
OBS = json.load(open(HERE / "observations.json"))
NB = nbformat.read(HERE / f"{NAME}.ipynb", 4)


def cell(marker: str, start: str | None = None, end: str | None = None) -> str:
    """Source of the first code cell containing `marker`, optionally cut between two markers."""
    src = next(c.source for c in NB.cells if c.cell_type == "code" and marker in c.source)
    if start:
        src = src[src.index(start):]
    if end:
        src = src[:src.index(end)]
    return src.rstrip()


def output_text(marker: str) -> str:
    """Printed output of the code cell containing `marker` (for training logs)."""
    c = next(c for c in NB.cells if c.cell_type == "code" and marker in c.source)
    return "".join(o.get("text", "") for o in c.outputs if o.get("output_type") == "stream").strip()


def bullets_of(key):
    return [re.sub(r"^\s*[-*]\s*", "", l) for l in OBS[key].strip().splitlines() if l.strip()]


dae, vae, cg, ad = M["dae"], M["vae"], M["cgan"], M["adain"]
dev = M["device"]

d = new_document("Week 19 Graded Mini Project: Fashion-MNIST Generative Modelling",
                 f"Arpit Jain  |  Notebook: {NAME}.ipynb (all code and outputs)  |  Figures: results/")
c = Body(d)
c.text("**Objective:** build four generative components on Fashion-MNIST - a Denoising Autoencoder, a VAE, a "
       "conditional GAN and a StyleGAN-lite AdaIN demo - to compare compression, sampling, conditional control and "
       "style transfer.")

# ================================================================== Task 1
section(d, "Task 1: Setup")
section(d, "1.1 Enable GPU", 2)
c = Body(d)
c.text(f"In Colab: Runtime → Change runtime type → GPU (T4). The notebook selects cuda, then mps (Apple-silicon GPU), "
       f"then cpu. **Device used for this run: {dev}.**")
c.code(cell("torch.device("), "Device check")
section(d, "1.2 Install and Import Libraries", 2)
c = Body(d)
c.text("torch, torchvision, matplotlib and tqdm are imported; missing packages are pip-installed automatically. Seeds are "
       "fixed (42) and every figure is saved to results/.")
c.code(cell("importlib.util", end="SEED = 42"), "Install / import")
section(d, "1.3 Load Fashion-MNIST", 2)
c = Body(d)
c.code(cell("datasets.FashionMNIST"), "Datasets and DataLoaders (batch size 128)")
c.text("60,000 training and 10,000 test images, 28×28 greyscale in [0, 1], 10 clothing classes.")
section(d, "1.4 Create Visualisation Helper", 2)
c = Body(d)
c.code(cell("def show_grid", end="show_grid(x0"), "show_grid(...)")
c.image(R / "00_dataset_batch.png", 9.5, "A training batch drawn with show_grid")

# ================================================================== Task 2
section(d, "Task 2: Denoising Autoencoder")
section(d, "2.1 Build the Model", 2)
c = Body(d)
c.text("CNN encoder (28→14→7, then linear to a **32-dim z**) and a transposed-convolution decoder (7→14→28) with a "
       "**Sigmoid** output.")
c.code(cell("class Encoder", end="dae = DAE()"), "Encoder / Decoder / DAE")
section(d, "2.2 Add Noise", 2)
c = Body(d)
c.code(cell("def add_noise", end="show_grid("), "Gaussian noise σ = 0.3, clamped to [0, 1]")
section(d, "2.3 Train the Model", 2)
c = Body(d)
c.text("MSE between the denoised output and the clean image, Adam (lr 1e-3), 10 epochs.")
c.code(cell("EPOCHS_DAE = 10", start="opt = torch.optim.Adam", end="METRICS[\"dae_train_minutes\"]"), "Training loop")
c.table([["epoch", "train MSE", "test MSE"]] +
        [[h["epoch"], f"{h['train_mse']:.4f}", f"{h['test_mse']:.4f}"] for h in M["dae_history"]], [2, 4, 4], size=7.5)
c.image(R / "dae_loss.png", 8.5)
section(d, "2.4 Generate Outputs", 2)
c = Body(d)
c.image(R / "fig1_dae_clean_noisy_denoised.png", 16.5, "Figure 1 - clean vs noisy vs denoised (6×6 test images each)")
c.text(f"**Denoising quality:** PSNR rises from {dae['psnr_noisy_db']:.1f} dB (noisy) to "
       f"**{dae['psnr_denoised_db']:.1f} dB** (denoised); final test MSE {dae['test_mse']:.4f}. Hardest classes: "
       f"{', '.join(dae['worst_classes'])}; easiest: {', '.join(dae['best_classes'])}.")
c.image(R / "fig1b_dae_failure_cases.png", 11, "Worst 8 test reconstructions (clean / noisy / denoised)")
c.heading("Observations")
c.bullets(bullets_of("dae"))

# ================================================================== Task 3
section(d, "Task 3: Variational Autoencoder")
section(d, "3.1 Encoder Outputs", 2)
c = Body(d)
c.text("The same convolutional encoder outputs 2 × 16 numbers, split into **μ** and **log σ²**.")
c.code(cell("class VAE", end="vae = VAE()"), "VAE: encode → (μ, log σ²), reparameterise, decode")
section(d, "3.2 Reparameterisation", 2)
c = Body(d)
c.text("z = μ + exp(0.5 · log σ²) · ε with ε ~ N(0, I): the randomness is moved into ε, so gradients flow through μ and "
       "log σ² (see reparameterize above).")
section(d, "3.3 Training", 2)
c = Body(d)
c.text("Loss = −ELBO = BCE (reconstruction, summed over pixels) + KL(q(z|x) ‖ N(0, I)), both per image; Adam (lr 2e-3), "
       "12 epochs. **ELBO, BCE and KL are logged every epoch:**")
c.code(cell("def vae_loss", end="opt = torch.optim.Adam"), "ELBO loss")
c.table([["epoch", "−ELBO (train)", "BCE", "KL", "−ELBO (test)"]] +
        [[h["epoch"], f"{h['elbo_loss']:.2f}", f"{h['bce']:.2f}", f"{h['kl']:.2f}", f"{h['test_elbo_loss']:.2f}"]
         for h in M["vae_history"]], [2, 3, 3, 3, 3], size=7.5)
c.image(R / "vae_elbo_bce_kl.png", 14)
section(d, "3.4 Generate Samples", 2)
c = Body(d)
c.image(R / "fig2_vae_random_samples.png", 9.5, "Figure 2 - random samples from z ~ N(0, I)")
c.image(R / "fig3_vae_interpolations.png", 13.5, "Figure 3 - interpolations between test images of two classes")
c.image(R / "fig4_vae_traversals.png", 13, f"Figure 4 - traversals of the 6 highest-KL latent dims {vae['top_dims']}")
c.image(R / "fig4b_vae_latent_pca.png", 9, "Figure 4b - latent means (PCA), coloured by class")
c.image(R / "fig4c_ae_vs_vae.png", 16.5, "Figure 4c - same random codes through the AE and the VAE decoder; reconstructions")
c.heading("Traversal notes")
c.bullets(bullets_of("traversal"))
c.heading("Observations: AE vs VAE")
c.bullets(bullets_of("ae_vs_vae"))

# ================================================================== Task 4
section(d, "Task 4: Conditional GAN")
section(d, "4.1 Implement cGAN Architecture", 2)
c = Body(d)
c.text("Generator: z (100) **concatenated with a 50-dim label embedding** → 128×7×7 → transposed convs → Tanh. "
       "Discriminator: the image stacked with a **28×28 label map** → strided convs → one logit.")
c.code(cell("class Generator", end="G, D = "), "Generator / Discriminator")
section(d, "4.2 Train the cGAN", 2)
c = Body(d)
c.text("BCEWithLogits with real target 0.9 (one-sided label smoothing), Adam (2e-4, betas 0.5/0.999), 15 epochs; each "
       "step updates D on real + fake, then G with fresh labels. After every epoch labels 0-9 are drawn with a fixed z.")
c.code(cell("EPOCHS_GAN, REAL", start="for epoch in range", end="METRICS[\"cgan_train_minutes\"]"), "D and G update steps")
c.table([["epoch", "loss D", "loss G", "D(x)", "D(G(z))"]] +
        [[h["epoch"], f"{h['loss_D']:.3f}", f"{h['loss_G']:.3f}", f"{h['D(x)']:.2f}", f"{h['D(G(z))']:.2f}"]
         for h in M["cgan_history"]], [2, 3, 3, 3, 3], size=7.5)
c.image(R / "cgan_losses.png", 14)
c.image(R / "cgan_epoch_progress.png", 8, "Per-epoch class grid (row = epoch, columns = labels 0-9, fixed z)")
section(d, "4.3 Generate Samples", 2)
c = Body(d)
c.image(R / "fig5_cgan_per_class.png", 16.5, "Figure 5 - one generated sample per class 0-9")
c.image(R / "fig6_cgan_diversity.png", 12, "Figure 6 - diversity grid (rows = classes, columns = different z)")
pc = cg["per_class_conditional_accuracy"]
c.text(f"**Conditioning check:** a CNN classifier ({cg['classifier_test_acc']:.1%} accurate on real test images) labels "
       f"1,000 generated images; **{cg['conditional_accuracy']:.1%}** match the requested class (chance 10%).")
c.table([["class", "cond. accuracy", "diversity (gen)", "diversity (real)"]] +
        [[k, f"{pc[k]:.0%}", f"{cg['diversity_generated'][k]:.2f}", f"{cg['diversity_real'][k]:.2f}"] for k in pc],
        [4, 3, 3, 3], size=7.5)
c.heading("Observations: conditioning behaviour")
c.bullets(bullets_of("cgan"))

# ================================================================== Task 5
section(d, "Task 5: StyleGAN-lite (AdaIN Demo)")
section(d, "5.1 Implement StyleGAN-lite", 2)
c = Body(d)
c.text("AdaIN(c, s) = σ(s)·(c − μ(c))/σ(c) + μ(s) on the 64×7×7 feature maps of a small fully-convolutional "
       "autoencoder: content features are normalised per channel and re-scaled with the style image's channel mean/std, "
       "then decoded. 8 content and 8 style images.")
c.code(cell("def adain"), "Conv autoencoder + AdaIN")
section(d, "5.2 Generate Samples", 2)
c = Body(d)
c.image(R / "fig7_adain_content_style_mixed.png", 13.5, "Figure 7 - content | style | mixed")
c.image(R / "fig7b_adain_alpha.png", 7, "Style strength α from 0 to 1 for the first pair")
c.heading("Observations")
c.bullets(bullets_of("adain"))

section(d, "Analysis and Short Report")
c = Body(d)
c.text("The 1-2 page analysis is provided separately (" + NAME + "_Analysis.pdf) and in the notebook's final section.")
for k in ["analysis_ae_vae", "analysis_vae_cgan", "analysis_labels", "analysis_takeaways", "analysis_practical"]:
    c.heading(OBS[k + "_title"]); c.bullets(bullets_of(k))

out = HERE / f"{NAME}.docx"
d.save(out)
soffice = shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(out)], check=True, capture_output=True)
out.unlink()                                   # the PDF is the deliverable; the .docx was only an intermediate
print("saved", out.with_suffix(".pdf").name)

# ================================================================== 1-2 page analysis PDF
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12, spaceAfter=2)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, textColor=colors.HexColor("#1f3b57"), spaceAfter=2)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, textColor=colors.HexColor("#1f3b57"), spaceBefore=6, spaceAfter=2)
small = ParagraphStyle("s", parent=body, fontSize=7.8, leading=10, textColor=colors.HexColor("#555"))
P = lambda t, s=body: Paragraph(t, s)
md2html = lambda t: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
B = lambda t: Paragraph(md2html(t), ParagraphStyle("bu", parent=body, leftIndent=9), bulletText="•")


def tbl(rows, widths):
    cs = ParagraphStyle("c", parent=body, fontSize=7.8, leading=9.5, spaceAfter=0)
    hs = ParagraphStyle("h", parent=cs, textColor=colors.white, fontName="Helvetica-Bold")
    t = Table([[Paragraph(str(x), hs if i == 0 else cs) for x in r] for i, r in enumerate(rows)], colWidths=[w * cm for w in widths])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#bbb")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


fv = M["vae"]["final"]
s = [P("Fashion-MNIST Generative Modelling - Analysis", h1),
     P(f"Week 19 Graded Mini Project · Arpit Jain · trained on {dev} · details and figures: notebook + results/", small), Spacer(1, 4),
     tbl([["Model", "Objective", "Key result"],
          ["Denoising AE (z = 32)", "MSE to the clean image", f"PSNR {dae['psnr_noisy_db']:.1f} → {dae['psnr_denoised_db']:.1f} dB; test MSE {dae['test_mse']:.4f}"],
          ["VAE (z = 16)", "−ELBO = BCE + KL", f"test −ELBO {fv['test_elbo_loss']:.1f} (BCE {fv['test_bce']:.1f} + KL {fv['test_kl']:.1f}); "
                                               f"{vae['active_dims_kl_gt_0.1']}/16 dims active"],
          ["cGAN (z = 100 + label)", "adversarial BCE, real = 0.9", f"conditional accuracy {cg['conditional_accuracy']:.1%} "
                                                                    f"(classifier {cg['classifier_test_acc']:.1%} on real); D(x) {cg['final']['D(x)']:.2f}, D(G(z)) {cg['final']['D(G(z))']:.2f}"],
          ["AdaIN (StyleGAN-lite)", "feature-statistics swap", f"mixed silhouette follows content (r = {ad['silhouette_corr_with_content']:.2f}) "
                                                               f"more than style (r = {ad['silhouette_corr_with_style']:.2f})"]],
         [3.6, 3.6, 10.4])]
for k in ["analysis_ae_vae", "analysis_vae_cgan", "analysis_labels", "analysis_takeaways", "analysis_practical"]:
    s += [P(OBS[k + "_title"], h2)] + [B(t) for t in bullets_of(k)]
apdf = HERE / f"{NAME}_Analysis.pdf"
SimpleDocTemplate(str(apdf), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                  title="Fashion-MNIST Generative Modelling - Analysis", author="Arpit Jain").build(s)
pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(apdf)], capture_output=True, text=True).stdout).group(1))
print(f"saved {apdf.name} ({pages} pages)" + ("  WARNING: longer than 2 pages" if pages > 2 else ""))
