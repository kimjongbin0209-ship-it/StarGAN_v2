# StarGAN v2 — AFHQ Image Translation Experiments

English | [한국어](README_kr.md)

A PyTorch research project for training and evaluating cross-domain animal face translation on AFHQ, based on [StarGAN v2: Diverse Image Synthesis for Multiple Domains](https://arxiv.org/abs/1912.01865). The dataset contains three domains: cats (`cat`), dogs (`dog`), and wildlife (`wild`).

The project root contains custom model, training, and evaluation scripts. The `stargan-v2/` directory contains the [official PyTorch implementation](https://github.com/clovaai/stargan-v2). The two implementations use different entry points and checkpoint formats; their workflows are described separately below.

## Project Structure

```text
StarGAN_v2/
├── README.md                  # Default documentation in English
├── README_kr.md               # Korean documentation
├── model.py                   # Custom model used by train.py
├── model_fixed.py             # Alternative architecture selected automatically by test.py
├── train.py                   # Latent-guided training, sample and checkpoint saving
├── test.py                    # Checkpoint evaluation and translated image saving
├── download_data.py           # AFHQ download via the official download script
├── 1912.01865v2.pdf            # StarGAN v2 paper
├── checkpoints/
│   └── 20000_nets.ckpt         # Existing experiment checkpoint
├── samples/                   # Input and translated image grids saved during training
├── eval_results/
│   └── metrics.json            # Saved evaluation results
└── stargan-v2/
    ├── README.md              # Detailed instructions for the official implementation
    ├── LICENSE                # Official implementation license
    ├── main.py                # Official training, sampling, and evaluation entry point
    ├── download.sh
    ├── core/
    ├── metrics/
    │   └── lpips_weights.ckpt
    ├── assets/
    └── data/afhq/
        ├── train/{cat,dog,wild}/
        └── val/{cat,dog,wild}/
```

## Model Components and Training Objective

| Component | Role |
| --- | --- |
| Generator (`G`) | Generates a translated image from an input image and a style code |
| Mapping Network (`F`) | Maps a latent vector `z` and a target domain to a style code |
| Style Encoder (`E`) | Extracts domain-specific style codes from an image |
| Discriminator (`D`) | Computes real/fake scores for each domain |

`train.py` combines the following losses:

```text
L_G = L_adv + lambda_sty * L_sty
            - lambda_ds * L_ds + lambda_cyc * L_cyc
```

- **Adversarial loss** encourages outputs to resemble real images in the target domain.
- **Style reconstruction loss** encourages the style extracted from a generated image to match the target style.
- **Style diversification loss** encourages different outputs when different styles are applied to the same input.
- **Cycle consistency loss** encourages recovery of the input when a translated image is converted back using its original style.

The current root training script does not implement the official implementation's R1 regularization, EMA updates, additional reference-guided updates, or decay of the diversity loss weight. Refer to the official implementation in `stargan-v2/` for the paper's full training procedure.

## Experiment Environment

The saved experiments were run on a **MacBook Pro with an Apple M4 Pro and 24 GB RAM, using PyTorch MPS**.

| Item | Experiment environment |
| --- | --- |
| Computer | MacBook Pro |
| Chip | Apple M4 Pro |
| Memory | 24 GB RAM |
| Operating system | macOS |
| Compute backend | PyTorch MPS |
| PyTorch device | `mps` |
| Evaluated checkpoint | `checkpoints/20000_nets.ckpt` |
| Evaluation data | AFHQ `val` — cat, dog, wild |

The hardware and MPS details were provided by the experimenter. Exact macOS, Python, and PyTorch versions and training duration were not recorded. The training configuration table below lists the defaults in the current `train.py`; it does not reconstruct the complete configuration used to produce the saved checkpoint.

## Environment Setup

Run the commands below from the project root. These instructions target macOS/Linux. Dataset downloading requires `git`, `bash`, `wget`, and `unzip`. On Windows, use an environment with Bash support, such as WSL.

A virtual environment with Python 3.10 or later is a starting point. Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your operating system and GPU. The project root currently has no file pinning dependency versions, so the commands below are installation examples rather than a verified dependency combination.

```bash
cd /Users/jongbinkim/Documents/Research/PP/StarGAN_v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Basic installation example for macOS
python -m pip install torch torchvision numpy pillow
```

For CUDA, replace the `torch torchvision` installation with the appropriate CUDA command from the official guide, and also install `numpy pillow`.

Check device availability after installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

The root `train.py` and `test.py --device auto` select devices in the order **MPS → CUDA → CPU**. For evaluation, specify a device explicitly with `--device cpu`, `--device mps`, or `--device cuda`.

## Preparing AFHQ

Skip downloading if images already exist in `stargan-v2/data/afhq/train` and `val`. To download the dataset:

```bash
python download_data.py
```

The script clones the official repository if `stargan-v2/` is missing, then runs `bash download.sh afhq-dataset` from that directory. The extracted dataset must have the following structure:

```text
stargan-v2/data/afhq/
├── train/
│   ├── cat/
│   ├── dog/
│   └── wild/
└── val/
    ├── cat/
    ├── dog/
    └── wild/
```

`torchvision.datasets.ImageFolder` assigns labels by sorting directory names. With this structure, the labels are `cat=0`, `dog=1`, and `wild=2`. The code refers to wildlife descriptively, but the actual directory name is `wild`.

The currently available dataset contains:

| Domain | Training images | Validation images |
| --- | ---: | ---: |
| cat | 5,153 | 500 |
| dog | 4,739 | 500 |
| wild | 4,738 | 500 |
| Total | 14,630 | 1,500 |

Training preprocessing resizes images to `256 × 256`, applies random horizontal flipping, converts images to tensors, and normalizes them to `[-1, 1]`. Evaluation omits horizontal flipping.

## Training the Custom Implementation

```bash
python train.py
```

Edit the `CONFIG` dictionary at the top of `train.py` to change training settings. The script has no command-line argument parser.

| Setting | Default | Meaning |
| --- | --- | --- |
| `img_size` | `256` | Input image resolution |
| `batch_size` | `4` | Training batch size |
| `num_domains` | `3` | Number of domains |
| `latent_dim` / `style_dim` | `16` / `64` | Latent vector and style code dimensions |
| `total_iters` | `200000` | Total training iterations |
| `lr_G` / `lr_D` / `lr_E` | `1e-4` | Learning rates for G, D, and E |
| `lr_F` | `1e-6` | Mapping Network learning rate |
| `beta1` / `beta2` | `0.0` / `0.99` | Adam coefficients |
| `lambda_sty` / `lambda_ds` / `lambda_cyc` | `1.0` / `2.0` / `1.0` | Loss weights |
| `dataset_path` | `./stargan-v2/data/afhq/train` | Training data location |
| `sample_every` / `save_every` | `1000` / `10000` | Sample and checkpoint saving intervals |
| `sample_dir` / `checkpoint_dir` | `samples` / `checkpoints` | Output directories |

Losses are printed every 100 iterations. The following files are saved at the configured intervals:

```text
samples/<iteration>.png                # Input and translated image grid
checkpoints/<iteration>_nets.ckpt       # State dictionaries for G, F, E, and D
```

**Resuming training:** The current `resume_iter` only changes the loop's starting iteration; it does not load a checkpoint. Files saved by the current `train.py` do not contain optimizer states or EMA weights. Changing `resume_iter` alone does not resume a previous training run.

**Model selection:** `train.py` imports `model.py`. The presence of `model_fixed.py` does not change the training architecture automatically. The existing `20000_nets.ckpt` contains weights compatible with `model_fixed.py`, EMA weights, and optimizer states, which differ from the current `train.py` save format. The evaluation script detects this architecture, but the training script has no restoration procedure for that checkpoint.

## Checkpoint Evaluation and Image Generation

Evaluate the existing checkpoint:

```bash
python test.py --step 20000 --device mps
```

If no checkpoint selector is supplied, the script selects the file with the largest numeric iteration among `<number>_nets.ckpt` files in `checkpoints/`:

```bash
python test.py
```

Specify the checkpoint, dataset, and output directory and save translated images:

```bash
python test.py \
  --checkpoint checkpoints/20000_nets.ckpt \
  --data_dir stargan-v2/data/afhq/val \
  --device mps \
  --output_dir eval_results/step_20000 \
  --save_images
```

For a quick translation and reconstruction check, reduce the sample count and skip FID and LPIPS:

```bash
python test.py \
  --step 20000 \
  --device mps \
  --batch_size 4 \
  --max_samples 20 \
  --skip_fid --skip_lpips \
  --save_images \
  --output_dir eval_results/quick_check
```

The main evaluation options are listed below. Run `python test.py --help` for the complete list.

| Option | Default | Description |
| --- | --- | --- |
| `--data_dir` | `./stargan-v2/data/afhq/val`* | Evaluation dataset directory |
| `--checkpoint_dir` | `checkpoints`* | Directory searched for checkpoints |
| `--checkpoint` / `--step` | Unspecified | Select a file path or an iteration; an explicit file path takes precedence |
| `--target_domains` | `all` | Domain names or indices, such as `cat,dog`, `wild`, or `0,1` |
| `--batch_size` | `16` | Evaluation batch size |
| `--max_samples` | `1000` | Separate upper limits on generated images and real FID samples per target domain; values of `0` or less disable the limit |
| `--max_lpips_sources` | `100` | Maximum number of input images used for LPIPS per target domain |
| `--num_lpips_outputs` | `5` | Outputs generated per input for diversity comparisons; use at least `2` |
| `--save_images` | Disabled | Save generated images by target domain |
| `--save_image_limit` | `256` | Maximum number of saved images per target domain |
| `--include_same_domain` | Disabled | Also use inputs from the target domain |
| `--no_ema` | Disabled | Use regular weights instead of EMA weights |
| `--skip_fid` / `--skip_lpips` | Disabled | Skip the corresponding metric |
| `--lpips_weights` | Automatic search | Explicit path to the LPIPS weights |
| `--device` / `--seed` | `auto` / `777` | Execution device and random seed |
| `--output_dir` | `eval_results` | Directory for metrics and generated images |

\* The default for `--data_dir` is resolved in this order: `STARGAN_TEST_DATASET_PATH`, `STARGAN_DATASET_PATH`, then the built-in path. The default for `--checkpoint_dir` can be changed with `STARGAN_CHECKPOINT_DIR`. Command-line arguments override environment variables. These environment variables do not affect the `CONFIG` in `train.py`.

For evaluation, `img_size`, `num_domains`, `latent_dim`, and `style_dim` must match the settings used to create the checkpoint. Their defaults are `256`, `3`, `16`, and `64`, respectively. By default, the script uses `G_ema`, `F_ema`, and `E_ema` when available, falling back to `G`, `F`, and `E` otherwise.

Computing FID and LPIPS may download torchvision's pretrained Inception v3 and AlexNet weights on the first run. The script searches locations including `stargan-v2/metrics/lpips_weights.ckpt` for the LPIPS calibration weights.

Output files are organized as follows:

```text
<output_dir>/metrics.json          # Checkpoint and dataset paths; per-domain and overall metrics
<output_dir>/cat/000000.png        # Generated images when --save_images is enabled
<output_dir>/dog/000000.png
<output_dir>/wild/000000.png
```

Reusing the default output directory overwrites `metrics.json`. Set a different `--output_dir` for each run to retain previous results. Skipped metrics are recorded as `NaN`.

### Interpreting Evaluation Metrics

| Metric | What is compared | Interpretation |
| --- | --- | --- |
| FID | Inception feature distributions of real target-domain images and generated images | Lower values indicate closer feature distributions |
| LPIPS diversity | Pairs of images generated from the same input and target domain with different latent vectors | Higher values indicate larger perceptual differences between outputs |
| PSNR cycle | The input and its reconstruction through `input → target domain → original style` | Higher values indicate smaller reconstruction errors; measured in dB |
| SSIM cycle | The same reconstruction and input | Higher values indicate greater structural similarity |

The root evaluation script evaluates latent-guided translation. By default, inputs come from domains other than the target domain. PSNR and SSIM measure **cycle reconstruction**, rather than agreement with a ground-truth target-domain image.

Overall FID is recomputed from features pooled across domains; it is not the arithmetic mean of per-domain FIDs. The other overall metrics are arithmetic means of their per-domain values. The root script's FID preprocessing, feature extraction, and sample composition differ from the official evaluation, so evaluation conditions must be aligned before comparing these results directly with the paper.

## Experiment Results

### Quantitative Evaluation — eval_results

These are saved results from experiments on a MacBook Pro with an M4 Pro, 24 GB RAM, and MPS. The values below are taken from [`eval_results/metrics.json`](eval_results/metrics.json); no new evaluation was run while preparing this README. The **20,000-iteration checkpoint** (`checkpoints/20000_nets.ckpt`) was evaluated on AFHQ `val`, generating 1,000 images per target domain for a total of 3,000 images. The original results file does not record the complete run configuration, including the seed and batch size.

| Target domain | FID | LPIPS diversity | PSNR cycle (dB) | SSIM cycle | Generated images |
| --- | ---: | ---: | ---: | ---: | ---: |
| cat | 24.3739 | 0.4476 | 15.4969 | 0.3647 | 1,000 |
| dog | 73.5681 | 0.4569 | 16.0035 | 0.3665 | 1,000 |
| wild | 73.8179 | 0.3765 | 16.1944 | 0.4195 | 1,000 |
| Overall | 44.1102 | 0.4270 | 15.8983 | 0.3836 | 3,000 |

The cat domain has the lowest FID at **24.3739**, compared with **73.5681** for dogs and **73.8179** for wildlife. Within this evaluation, generated cat-domain features are closer to the real feature distribution. The higher dog and wildlife FIDs indicate room to improve how well the model reproduces those target-domain distributions.

LPIPS diversity is highest for dogs at **0.4569**, followed by cats at **0.4476** and wildlife at **0.3765**. Different styles produce perceptually different outputs, with relatively smaller differences for wildlife in this evaluation. Higher LPIPS alone does not establish better image quality.

Overall cycle reconstruction reaches **15.8983 dB** PSNR and **0.3836** SSIM. Wildlife has the highest reconstruction metrics among the three domains, but reconstruction remains imperfect overall. These values describe recovery of the input using its original style, rather than target-domain translation accuracy against ground truth.

### Qualitative Evaluation — samples

The `samples/` directory contains **20 PNG files saved every 1,000 iterations, from iteration 1,000 through 20,000**. Representative early, intermediate, and later samples are shown below.

Each saved grid has **4 columns and 3 rows**. The first row contains input animal images, and the two lower rows contain translated outputs that can be compared within each column. The images do not record the detailed generation settings or target-domain labels for each row. The current `train.py` saves a two-row grid by concatenating inputs and generated outputs, so the existing three-row samples differ from its current save format.

#### Iteration 1,000 — Early Training

![Iteration 1,000: input images and blurry generated outputs](samples/1000.png)

Generated outputs show rough face layouts, but the eyes, noses, and fur contours remain blurry. Grid-like patterns and color bleeding are prominent. At this stage, the model's ability to generate clear animal faces is limited.

#### Iteration 5,000 — Emerging Face Shapes and Domain Characteristics

![Iteration 5,000: translated outputs with emerging cat and wildlife features](samples/5000.png)

Cat and wildlife ears, eyes, and muzzles become clearer, and fur patterns and face contours are easier to distinguish than in the early sample. The two generated outputs within a column also show differences in fur color and patterns. Some outputs still contain blurry regions and unnatural facial proportions.

#### Iteration 10,000 — Improved Detail

![Iteration 10,000: clearer eyes and fur patterns in translated outputs](samples/10000.png)

Cat eyes, whiskers, and fur patterns, along with dog face shapes, appear more clearly. Some outputs express another animal's shape and coloring while partially retaining the input's face orientation. Eye regions and face contours can still look unnatural, and sharpness varies across outputs.

#### Iteration 20,000 — Latest Saved Sample

![Iteration 20,000: cat, dog, and wildlife translation with varied appearances](samples/20000.png)

In the first column, the black cat becomes fox-like and lion-like in the lower rows. The leopard and tiger in the third and fourth columns become cat-like, illustrating changes in appearance across domains. The dog in the second column retains an open-mouth face composition while its fur color and appearance change. Different appearances are generated from the same input, and eye, nose, and fur details are clearer than in the early sample.

The representative samples show fewer early-stage blurring and grid artifacts and a stronger ability to depict animal appearances. However, **the inputs and generation conditions differ across iterations**, so these grids alone do not establish monotonic improvement or convergence. The relatively high dog and wildlife FIDs, together with facial distortions in some samples, indicate room for further improvement at iteration 20,000.

## Using the Official Implementation

Detailed installation and usage instructions are available in [`stargan-v2/README.md`](stargan-v2/README.md). The official code requires additional libraries and FFmpeg; its original dependency versions target an older environment. If only the root scripts' minimal dependencies have been installed, the following additional packages are also needed:

```bash
python -m pip install munch tqdm scipy opencv-python scikit-image ffmpeg-python
```

Image and video sampling also requires the system `ffmpeg` executable. Run the commands below **from inside `stargan-v2/`**. The current official `Solver` selects CUDA → CPU and does not automatically select MPS.

```bash
cd stargan-v2

# Train on AFHQ
python main.py --mode train --num_domains 3 --w_hpf 0 \
  --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
  --train_img_dir data/afhq/train \
  --val_img_dir data/afhq/val

# Download the official pretrained AFHQ model, then generate reference-guided samples
bash download.sh pretrained-network-afhq
python main.py --mode sample --num_domains 3 --resume_iter 100000 --w_hpf 0 \
  --checkpoint_dir expr/checkpoints/afhq \
  --result_dir expr/results/afhq \
  --src_dir assets/representative/afhq/src \
  --ref_dir assets/representative/afhq/ref
```

Official checkpoints use keys such as `generator`, `mapping_network`, and `style_encoder`, with separate `*_nets_ema.ckpt` files. This differs from the `G`, `F`, and `E` format expected by the root `test.py`. Use the official `main.py` with official pretrained checkpoints.

## Troubleshooting

- **Dataset not found:** Run from the project root and point `train.py`'s `dataset_path` or `test.py --data_dir` to the actual `train` or `val` directory.
- **`No '*_nets.ckpt' files found`:** Prepare a training checkpoint or specify `--checkpoint` or `--checkpoint_dir`.
- **Weight key or shape mismatch:** Check the checkpoint format and model dimensions. `model.py` and `model_fixed.py` have different architectures, and official checkpoints use another format.
- **LPIPS weights not found:** Specify `--lpips_weights stargan-v2/metrics/lpips_weights.ckpt` or use `--skip_lpips`.
- **Out of memory:** Reduce `CONFIG['batch_size']` for training or `--batch_size` for evaluation. Adjust evaluation workload with `--max_samples`, `--max_lpips_sources`, and `--num_lpips_outputs`.
- **MPS operation error:** Evaluation can use `--device cpu`. Training has no device argument, so change `get_device()` to choose another device.
- **Too few FID samples:** FID requires at least two real and two generated images. Do not compute FID with `--max_samples 1`.
- **Download failure:** Check that `wget` and `unzip` are installed and that the download URL is accessible. `download_data.py` prints some download errors and then exits, so verify completion by checking the dataset directories and image files.

## References and License

- [StarGAN v2 paper](https://arxiv.org/abs/1912.01865)
- [Official PyTorch repository](https://github.com/clovaai/stargan-v2)
- [Included official README](stargan-v2/README.md)
- [Included official license](stargan-v2/LICENSE)

Refer to the official repository's CC BY-NC 4.0 notice and [`stargan-v2/LICENSE`](stargan-v2/LICENSE) for the terms governing the included official code, pretrained models, and dataset. The root's custom files currently have no separate license file.

Use the following BibTeX entry to cite the original paper in research:

```bibtex
@inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Choi, Yunjey and Uh, Youngjung and Yoo, Jaejun and Ha, Jung-Woo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
