import argparse
import itertools
import json
import math
import os
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.utils import save_image

try:
    import model_fixed as fixed_model
except ImportError:
    fixed_model = None
import model as default_model


DEFAULT_DATA_DIR = (
    os.environ.get("STARGAN_TEST_DATASET_PATH")
    or os.environ.get("STARGAN_DATASET_PATH")
    or "./stargan-v2/data/afhq/val"
)
DEFAULT_CHECKPOINT_DIR = os.environ.get("STARGAN_CHECKPOINT_DIR", "checkpoints")


def get_device(name):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_dataset(data_dir, img_size):
    transform = transforms.Compose(
        [
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return datasets.ImageFolder(data_dir, transform=transform)


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    pattern = re.compile(r"^(\d+)_nets\.ckpt$")
    candidates = []
    for path in checkpoint_dir.glob("*_nets.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No '*_nets.ckpt' files found in {checkpoint_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def resolve_checkpoint(args):
    if args.checkpoint is not None:
        path = Path(args.checkpoint)
    elif args.step is not None:
        path = Path(args.checkpoint_dir) / f"{args.step}_nets.ckpt"
    else:
        path = find_latest_checkpoint(args.checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_nets(args, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    use_ema = not args.no_ema
    key_g = "G_ema" if use_ema and "G_ema" in ckpt else "G"
    key_f = "F_ema" if use_ema and "F_ema" in ckpt else "F"
    key_e = "E_ema" if use_ema and "E_ema" in ckpt else "E"

    model_module = select_model_module(ckpt[key_g])
    netG = model_module.Generator(args.img_size, args.style_dim).to(device).eval()
    netF = model_module.MappingNetwork(args.latent_dim, args.style_dim, args.num_domains).to(device).eval()
    netE = model_module.StyleEncoder(args.img_size, args.style_dim, args.num_domains).to(device).eval()

    netG.load_state_dict(ckpt[key_g])
    netF.load_state_dict(ckpt[key_f])
    netE.load_state_dict(ckpt[key_e])

    for net in (netG, netF, netE):
        for param in net.parameters():
            param.requires_grad_(False)

    loaded_step = ckpt.get("step", None)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Using weights: G={key_g}, F={key_f}, E={key_e}")
    print(f"Using model module: {model_module.__name__}")
    if loaded_step is not None:
        print(f"Checkpoint step: {loaded_step}")
    return netG, netF, netE


def select_model_module(generator_state):
    is_fixed_checkpoint = any(
        ("conv1x1" in key) or (".norm1.fc" in key) or (".norm2.fc" in key)
        for key in generator_state.keys()
    )
    if is_fixed_checkpoint:
        return fixed_model if fixed_model is not None else default_model
    return default_model


def encode_style(netE, x, y):
    try:
        return netE(x, y)
    except TypeError:
        out = netE(x)
        idx = torch.arange(y.size(0), device=y.device)
        return out[idx, y]


def imagenet_normalize(x, size=299):
    x = (x.clamp(-1, 1) + 1) / 2
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            self.inception = models.inception_v3(weights=weights, transform_input=False)
        except AttributeError:
            self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()

    def forward(self, x):
        x = imagenet_normalize(x)
        features = self.inception(x)
        if isinstance(features, tuple):
            features = features[0]
        return features


def matrix_sqrt_psd(x, eps=1e-10):
    x = (x + x.t()) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals = torch.clamp(eigvals, min=eps)
    return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.t()


def frechet_distance(features1, features2):
    features1 = features1.double()
    features2 = features2.double()
    if features1.size(0) < 2 or features2.size(0) < 2:
        raise ValueError("FID requires at least two real and two fake samples.")

    mu1 = features1.mean(dim=0)
    mu2 = features2.mean(dim=0)
    xc1 = features1 - mu1
    xc2 = features2 - mu2
    cov1 = xc1.t().matmul(xc1) / (features1.size(0) - 1)
    cov2 = xc2.t().matmul(xc2) / (features2.size(0) - 1)

    cov1_sqrt = matrix_sqrt_psd(cov1)
    covmean = matrix_sqrt_psd(cov1_sqrt.matmul(cov2).matmul(cov1_sqrt))
    fid = (mu1 - mu2).pow(2).sum() + torch.trace(cov1 + cov2 - 2 * covmean)
    return max(fid.item(), 0.0)


def lp_normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = models.AlexNet_Weights.IMAGENET1K_V1
            self.layers = models.alexnet(weights=weights).features
        except AttributeError:
            self.layers = models.alexnet(pretrained=True).features
        self.channels = [layer.out_channels for layer in self.layers if isinstance(layer, nn.Conv2d)]

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, 1, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.alexnet = AlexNetFeatures()
        self.lpips_weights = nn.ModuleList([Conv1x1(ch) for ch in self.alexnet.channels])
        state = torch.load(weights_path, map_location="cpu")
        own_state = self.state_dict()
        for name, param in state.items():
            if name in own_state:
                own_state[name].copy_(param)
        self.register_buffer("mu", torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1))
        self.register_buffer("sigma", torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            diff = (lp_normalize(x_fmap) - lp_normalize(y_fmap)).pow(2)
            value = value + torch.mean(conv1x1(diff))
        return value


def find_lpips_weights(explicit_path=None):
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"LPIPS weights not found: {path}")
        return path

    bases = [Path.cwd(), Path(__file__).resolve().parent]
    bases.extend(Path(__file__).resolve().parents)
    candidates = []
    for base in bases:
        candidates.append(base / "metrics" / "lpips_weights.ckpt")
        candidates.append(base / "stargan-v2" / "metrics" / "lpips_weights.ckpt")
    candidates.append(Path("/Users/jongbinkim/Documents/Research/PP/StarGAN_v2/stargan-v2/metrics/lpips_weights.ckpt"))

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "LPIPS weights were not found. Pass --lpips_weights /path/to/lpips_weights.ckpt "
        "or run with --skip_lpips."
    )


def to_unit_range(x):
    return (x.clamp(-1, 1) + 1) / 2


def psnr(x, y, eps=1e-10):
    x = to_unit_range(x)
    y = to_unit_range(y)
    mse = (x - y).pow(2).flatten(1).mean(dim=1)
    return 10 * torch.log10(1.0 / (mse + eps))


def gaussian_window(channels, device, dtype, size=11, sigma=1.5):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g).view(1, 1, size, size)
    return kernel.repeat(channels, 1, 1, 1)


def ssim(x, y, window_size=11):
    x = to_unit_range(x)
    y = to_unit_range(y)
    channels = x.size(1)
    window = gaussian_window(channels, x.device, x.dtype, size=window_size)
    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return value.flatten(1).mean(dim=1)


def parse_domains(text, classes):
    if text == "all":
        return list(range(len(classes)))
    domains = []
    for token in text.split(","):
        token = token.strip()
        if token.isdigit():
            domains.append(int(token))
        else:
            if token not in classes:
                raise ValueError(f"Unknown domain '{token}'. Available domains: {classes}")
            domains.append(classes.index(token))
    return domains


@torch.no_grad()
def collect_real_features(dataset, domain_idx, extractor, args, device):
    indices = [idx for idx, target in enumerate(dataset.targets) if target == domain_idx]
    random.shuffle(indices)
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    features = []
    for x, _ in loader:
        x = x.to(device)
        features.append(extractor(x).cpu())
    if not features:
        raise ValueError(f"No real images found for domain index {domain_idx}")
    return torch.cat(features, dim=0)


@torch.no_grad()
def evaluate_domain(dataset, domain_idx, class_name, nets, extractor, lpips_metric, args, device):
    netG, netF, netE = nets
    source_indices = [
        idx for idx, target in enumerate(dataset.targets)
        if args.include_same_domain or target != domain_idx
    ]
    random.shuffle(source_indices)
    if args.max_samples > 0:
        source_indices = source_indices[: args.max_samples]

    loader = DataLoader(
        Subset(dataset, source_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    fake_features = [] if extractor is not None else None
    psnr_values = []
    ssim_values = []
    lpips_values = []
    generated = 0
    lpips_sources = 0
    save_dir = Path(args.output_dir) / class_name
    if args.save_images:
        save_dir.mkdir(parents=True, exist_ok=True)

    for x_real, y_org in loader:
        if args.max_samples > 0 and generated >= args.max_samples:
            break

        x_real = x_real.to(device)
        y_org = y_org.to(device)
        if args.max_samples > 0:
            remaining = args.max_samples - generated
            x_real = x_real[:remaining]
            y_org = y_org[:remaining]

        batch = x_real.size(0)
        y_trg = torch.full((batch,), domain_idx, dtype=torch.long, device=device)
        z_trg = torch.randn(batch, args.latent_dim, device=device)
        s_trg = netF(z_trg, y_trg)
        x_fake = netG(x_real, s_trg)

        if extractor is not None:
            fake_features.append(extractor(x_fake).cpu())

        s_org = encode_style(netE, x_real, y_org)
        x_rec = netG(x_fake, s_org)
        psnr_values.append(psnr(x_rec, x_real).cpu())
        ssim_values.append(ssim(x_rec, x_real).cpu())

        if lpips_metric is not None and lpips_sources < args.max_lpips_sources:
            lpips_batch = min(batch, args.max_lpips_sources - lpips_sources)
            x_lp = x_real[:lpips_batch]
            y_lp = y_trg[:lpips_batch]
            outputs = []
            for _ in range(args.num_lpips_outputs):
                z = torch.randn(lpips_batch, args.latent_dim, device=device)
                outputs.append(netG(x_lp, netF(z, y_lp)))
            for left, right in itertools.combinations(range(args.num_lpips_outputs), 2):
                lpips_values.append(lpips_metric(outputs[left], outputs[right]).detach().cpu())
            lpips_sources += lpips_batch

        if args.save_images:
            for j in range(batch):
                if generated + j >= args.save_image_limit:
                    break
                save_image(
                    x_fake[j].detach().cpu(),
                    save_dir / f"{generated + j:06d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )

        generated += batch

    if generated == 0:
        raise ValueError(f"No fake images generated for domain '{class_name}'")

    if fake_features is not None:
        fake_features = torch.cat(fake_features, dim=0)
    psnr_value = torch.cat(psnr_values, dim=0).mean().item()
    ssim_value = torch.cat(ssim_values, dim=0).mean().item()
    lpips_value = float("nan") if not lpips_values else torch.stack(lpips_values).mean().item()

    return {
        "domain": class_name,
        "domain_index": domain_idx,
        "fake_features": fake_features,
        "psnr_cycle": psnr_value,
        "ssim_cycle": ssim_value,
        "lpips_diversity": lpips_value,
        "num_fake": int(generated),
    }


def print_table(results, overall):
    header = f"{'domain':<14} {'FID':>10} {'LPIPS':>10} {'PSNR(cyc)':>12} {'SSIM(cyc)':>12} {'N':>8}"
    print("\n" + header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item['domain']:<14} "
            f"{item['fid']:>10.4f} "
            f"{item['lpips_diversity']:>10.4f} "
            f"{item['psnr_cycle']:>12.4f} "
            f"{item['ssim_cycle']:>12.4f} "
            f"{item['num_fake']:>8d}"
        )
    print("-" * len(header))
    print(
        f"{'overall':<14} "
        f"{overall['fid']:>10.4f} "
        f"{overall['lpips_diversity']:>10.4f} "
        f"{overall['psnr_cycle']:>12.4f} "
        f"{overall['ssim_cycle']:>12.4f} "
        f"{overall['num_fake']:>8d}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved StarGAN v2 checkpoints.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--target_domains", type=str, default="all")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_domains", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--style_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_lpips_sources", type=int, default=100)
    parser.add_argument("--num_lpips_outputs", type=int, default=5)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_image_limit", type=int, default=256)
    parser.add_argument("--include_same_domain", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--lpips_weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_path = resolve_checkpoint(args)
    dataset = build_eval_dataset(args.data_dir, args.img_size)
    if len(dataset.classes) != args.num_domains:
        raise ValueError(
            f"--num_domains={args.num_domains}, but dataset has {len(dataset.classes)} classes: "
            f"{dataset.classes}"
        )
    target_domains = parse_domains(args.target_domains, dataset.classes)

    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Classes: {dataset.classes}")
    print(f"Target domains: {[dataset.classes[i] for i in target_domains]}")

    nets = load_nets(args, checkpoint_path, device)
    extractor = None if args.skip_fid else InceptionFeatureExtractor().to(device).eval()
    lpips_metric = None
    if not args.skip_lpips:
        lpips_weights = find_lpips_weights(args.lpips_weights)
        print(f"LPIPS weights: {lpips_weights}")
        lpips_metric = LPIPS(lpips_weights).to(device).eval()

    results = []
    all_real_features = []
    all_fake_features = []

    for domain_idx in target_domains:
        class_name = dataset.classes[domain_idx]
        print(f"\nEvaluating target domain: {class_name}")
        result = evaluate_domain(dataset, domain_idx, class_name, nets, extractor, lpips_metric, args, device)

        if extractor is not None:
            real_features = collect_real_features(dataset, domain_idx, extractor, args, device)
            result["fid"] = frechet_distance(real_features, result["fake_features"])
            all_real_features.append(real_features)
            all_fake_features.append(result["fake_features"])
        else:
            result["fid"] = float("nan")

        result.pop("fake_features")
        results.append(result)

    if extractor is not None:
        overall_fid = frechet_distance(torch.cat(all_real_features, dim=0), torch.cat(all_fake_features, dim=0))
    else:
        overall_fid = float("nan")

    def mean_metric(name):
        values = [item[name] for item in results if not math.isnan(item[name])]
        return float("nan") if not values else sum(values) / len(values)

    overall = {
        "fid": overall_fid,
        "lpips_diversity": mean_metric("lpips_diversity"),
        "psnr_cycle": mean_metric("psnr_cycle"),
        "ssim_cycle": mean_metric("ssim_cycle"),
        "num_fake": sum(item["num_fake"] for item in results),
    }

    print_table(results, overall)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "data_dir": args.data_dir,
                "target_domains": [dataset.classes[i] for i in target_domains],
                "per_domain": results,
                "overall": overall,
            },
            f,
            indent=2,
        )
    print(f"\nSaved metrics to {metrics_path}")
    print("PSNR/SSIM are computed on cycle reconstruction: x -> target -> original.")


if __name__ == "__main__":
    main()
