import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import Generator, MappingNetwork, StyleEncoder, Discriminator

# --- Configuration ---
CONFIG = {
    'img_size': 256,
    'batch_size': 4,
    'num_domains': 3,   # AFHQ: Cat, Dog, Wildlife
    'latent_dim': 16,
    'style_dim': 64,
    'lambda_sty': 1.0,
    'lambda_ds': 2.0,
    'lambda_cyc': 1.0,
    'lr_G': 1e-4,
    'lr_D': 1e-4,
    'lr_E': 1e-4,
    'lr_F': 1e-6,
    'beta1': 0.0,
    'beta2': 0.99,
    'total_iters': 200000,
    'resume_iter': 0,
    'sample_every': 1000,
    'save_every': 10000,
    'dataset_path': './stargan-v2/data/afhq/train',
    'sample_dir': 'samples',
    'checkpoint_dir': 'checkpoints'
}

def build_model(device):
    generator = Generator(CONFIG['img_size'], CONFIG['style_dim']).to(device)
    mapping_network = MappingNetwork(CONFIG['latent_dim'], CONFIG['style_dim'], CONFIG['num_domains']).to(device)
    style_encoder = StyleEncoder(CONFIG['img_size'], CONFIG['style_dim'], CONFIG['num_domains']).to(device)
    discriminator = Discriminator(CONFIG['img_size'], CONFIG['num_domains']).to(device)
    
    return generator, mapping_network, style_encoder, discriminator

def build_optims(G, F, E, D):
    opt_G = torch.optim.Adam(G.parameters(), lr=CONFIG['lr_G'], betas=(CONFIG['beta1'], CONFIG['beta2']))
    opt_F = torch.optim.Adam(F.parameters(), lr=CONFIG['lr_F'], betas=(CONFIG['beta1'], CONFIG['beta2']))
    opt_E = torch.optim.Adam(E.parameters(), lr=CONFIG['lr_E'], betas=(CONFIG['beta1'], CONFIG['beta2']))
    opt_D = torch.optim.Adam(D.parameters(), lr=CONFIG['lr_D'], betas=(CONFIG['beta1'], CONFIG['beta2']))

    return opt_G, opt_F, opt_E, opt_D

def compute_d_loss(netD, x_real, y_org, y_trg, z_trg, x_fake):
    # Real images
    out = netD(x_real)
    out = out[torch.arange(x_real.size(0)), y_org]
    loss_real = torch.mean(F.softplus(-out))

    # Fake images
    out = netD(x_fake)
    out = out[torch.arange(x_fake.size(0)), y_trg]
    loss_fake = torch.mean(F.softplus(out))

    return loss_real + loss_fake

def compute_g_loss(netD, netG, netF, netE, x_real, y_org, y_trg, z_trgs, x_fakes):
    # 1. Adversarial loss
    out = netD(x_fakes)
    out = out[torch.arange(x_fakes.size(0)), y_trg]
    loss_adv = torch.mean(F.softplus(-out))

    # 2. Style Reconstruction
    s_pred = netE(x_fakes)
    s_pred = s_pred[torch.arange(x_fakes.size(0)), y_trg]
    s_trg = netF(z_trgs, y_trg)     # Re-generate target style
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # 3. Style Diversification
    z_trg2 = torch.randn_like(z_trgs)
    s_trg2 = netF(z_trg2, y_trg)
    x_fake2 = netG(x_real, s_trg2)
    loss_ds = torch.mean(torch.abs(x_fakes - x_fake2))

    # 4. Cycle Consistency Loss
    s_org = netE(x_real)
    s_org = s_org[torch.arange(x_real.size(0)), y_org]
    x_rec = netG(x_fakes, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss_total = loss_adv + CONFIG['lambda_sty']*loss_sty - CONFIG['lambda_ds']*loss_ds + CONFIG['lambda_cyc']*loss_cyc

    return loss_total

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def main():
    device = torch.device(get_device())
    print(f"Device: {device}")

    os.makedirs(CONFIG['sample_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    # Prep. Dataset
    transform = transforms.Compose([
        transforms.Resize([CONFIG['img_size'], CONFIG['img_size']]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Check if AFHQ dataset path is correct
    dataset = datasets.ImageFolder(CONFIG['dataset_path'], transform)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(dataloader)

    # Model & Optimizer Initialization
    netG, netF, netE, netD = build_model(device)
    opt_G, opt_F, opt_E, opt_D = build_optims(netG, netF, netE, netD)

    print("Start Training...")
    for i in range(CONFIG['resume_iter'], CONFIG['total_iters']):
        # Data Load
        try:
            x_real, y_org = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_real, y_org = next(data_iter)

        x_real = x_real.to(device)
        y_org = y_org.to(device)

        # Target domain random sampling
        y_trg = torch.randint(0, CONFIG['num_domains'], (x_real.size(0),)).to(device)
        z_trg = torch.randn(x_real.size(0), CONFIG['latent_dim']).to(device)

        # --- Discriminator Update ---
        with torch.no_grad():
            s_trg = netF(z_trg, y_trg)
            x_fake = netG(x_real, s_trg)

        loss_d = compute_d_loss(netD, x_real, y_org, y_trg, z_trg, x_fake)

        opt_D.zero_grad()
        loss_d.backward()
        opt_D.step()

        # --- Generator Update ---
        # Fake D, transfer style, get diversity, and preserve original
        s_trg = netF(z_trg, y_trg)
        x_fake = netG(x_real, s_trg)

        loss_g = compute_g_loss(netD, netG, netF, netE, x_real, y_org, y_trg, z_trg, x_fake)

        opt_G.zero_grad()
        opt_F.zero_grad()
        opt_E.zero_grad()

        loss_g.backward()

        opt_G.step()
        opt_F.step()
        opt_E.step()

        # --- Logging ---
        if (i+1) % 100 == 0:
            print(f"Iter [{i+1}/{CONFIG['total_iters']//1000}k] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

        # --- Sample Saving ---
        if (i+1) % CONFIG['sample_every'] == 0:
            with torch.no_grad():
                x_concat = torch.cat([x_real, x_fake], dim=0)
                save_image(x_concat, os.path.join(CONFIG['sample_dir'], f'{i+1}.png'), nrow=CONFIG['batch_size'], normalize=True)

        # --- Checkpoint Saving ---
        if (i+1) % CONFIG['save_every'] == 0:
            ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f'{i+1}_nets.ckpt')
            torch.save({
                'G': netG.state_dict(),
                'F': netF.state_dict(),
                'E': netE.state_dict(),
                'D': netD.state_dict()
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
