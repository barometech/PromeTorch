"""
PyTorch baseline VAE on MNIST — mirrors examples/vae/train_vae.cpp.
Architecture: 784 -> 400 (ReLU) -> (mu:20, logvar:20); 20 -> 400 (ReLU) -> 784 (Sigmoid).
Loss: BCE sum per batch / batch_size + KL divergence mean.

Run: python bench_pytorch_vae.py --epochs 50 --batch_size 128 --lr 1e-3 --data ../data/mnist
"""
import argparse
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        buf = f.read(num * rows * cols)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows * cols)
    return arr.astype(np.float32) / 255.0


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden=400, latent=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.fc3 = nn.Linear(latent, hidden)
        self.fc4 = nn.Linear(hidden, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, batch_size):
    # recon_loss: BCE summed / B (matches PromeTorch train_vae.cpp)
    bce_sum = F.binary_cross_entropy(recon, x, reduction="sum")
    recon_loss = bce_sum / batch_size
    # KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / B
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + kl, recon_loss.item(), kl.item()


def evaluate(model, loader, device):
    model.eval()
    total_elbo = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_batches = 0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            recon, mu, logvar = model(xb)
            loss, r, k = vae_loss(recon, xb, mu, logvar, xb.size(0))
            total_elbo += loss.item()
            total_recon += r
            total_kl += k
            total_batches += 1
    return total_elbo / total_batches, total_recon / total_batches, total_kl / total_batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "mnist"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Torch threads: {torch.get_num_threads()}")

    train = load_mnist_images(os.path.join(args.data, "train-images-idx3-ubyte"))
    test = load_mnist_images(os.path.join(args.data, "t10k-images-idx3-ubyte"))
    print(f"train={train.shape[0]} test={test.shape[0]}")

    train_t = torch.from_numpy(train)
    test_t = torch.from_numpy(test)

    train_ds = TensorDataset(train_t)
    test_ds = TensorDataset(test_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_times = []
    total_t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        batches = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar, xb.size(0))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1
        dt = time.time() - t0
        epoch_times.append(dt)
        train_avg = total_loss / batches
        test_elbo, test_recon, test_kl = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:2d}  train_loss={train_avg:.3f}  "
              f"test_loss={test_elbo:.3f}  (recon={test_recon:.2f} kl={test_kl:.2f})  ({dt:.1f} s)",
              flush=True)

    total = time.time() - total_t0
    final_elbo, final_recon, final_kl = evaluate(model, test_loader, device)
    print()
    print("=" * 60)
    print(f"FINAL PyTorch test ELBO = {final_elbo:.3f}  (recon={final_recon:.2f}  kl={final_kl:.2f})")
    print(f"Wall-clock per epoch (avg): {sum(epoch_times)/len(epoch_times):.2f} s")
    print(f"Total wall-clock: {total:.1f} s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
