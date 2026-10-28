
import math
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
import re
import shutil
try:
    from IPython import get_ipython as _get_ipython
except ImportError:
    _get_ipython = None

STEPS = 400000
EVAL_EVERY = 2500
N_PDE = 24576
N_BC = 2024
W_CUR = 1.0
W_DIV = 1.0
W_BC = 10.0
W_TOP = 1.0
# Interior sampling density ratio (bottom : top)
Z_RATIO = 10.0
# Learning-rate parameters
LR_MAX = 1e-3
LR_MIN = 1e-4
LR_T_HALF = 100000
LR_T_WIDTH =  90000
# Equivalent midpoint and slope to a 10000-step cosine annealing schedule
# LR_T_HALF = 10000 / 2
# LR_T_WIDTH = 10000 / pi
WIDTH = 128
DEPTH = 5
SEED = 1234
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_EVERY = 50_000
MODEL_ROOT = Path("/gwork0/yokoyakd/PINN_cff/model")

def make_multimode_bottom_bz(nx: int = 64, ny: int = 64, lx: float = 2.0 * np.pi, ly: float = 2.0 * np.pi,) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    bz_bottom = (
        np.sin(X) * np.cos(Y) + 0.5 * np.sin(2.0 * X + 0.3) * np.sin(Y)
    ).astype(np.float32)
    #単純な磁場を生成, 第一項:1周期の波. 第二:高周波で位相をずらす

    return lx, ly, x, y, bz_bottom.T

def solve_potential_field_fft_cartesian_at_z(b0_grid: np.ndarray, lx: float, ly: float, lz: float, z: np.ndarray,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    b0_grid = np.asarray(b0_grid, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if z.ndim != 1:
        raise ValueError("z must be a one-dimensional array.")

    if np.any(z < 0.0) or np.any(z > lz):
        raise ValueError("z must satisfy 0 <= z <= lz.")

    ny, nx = b0_grid.shape
    nz = z.size

    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)

    b0_hat = np.fft.rfft2(b0_grid)

    nkx = nx // 2 + 1

    psi_hat = np.zeros((nz, ny, nkx), dtype=np.complex128)
    bx_hat = np.zeros_like(psi_hat)
    by_hat = np.zeros_like(psi_hat)
    bz_hat = np.zeros_like(psi_hat)

    for j in range(ny):
        kyj = ky[j]

        for i in range(nkx):
            kxi = kx[i]
            k = np.sqrt(kxi**2 + kyj**2)

            b0_mode = b0_hat[j, i]

            if k == 0.0:
                # Mean vertical flux mode:
                #
                # Psi = B0_mean * (Lz - z)
                # Bz  = B0_mean
                psi_mode = b0_mode * (lz - z)
                bz_mode = np.full_like(z, b0_mode, dtype=np.complex128)
            else:
                # Stable expressions:
                #
                # sinh(k(L-z)) / cosh(kL)
                # = [exp(-kz) - exp(-k(2L-z))] / [1 + exp(-2kL)]
                #
                # cosh(k(L-z)) / cosh(kL)
                # = [exp(-kz) + exp(-k(2L-z))] / [1 + exp(-2kL)]
                e1 = np.exp(-k * z)
                e2 = np.exp(-k * (2.0 * lz - z))
                den = 1.0 + np.exp(-2.0 * k * lz)

                sinh_ratio = (e1 - e2) / den
                cosh_ratio = (e1 + e2) / den

                psi_mode = (b0_mode / k) * sinh_ratio
                bz_mode = b0_mode * cosh_ratio

            psi_hat[:, j, i] = psi_mode

            bx_hat[:, j, i] = -1j * kxi * psi_mode
            by_hat[:, j, i] = -1j * kyj * psi_mode
            bz_hat[:, j, i] = bz_mode

    psi = np.fft.irfft2(psi_hat, s=(ny, nx), axes=(1, 2))
    bx = np.fft.irfft2(bx_hat, s=(ny, nx), axes=(1, 2))
    by = np.fft.irfft2(by_hat, s=(ny, nx), axes=(1, 2))
    bz = np.fft.irfft2(bz_hat, s=(ny, nx), axes=(1, 2))
        # (z, y, x) -> (x, y, z)
    psi = np.transpose(psi, (2, 1, 0))
    bx = np.transpose(bx, (2, 1, 0))
    by = np.transpose(by, (2, 1, 0))
    bz = np.transpose(bz, (2, 1, 0))

    return psi, bx, by, bz


class PINN_NN(nn.Module):
    
    def __init__(self, Lx, Ly, Lz, width=128, depth=5):
        super().__init__()
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        # x, y は sin/cos で入力して水平周期境界を厳密に満たす: sin x, cos x, sin y, cos y, z
        dims = [5] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(width, 3)

    def forward(self, x, y, z):
        xx = 2.0 * np.pi * x / self.Lx
        yy = 2.0 * np.pi * y / self.Ly
        zz = 2.0 * z / self.Lz - 1.0
        h = torch.cat([torch.sin(xx), torch.cos(xx), torch.sin(yy), torch.cos(yy), zz], dim=1)
        for lin in self.hidden:
            h = torch.tanh(lin(h))
        return self.out(h)


def grad(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True,)[0]

def curl_and_div(B, x, y, z):
    d = [[grad(B[:, c:c + 1], v) for v in (x, y, z)] for c in range(3)]  
    J = torch.cat([d[2][1] - d[1][2], d[0][2] - d[2][0], d[1][0] - d[0][1]], dim=-1)
    return J, d[0][0] + d[1][1] + d[2][2]

def current_free_losses(model, x, y, z):
    B = model(x, y, z)
    J, div_b = curl_and_div(B, x, y, z)
    # |J|^2  (ポテンシャル場の条件は J = 0。J x B = 0 (force-free) ではない)
    loss_cur = torch.mean(torch.sum(J**2, dim=-1))
    # |div B|^2
    loss_div = torch.mean(div_b**2)

    return loss_cur, loss_div


class Bottom:
    """底面 Bz の周期的な双線形補間。格子点 i は x = i * dx (i = 0..nx-1)、x = nx * dx は i = 0 と同じ点。"""
    def __init__(self, bz0, dx, dy, B0, device):
        self.g = torch.as_tensor(bz0 / B0, dtype=torch.float32, device=device)  # (nx, ny)
        self.dx, self.dy = dx, dy                                              # 無次元の格子幅

    def __call__(self, x, y):
        nx, ny = self.g.shape
        gx = (x / self.dx).squeeze(-1)
        gy = (y / self.dy).squeeze(-1)
        i0 = gx.floor().long()
        j0 = gy.floor().long()
        tx = (gx - i0)[:, None]
        ty = (gy - j0)[:, None]
        i0, j0 = i0 % nx, j0 % ny
        i1, j1 = (i0 + 1) % nx, (j0 + 1) % ny
        g = self.g[..., None]
        return ((1 - tx) * (1 - ty) * g[i0, j0] + tx * (1 - ty) * g[i1, j0]
                + (1 - tx) * ty * g[i0, j1] + tx * ty * g[i1, j1])



def make_points(n, lx, ly, lz, device):
    x = (lx * torch.rand(n, 1, device=device)).requires_grad_(True)
    y = (ly * torch.rand(n, 1, device=device)).requires_grad_(True)
    z = (lz * torch.rand(n, 1, device=device)).requires_grad_(True)
    return x, y, z


def make_points_near_top(n, lx, ly, lz, device, thickness=0.2):
    x = (lx * torch.rand(n, 1, device=device)).requires_grad_(True)
    y = (ly * torch.rand(n, 1, device=device)).requires_grad_(True)

    # z in [(1-thickness)*lz, lz]
    z = lz * (1.0 - thickness * torch.rand(n, 1, device=device))
    z.requires_grad_(True)

    return x, y, z

def lr_at(step):
    """学習率（tanh 型）。ステップ数だけで決まり、総ステップ数には依存しない。"""
    return LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1.0 - math.tanh((step - LR_T_HALF) / LR_T_WIDTH))

def create_model_run_dir(root=MODEL_ROOT):
    """Create the next available relative directory: model/model0, model/model1, ..."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        match = re.fullmatch(r"model(\d+)", path.name)
        if path.is_dir() and match:
            existing.append(int(match.group(1)))
    run_dir = root / f"model{max(existing, default=-1) + 1}"
    run_dir.mkdir()
    return run_dir


def open_loss_log(run_dir):
    """Open loss.txt and write its TSV header. Keep the returned handle open while training."""
    path = Path(run_dir) / "loss.txt"
    f = path.open("w", encoding="utf-8")
    f.write(
        "total_loss\tcurrent\tdivB\tbottom_bz\ttop_h\t"
        "weight_current\tweight_divB\tweight_bottom\tlr\n"
    )
    f.flush()
    return f


def log_step(loss_file, step_num, total_loss, loss_cur, loss_div, loss_bottom, loss_top, weights_used, lr):
    """Write one row for each completed optimizer step and flush immediately."""
    vals = [
        total_loss, loss_cur, loss_div, loss_bottom, loss_top,
        weights_used[0], weights_used[1], weights_used[2], lr,
    ]
    loss_file.write(str(vals[0]) + "\t" + "\t".join(f"{float(v):.12e}" for v in vals[1:]) + "\n")
    loss_file.flush()


def save_checkpoint(model, run_dir, step_num):
    """Save model weights as model/modelN/<step>.pt."""
    path = Path(run_dir) / f"{step_num}.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")
    return path


def save_executed_kernel_code(run_dir, source_file=None):
    """Save the source of a regular Python script to executed_kernel.py.

    For a normal .py file, pass source_file=__file__ when calling this
    function. The script is copied into run_dir/executed_kernel.py.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "executed_kernel.py"

    if source_file is None:
        raise ValueError(
            "source_file must be specified for a regular Python script. "
            "Call save_executed_kernel_code(run_dir, source_file=__file__)."
        )

    source_path = Path(source_file).resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Python source file not found: {source_path}")

    if source_path == out.resolve():
        raise ValueError("source_file and output file refer to the same path.")

    shutil.copy2(source_path, out)
    print(f"Saved Python source to: {out.resolve()}")
    return out

def train(b0_grid, lx, ly, lz, steps=STEPS, n_pde=N_PDE, n_bc=N_BC, device="cuda", run_dir = None):
    # b0_grid: 底面 Bz, shape (nx, ny), x = i * lx / nx (endpoint=False の周期格子)
    nx, ny = b0_grid.shape
    # 規格化は入力データ(底面 Bz)だけで決める。参照解は使わない
    B0 = float(np.sqrt(np.mean(b0_grid**2)))
    print(B0)
    L0 = lx

    Lx, Ly, Lz = lx / L0, ly / L0, lz / L0
    h = (Lx / nx, Ly / ny)   # 周期格子なので (n - 1) ではなく n で割る

    bottom = Bottom(b0_grid, h[0], h[1], B0, device)
    model = PINN_NN(Lx, Ly, Lz).to(device)

    a = 1.0 - 1.0 / np.sqrt(Z_RATIO)
    c = 1.0 - (1.0 - a) ** 3

    def rand(n, L):
        return (0.0 + 1.0 * torch.rand(n, 1, device=device)) * L

    def rand_z(n):
        u = torch.rand(n, 1, device=device)
        return (1.0 - (1.0 - c * u) ** (1.0 / 3.0)) / a * Lz

    weight = [W_CUR, W_DIV, W_BC]

    opt = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_at(step) / LR_MAX)

    loss_file = open_loss_log(run_dir)



    for step in range(steps):
        opt.zero_grad()
        x, y, z = make_points(n_pde, Lx, Ly, Lz, device)
        xti, yti, zti = make_points_near_top(n_pde // 2, Lx, Ly, Lz, device)


        xb, yb = rand(n_bc, Lx), rand(n_bc, Ly)
        zb = torch.zeros(n_bc, 1, device=device)

        # ---- Interior current-free and divB losses ----

        loss_cur, loss_div = current_free_losses(model, x, y, z)
        loss_cur_t, loss_div_t = current_free_losses(model, xti, yti, zti)
        loss_cur = loss_cur + loss_cur_t
        loss_div = loss_div + loss_div_t

        # ---- Bottom boundary: Bz = B0 ----
        # 観測量は底面の Bz だけなので Bx, By は教師にしない

        bottom_pred = model(xb, yb, zb)
        g = bottom(xb, yb)
        loss_bottom = torch.mean((bottom_pred[:, 2:3] - g) ** 2)

        # ---- Top boundary: Bx = By = 0 (psi = 0, FFT 参照解と同じ条件) ----

        top_pred = model(xb, yb, torch.full_like(zb, Lz))
        loss_top = torch.mean(torch.sum(top_pred[:, :2] ** 2, dim=-1))

        weights_used = tuple(weight)
        # ---- Total loss ----
        loss = (weight[0] * loss_cur + weight[1] * loss_div + weight[2] * loss_bottom + W_TOP * loss_top)

        loss.backward()
        opt.step()
        scheduler.step()

        step_num = step + 1

        log_step(loss_file, step_num, loss.item(), loss_cur.item(), loss_div.item(), loss_bottom.item(), loss_top.item(), weights_used, opt.param_groups[0]["lr"],)

        if step_num % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, run_dir, step_num)

        if step % 500 == 0:
            print(
                step,
                {
                    "loss": float(loss.detach().cpu()),
                    "current": float(loss_cur.detach().cpu()),
                    "divB": float(loss_div.detach().cpu()),
                    "bottom_bz": float(loss_bottom.detach().cpu()),
                    "top_h": float(loss_top.detach().cpu()),
                },
            )

    loss_file.close()

    return model, B0


@torch.no_grad()
def evaluate_field(model, x1d, y1d, z1d, device="cuda"):
    """
    Returns Bx, By, Bz on meshgrid with shape (nz, ny, nx).
    """
    X, Y, Z = torch.meshgrid(
        torch.as_tensor(x1d, dtype=torch.float32, device=device),
        torch.as_tensor(y1d, dtype=torch.float32, device=device),
        torch.as_tensor(z1d, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x = X.reshape(-1, 1)
    y = Y.reshape(-1, 1)
    z = Z.reshape(-1, 1)

    B = model(x, y, z)

    nx = len(x1d)
    ny = len(y1d)
    nz = len(z1d)

    bx = B[:, 0].reshape(nx, ny, nz).permute(0, 1, 2).detach().cpu()
    by = B[:, 1].reshape(nx, ny, nz).permute(0, 1, 2).detach().cpu()
    bz = B[:, 2].reshape(nx, ny, nz).permute(0, 1, 2).detach().cpu()

    return bx, by, bz


def evaluate_residuals(model, x1d, y1d, z1d, device="cuda"):
    """
    Evaluate |J x B| and divB on a grid.
    """
    X, Y, Z = torch.meshgrid(
        torch.as_tensor(x1d, dtype=torch.float32, device=device),
        torch.as_tensor(y1d, dtype=torch.float32, device=device),
        torch.as_tensor(z1d, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x = X.reshape(-1, 1).detach().requires_grad_(True)
    y = Y.reshape(-1, 1).detach().requires_grad_(True)
    z = Z.reshape(-1, 1).detach().requires_grad_(True)

    B = model(x, y, z)
    J, div_b = curl_and_div(B, x, y, z)
    j_cross_b = torch.cross(J, B, dim=-1)

    nx = len(x1d)
    ny = len(y1d)
    nz = len(z1d)

    jxb_abs = torch.sqrt(torch.sum(j_cross_b**2, dim=-1))
    div_b = div_b[:, 0]

    jxb_abs = jxb_abs.reshape(nx, ny, nz).permute(0, 1, 2).detach().cpu()
    div_b = div_b.reshape(nx, ny, nz).permute(0, 1, 2).detach().cpu()

    return jxb_abs, div_b



def main():

    run_dir = create_model_run_dir()
    save_executed_kernel_code(run_dir, source_file=__file__)
    print(f"Output directory: {run_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lx, ly, x, y, b0 = make_multimode_bottom_bz()
    lz = lx
    z = np.linspace(0.0, lz, len(x))

    # Train
    model, B0 = train(b0_grid=b0,  lx=lx,  ly=ly,  lz=lz,  device=device,  run_dir=run_dir,)

    _, _, bz = evaluate_field(model,  x / lx,  y / lx,  z / lx,  device=device,)

    bz = B0 * bz

    _, bx_ref, by_ref, bz_ref = (solve_potential_field_fft_cartesian_at_z(b0_grid=b0.T,  lx=lx,  ly=ly,  lz=lz,  z=z,))


    jxb_abs, div_b = evaluate_residuals(model,  x / lx,  y / lx,  z / lx,  device=device,)
    jxb_abs = B0 * jxb_abs
    div_b = B0 * div_b


    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15), constrained_layout=True)

    # plot bottom Bz
    ax = axs[0, 0]
    im = ax.pcolormesh(x, y, bz[:, :, 0].numpy().T, shading="auto", vmin = -1, vmax = 1)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.08, shrink=0.8)
    cb.set_label("Bz at z=0", labelpad=4)
    cb.ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # plot top Bz
    ax = axs[0, 1]
    im = ax.pcolormesh(x, y, bz[:, :, -1].numpy().T, shading="auto", vmin = -0.0003, vmax = 0.0003)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.08, shrink=0.8)
    cb.set_label("Bz at z=Lz", labelpad=4)
    cb.ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # plot top Bz - potential reference
    ax = axs[1, 0]
    im = ax.pcolormesh(x, y, (bz[:, :, -1] - bz_ref[:, :, -1]).numpy().T, shading="auto", vmin = -0.001, vmax = 0.001)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.08, shrink=0.8)
    cb.set_label("Bz (CF PINN) - Bz (potential FFT) at z=Lz", labelpad=4)
    cb.ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # plot potential reference top Bz
    ax = axs[1, 1]
    im = ax.pcolormesh(x, y, bz_ref[:, :, -1].T, shading="auto", vmin = -0.0003, vmax = 0.0003)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.08, shrink=0.8)
    cb.set_label("Bz (potential FFT) at z=Lz", labelpad=4)
    cb.ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # plot |J x B| at middle height
    iz = len(z) // 2
    ax = axs[2, 0]
    im = ax.pcolormesh(x, y, jxb_abs[:, :, iz].numpy().T, shading="auto", vmin = 0, vmax = 0.003)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.08, shrink=0.8)
    cb.set_label("|J x B| at middle z", labelpad=4)
    cb.ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # plot div B at middle height
    ax = axs[2, 1]
    im = ax.pcolormesh(x, y, abs(div_b[:, :, iz]).numpy().T, shading="auto", vmin = 0, vmax = 0.01)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.08, shrink=0.8)
    cb.set_label("div B at middle z", labelpad=4)
    cb.ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    fig_file = os.path.join(run_dir, "cff_pinns.png")
    fig.savefig(fig_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_file}")

    return bz, run_dir



if __name__ == "__main__":
    bz, run_dir = main()

    lx, ly, x, y, b0 = make_multimode_bottom_bz()
    lz = lx
    z = np.linspace(0.0, lz, len(x))
    _, bx_ref, by_ref, bz_ref = solve_potential_field_fft_cartesian_at_z(b0_grid=b0.T, lx=lx, ly=ly, lz=lz, z=z)
    mid_z = len(z) // 2

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 11), constrained_layout=True)

    # PINN Bz at middle z
    ax = axs[0]
    im = ax.pcolormesh(x, y, bz[:, :, mid_z].detach().cpu().numpy().T, shading="auto", vmin = -0.01, vmax = 0.01)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.08, shrink=0.8)
    cb.set_label("Bz at middle z", labelpad=4)
    cb.ax.xaxis.set_ticks_position("top")


    # Potential reference Bz at middle z
    ax = axs[1]
    im = ax.pcolormesh(x, y, bz_ref[:, :, mid_z].T, shading="auto", vmin = -0.01, vmax = 0.01)
    cb = fig.colorbar(im, ax=ax, location="top", pad=0.08, shrink=0.8)
    cb.set_label("Bz (potential FFT) at middle z", labelpad=4)
    cb.ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    fig_file = os.path.join(run_dir, "mid_pinns.png")
    fig.savefig(fig_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_file}")