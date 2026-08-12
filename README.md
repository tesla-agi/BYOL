# BYOL from Scratch

A from-scratch PyTorch implementation of **BYOL** (Bootstrap Your Own Latent) — self-supervised representation learning with no labels, no negative pairs, and no decoder. Trained and evaluated on CIFAR-10.

**Result: 80.07% linear-probe accuracy** on CIFAR-10 (100 epochs, ResNet-18 encoder, chance = 10%). A frozen encoder that never saw a single label produces features a bare linear classifier reads at 80%.

Built as part of a self-directed World Models curriculum — every line written from the paper, no self-supervised libraries used.

---

## Idea

Two augmented views of one image are pushed through two towers:

- **Online tower** — encoder → projector → predictor, trained by gradients.
- **Target tower** — encoder → projector (no predictor), frozen, updated only as an EMA of the online weights.

The online tower predicts the target tower's output; the loss is the normalized MSE (equivalently `2 − 2·cos`) between them, symmetrized over both views. The learning signal comes entirely from the augmentations: because two views share content but differ in nuisances (crop, colour, blur), the only way to agree is to encode what is invariant and discard the rest.

Collapse (encoder outputs a constant for every input) is prevented by three asymmetries: the **predictor** (online only), the **stop-gradient** (on the target), and the **EMA** target.

```
INPUT  view image                    [B, 3, 32, 32]
ENCODER  resnet18 (CIFAR stem)   ->  [B, 512]     <- representation (probe uses this)
PROJECTOR  512 -> 4096 -> 256    ->  [B, 256]
PREDICTOR  256 -> 4096 -> 256    ->  [B, 256]     (online only)

ONLINE :  view -> enc -> proj -> pred  =  p       (grads flow)
TARGET :  view -> enc -> proj          =  z'      (stop-grad, EMA weights)
LOSS   :  L = D(p1, z2) + D(p2, z1),  D(p,z) = 2 - 2*cos(norm(p), norm(z.detach()))
STEP   :  forward -> loss -> optimizer.step()  THEN  ema_update(target <- online)
```

---

## Files

| File | Role |
|------|------|
| `aug.py` | Two-view augmentation pipeline (crop, flip, jitter, grayscale, blur, solarize) — the supervision source |
| `networks.py` | `Encoder` (ResNet-18 with CIFAR stem surgery), `Projector`, `Predictor`, `Base` (encoder+projector) |
| `ema.py` | Build the frozen target copy and the EMA update `ξ ← τξ + (1−τ)θ` |
| `loss.py` | Normalized `2 − 2·cos` loss with stop-gradient on the target |
| `train.py` | Full training loop: two-tower forward, symmetric loss, optimizer step then EMA |
| `probe.py` | Linear evaluation: freeze encoder, train one linear layer on labels, report test accuracy |

---

## Usage

Requires PyTorch with MPS/CUDA/CPU, `torchvision`, and `tqdm`.

```bash
# train (downloads CIFAR-10 on first run; saves encoder checkpoints)
python train.py

# evaluate the frozen encoder with a linear probe
python probe.py
```

CIFAR-10 downloads automatically to `./data/`. Checkpoints are written to `./checkpoint/` (both are git-ignored).

---

## Implementation notes

- **CIFAR stem surgery** — the standard ResNet-18 stem (7×7 stride-2 conv + maxpool) crushes a 32px image before it learns anything. Replaced with a 3×3 stride-1 conv and the maxpool dropped, so the image keeps its spatial detail.
- **L2-normalize before the loss** — prevents the magnitude cheat (driving raw MSE to zero by shrinking vectors); once both vectors are unit-length the loss measures only directional agreement.
- **Step order** — the optimizer step runs *before* the EMA update each iteration, so the target trails the freshly-updated online weights (the ratchet).
- **BatchNorm in the probe** — the encoder is set to `.eval()` during probing so BatchNorm uses running stats and does not mutate under the frozen weights.

---

## Reference

Grill et al., *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning* (2020).
