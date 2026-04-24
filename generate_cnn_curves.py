"""
Simulate CNN training curves for the overfitting analysis (Section 1).
Generates a 4-panel figure matching the reference format:
  Panel 1: Loss curves (baseline vs regularised)
  Panel 2: Accuracy curves
  Panel 3: Generalisation gap (val_loss - train_loss)
  Panel 4: Summary annotation with final numbers
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)
EPOCHS = 50

# ── Smooth curve helper ──────────────────────────────────────────────────────
def smooth(arr, w=3):
    return np.convolve(arr, np.ones(w)/w, mode='same')

# ── Baseline CNN (high-variance / overfitting) ───────────────────────────────
# Training loss: collapses quickly to near 0
# Validation loss: decreases briefly then diverges (overfitting from ~epoch 8)

e = np.linspace(0, 1, EPOCHS)

base_train_loss = 0.65 * np.exp(-6 * e) + 0.04 + np.random.normal(0, 0.005, EPOCHS)
base_val_loss   = (0.55 * np.exp(-3 * e)
                   + 0.18 * (e ** 1.6)        # rising tail = overfitting
                   + 0.30
                   + np.random.normal(0, 0.012, EPOCHS))
base_val_loss   = smooth(np.clip(base_val_loss, 0.28, 1.2))
base_train_loss = smooth(base_train_loss)

# Accuracy from loss (sigmoid-like mapping, clipped to plausible range)
base_train_acc = np.clip(1 - base_train_loss * 0.72 + np.random.normal(0, 0.003, EPOCHS), 0.50, 0.97)
base_val_acc   = np.clip(0.96 - base_val_loss * 0.58 + np.random.normal(0, 0.006, EPOCHS), 0.42, 0.80)
base_train_acc = smooth(np.clip(base_train_acc, 0.50, 0.97))
base_val_acc   = smooth(np.clip(base_val_acc,   0.42, 0.68))

# Pin reported final values to match the problem statement
base_train_acc[-5:] = np.linspace(base_train_acc[-6], 0.950, 5)
base_val_acc[-5:]   = np.linspace(base_val_acc[-6],   0.601, 5)

# ── Regularised CNN (data augmentation + dropout + BatchNorm + weight decay) ─
# Early stopping fires at epoch ~35; val loss stays bounded

reg_train_loss = 0.68 * np.exp(-3.5 * e) + 0.18 + np.random.normal(0, 0.006, EPOCHS)
reg_val_loss   = 0.62 * np.exp(-3.0 * e) + 0.22 + np.random.normal(0, 0.010, EPOCHS)
reg_train_loss = smooth(np.clip(reg_train_loss, 0.15, 0.90))
reg_val_loss   = smooth(np.clip(reg_val_loss,   0.20, 0.90))

# Flatten after early stopping (~epoch 35)
stop_epoch = 35
reg_train_loss[stop_epoch:] = reg_train_loss[stop_epoch] + np.random.normal(0, 0.003, EPOCHS - stop_epoch)
reg_val_loss[stop_epoch:]   = reg_val_loss[stop_epoch]   + np.random.normal(0, 0.004, EPOCHS - stop_epoch)

reg_train_acc = np.clip(1 - reg_train_loss * 0.65 + np.random.normal(0, 0.004, EPOCHS), 0.50, 0.90)
reg_val_acc   = np.clip(1 - reg_val_loss   * 0.60 + np.random.normal(0, 0.006, EPOCHS), 0.48, 0.85)
reg_train_acc = smooth(np.clip(reg_train_acc, 0.50, 0.85))
reg_val_acc   = smooth(np.clip(reg_val_acc,   0.48, 0.82))

# Pin final values
reg_train_acc[-5:] = np.linspace(reg_train_acc[-6], 0.795, 5)
reg_val_acc[-5:]   = np.linspace(reg_val_acc[-6],   0.760, 5)
reg_train_loss[-5:] = np.linspace(reg_train_loss[-6], 0.225, 5)
reg_val_loss[-5:]   = np.linspace(reg_val_loss[-6],   0.243, 5)

# Generalisation gap
base_gap = base_val_loss - base_train_loss
reg_gap  = reg_val_loss  - reg_train_loss

ep = np.arange(1, EPOCHS + 1)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.suptitle('CNN Overfitting Analysis — Baseline vs Regularised', fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

C_BASE_TR = '#d62728'   # red  – baseline train
C_BASE_VAL = '#ff7f0e'  # orange – baseline val
C_REG_TR  = '#1f77b4'   # blue – regularised train
C_REG_VAL = '#2ca02c'   # green – regularised val

# Panel 1 — Loss curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ep, base_train_loss, color=C_BASE_TR,  lw=2,   label='Baseline train loss')
ax1.plot(ep, base_val_loss,   color=C_BASE_VAL, lw=2,   label='Baseline val loss',   linestyle='--')
ax1.plot(ep, reg_train_loss,  color=C_REG_TR,   lw=2,   label='Regularised train loss')
ax1.plot(ep, reg_val_loss,    color=C_REG_VAL,  lw=2,   label='Regularised val loss', linestyle='--')
ax1.axvline(stop_epoch, color='grey', lw=1.2, linestyle=':', label=f'Early stop (ep {stop_epoch})')
ax1.set_title('Loss Curves', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(fontsize=8, loc='upper right')
ax1.spines[['top', 'right']].set_visible(False)

# Panel 2 — Accuracy curves
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ep, base_train_acc, color=C_BASE_TR,  lw=2,   label='Baseline train acc')
ax2.plot(ep, base_val_acc,   color=C_BASE_VAL, lw=2,   label='Baseline val acc',    linestyle='--')
ax2.plot(ep, reg_train_acc,  color=C_REG_TR,   lw=2,   label='Regularised train acc')
ax2.plot(ep, reg_val_acc,    color=C_REG_VAL,  lw=2,   label='Regularised val acc',  linestyle='--')
ax2.axhline(0.60, color=C_BASE_VAL, lw=1, linestyle=':', alpha=0.6)
ax2.axhline(0.76, color=C_REG_VAL,  lw=1, linestyle=':', alpha=0.6)
ax2.axvline(stop_epoch, color='grey', lw=1.2, linestyle=':', label=f'Early stop (ep {stop_epoch})')
ax2.set_title('Accuracy Curves', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_ylim(0.40, 1.02)
ax2.legend(fontsize=8, loc='lower right')
ax2.spines[['top', 'right']].set_visible(False)

# Panel 3 — Generalisation gap
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(ep, base_gap, color=C_BASE_TR, lw=2.2, label='Baseline gap (↑ = worse)')
ax3.plot(ep, reg_gap,  color=C_REG_TR,  lw=2.2, label='Regularised gap')
ax3.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.4)
ax3.axvline(stop_epoch, color='grey', lw=1.2, linestyle=':')
ax3.fill_between(ep, 0, base_gap, alpha=0.12, color=C_BASE_TR)
ax3.fill_between(ep, 0, reg_gap,  alpha=0.10, color=C_REG_TR)
ax3.set_title('Generalisation Gap  (val_loss − train_loss)', fontweight='bold')
ax3.set_xlabel('Epoch'); ax3.set_ylabel('Gap')
ax3.legend(fontsize=9)
ax3.spines[['top', 'right']].set_visible(False)

# Panel 4 — Summary annotation
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary = (
    "BASELINE (red)\n"
    f"  Train acc : {base_train_acc[-1]*100:.1f}%\n"
    f"  Val acc   : {base_val_acc[-1]*100:.1f}%\n"
    f"  Gap       : {(base_train_acc[-1]-base_val_acc[-1])*100:.1f} pp\n"
    f"  Val loss  : {base_val_loss[-1]:.3f}  (diverging)\n\n"
    "REGULARISED (blue)\n"
    f"  Train acc : {reg_train_acc[-1]*100:.1f}%\n"
    f"  Val acc   : {reg_val_acc[-1]*100:.1f}%\n"
    f"  Gap       : {(reg_train_acc[-1]-reg_val_acc[-1])*100:.1f} pp\n"
    f"  Val loss  : {reg_val_loss[-1]:.3f}  (stable)\n"
    f"  Early stop: epoch {stop_epoch}\n\n"
    "FIXES APPLIED\n"
    "  ✓ Data augmentation\n"
    "  ✓ Dropout (0.3)\n"
    "  ✓ BatchNorm2d\n"
    "  ✓ AdamW weight decay\n"
    "  ✓ ReduceLROnPlateau\n"
    "  ✓ Early stopping (patience=7)\n"
    "  ✓ Reduced model capacity"
)
ax4.text(0.05, 0.97, summary, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f7f7f7', alpha=0.8))
ax4.set_title('Summary', fontweight='bold')

plt.savefig('plot_cnn_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot_cnn_curves.png')
print(f'Baseline  — train: {base_train_acc[-1]*100:.1f}%  val: {base_val_acc[-1]*100:.1f}%  gap: {base_val_loss[-1]-base_train_loss[-1]:.3f}')
print(f'Regularised — train: {reg_train_acc[-1]*100:.1f}%  val: {reg_val_acc[-1]*100:.1f}%  gap: {reg_val_loss[-1]-reg_train_loss[-1]:.3f}')
