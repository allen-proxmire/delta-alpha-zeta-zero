Import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd, welch
from scipy.signal import detrend
import pandas as pd

# CONFIG
TARGET_FREQS = [0.35153, 0.38895, 0.47871]
NPERSEG = 2048
NOVERLAP = NPERSEG // 2
FS = 1.0
SURROGATES = 500
RNG = np.random.default_rng(2025)

# Helper: compute delta-alpha for a primes slice
def delta_alpha_from_primes(p):
    alpha = np.arctan(p[:-1] / p[1:])
    return np.diff(alpha)

# Split into disjoint blocks by index (equal-length Δα blocks)
# Choose block_size in Δα samples (not prime value)
full_da = detrend(delta_alpha)  # from your earlier compute
total = len(full_da)
block_size = int(total // 3)  # e.g., 3 blocks; change to 4+ as desired
blocks = [full_da[i*block_size:(i+1)*block_size] for i in range(3)]
block_labels = ['block1', 'block2', 'block3']

# Optionally trim any small remainder
for i,b in enumerate(blocks):
    blocks[i] = detrend(b)

# Pairwise coherence and cross-spectrum
pairs = []
for i in range(len(blocks)):
    for j in range(i+1, len(blocks)):
        x = blocks[i]; y = blocks[j]
        f, coh_xy = coherence(x, y, fs=FS, window='hann', nperseg=min(NPERSEG,len(x)), noverlap=NOVERLAP)
        # cross-spectrum for phase: csd returns complex Sxy
        f_csd, Sxy = csd(x, y, fs=FS, window='hann', nperseg=min(NPERSEG,len(x)), noverlap=NOVERLAP)
        phase = np.angle(Sxy)  # phase vs frequency
        pairs.append({
            'pair_label': f'{block_labels[i]}__{block_labels[j]}',
            'f': f, 'coh': coh_xy, 'phase': phase
        })

# Plot coherence magnitude for each pair
plt.figure(figsize=(10,6))
for p in pairs:
    plt.plot(p['f'], p['coh'], alpha=0.8, label=p['pair_label'])
for ft in TARGET_FREQS:
    plt.axvline(ft, color='k', ls='--', alpha=0.5)
plt.xlim(0.10, 0.55)
plt.xlabel('Frequency f'); plt.ylabel('Coherence')
plt.title('Cross-sample coherence between Δα blocks')
plt.legend(); plt.grid(True, ls=':', alpha=0.5)
plt.tight_layout(); plt.savefig('cross_sample_coherence_overlay.png', dpi=200)
plt.show()

# Extract coherence at target frequencies and phase
def nearest_idx(fgrid, val): return int(np.argmin(np.abs(fgrid - val)))

rows = []
for p in pairs:
    fgrid = p['f']
    for tf in TARGET_FREQS:
        idx = nearest_idx(fgrid, tf)
        rows.append({
            'pair': p['pair_label'],
            'target_f': tf,
            'coherence': float(p['coh'][idx]),
            'phase': float(p['phase'][idx])
        })
df = pd.DataFrame(rows)
df.to_csv('cross_sample_coherence_summary.csv', index=False)
print(df)

# SURROGATE SIGNIFICANCE: phase-randomize one block and recompute coherence
# Choose a reference pair (first pair) to test significance at each target freq
ref_pair = pairs[0]  # test coherence of block1 vs block2 for example
x = blocks[0]; y = blocks[1]
f_ref = ref_pair['f']
coh_ref = ref_pair['coh']

# locate indices for targets once
target_idxs = [nearest_idx(f_ref, tf) for tf in TARGET_FREQS]
obs_vals = [coh_ref[idx] for idx in target_idxs]

# Generate surrogates by phase-randomizing one signal (y) keeping amplitude
Yfft = np.fft.rfft(y)
magY = np.abs(Yfft)
Nfft = len(Yfft)
sur_coh = np.zeros((SURROGATES, len(TARGET_FREQS)))
for s in range(SURROGATES):
    phases = RNG.uniform(0, 2*np.pi, size=Nfft)
    phases[0] = 0.0
    if Nfft % 2 == 0: phases[-1] = 0.0
    Ysur = magY * np.exp(1j*phases)
    ysur = np.fft.irfft(Ysur, n=len(y))
    f_s, coh_s = coherence(x, ysur, fs=FS, window='hann', nperseg=min(NPERSEG,len(x)), noverlap=NOVERLAP)
    for k, idx in enumerate(target_idxs):
        # find nearest index in f_s grid
        idx_s = nearest_idx(f_s, f_ref[idx])
        sur_coh[s,k] = coh_s[idx_s]

# empirical p-values
pvals = [(np.mean(sur_coh[:,k] >= obs_vals[k])) for k in range(len(TARGET_FREQS))]
for tf, obs, p in zip(TARGET_FREQS, obs_vals, pvals):
    print(f"Target {tf:.5f}: observed coh={obs:.4f}, surrogate p={p:.4f}")

# Save surrogate histograms
import os
os.makedirs('cross_coherence_surrogates', exist_ok=True)
for k, tf in enumerate(TARGET_FREQS):
    plt.figure(figsize=(6,4))
    plt.hist(sur_coh[:,k], bins=40, color='gray', alpha=0.8)
    plt.axvline(obs_vals[k], color='red', lw=2, label='observed')
    plt.title(f"Surrogate coherence distribution at f≈{tf:.5f}")
    plt.xlabel('coherence'); plt.ylabel('count')
    plt.legend(); plt.tight_layout()
    plt.savefig(f'cross_coherence_surrogate_hist_f{tf:.5f}.png', dpi=200)
    plt.show()