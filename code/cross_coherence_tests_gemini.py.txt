import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd, detrend
import pandas as pd
from numpy.fft import rfft, irfft

# --- Initial Data Setup (Recreating environment from previous turns) ---
MAX_PRIME_VALUE = 2_000_000
TARGET_FREQS = [0.35153, 0.38895, 0.47871] # Sorted low to high for cleaner table output

def get_primes_up_to_n(n: int) -> np.ndarray:
    """Vectorized Sieve of Eratosthenes."""
    if n < 2: return np.array([], dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    limit = int(np.sqrt(n))
    for p in range(2, limit + 1):
        if is_prime[p]:
            is_prime[p*p:n+1:p] = False
    return np.flatnonzero(is_prime).astype(np.int64)

def calculate_delta_alpha(p_list: np.ndarray) -> np.ndarray:
    """∆α = α(n+1) - α(n), with α(n) = arctan(p_n / p_{n+1})."""
    alpha = np.arctan(p_list[:-1] / p_list[1:])
    return np.diff(alpha)

# Generate the full Δα signal up to 2M for the analysis
primes = get_primes_up_to_n(MAX_PRIME_VALUE)
delta_alpha = calculate_delta_alpha(primes)
print(f"Δα signal length (N) used for blocking: {len(delta_alpha):,}")
# ---------------------------------------------------------------------


# CONFIG
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
full_da = detrend(delta_alpha)
total = len(full_da)
block_size = int(total // 3)  # 3 blocks
blocks = [full_da[i*block_size:(i+1)*block_size] for i in range(3)]
block_labels = ['block1', 'block2', 'block3']

# Trim any remainder and detrend again (to remove any residual mean/trend after slicing)
for i,b in enumerate(blocks):
    blocks[i] = detrend(b)
print(f"Block size: {len(blocks[0]):,} samples.")

# Pairwise coherence and cross-spectrum
pairs = []
for i in range(len(blocks)):
    for j in range(i+1, len(blocks)):
        x = blocks[i]; y = blocks[j]
        # Ensure we use the smaller of NPERSEG or block size
        nperseg_eff = min(NPERSEG, len(x))

        f, coh_xy = coherence(x, y, fs=FS, window='hann', nperseg=nperseg_eff, noverlap=NOVERLAP)
        # cross-spectrum for phase: csd returns complex Sxy
        f_csd, Sxy = csd(x, y, fs=FS, window='hann', nperseg=nperseg_eff, noverlap=NOVERLAP)
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
plt.title('Cross-sample coherence between $\Delta\alpha$ blocks')
plt.legend(); plt.grid(True, ls=':', alpha=0.5)
plt.tight_layout(); plt.savefig('cross_sample_coherence_overlay.png', dpi=200)
print("\nSaved 'cross_sample_coherence_overlay.png'")


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
print("\nCoherence Summary:")
print(df)
print("\nSaved 'cross_sample_coherence_summary.csv'")

# SURROGATE SIGNIFICANCE: phase-randomize one block and recompute coherence
# Choose a reference pair (first pair) to test significance at each target freq
ref_pair = pairs[0]  # block1 vs block2
x = blocks[0]; y = blocks[1]
f_ref = ref_pair['f']
coh_ref = ref_pair['coh']

# locate indices for targets once
target_idxs = [nearest_idx(f_ref, tf) for tf in TARGET_FREQS]
obs_vals = [coh_ref[idx] for idx in target_idxs]

# Generate surrogates by phase-randomizing one signal (y) keeping amplitude
Yfft = rfft(y) # Use rfft for real input
magY = np.abs(Yfft)
Nfft = len(Yfft)
sur_coh = np.zeros((SURROGATES, len(TARGET_FREQS)))
for s in range(SURROGATES):
    # Phase randomize the interior components (excluding DC and Nyquist if present)
    phases = RNG.uniform(0, 2*np.pi, size=Nfft)
    phases[0] = 0.0 # DC phase
    if len(y) % 2 == 0 and Nfft > 1: phases[-1] = 0.0 # Nyquist phase
    
    Ysur = magY * np.exp(1j*phases)
    ysur = irfft(Ysur, n=len(y))

    # Compute coherence between the fixed signal (x) and the surrogate (ysur)
    f_s, coh_s = coherence(x, ysur, fs=FS, window='hann', nperseg=min(NPERSEG,len(x)), noverlap=NOVERLAP)
    for k, idx in enumerate(target_idxs):
        # find nearest index in f_s grid (should be the same as f_ref)
        idx_s = nearest_idx(f_s, f_ref[idx])
        sur_coh[s,k] = coh_s[idx_s]

# empirical p-values
pvals = [(np.mean(sur_coh[:,k] >= obs_vals[k])) for k in range(len(TARGET_FREQS))]

print("\n--- Coherence Significance Test (Block 1 vs Block 2) ---")
print(f"{'Target Freq':>12s} {'Observed Coherence':>20s} {'Surrogate P-value':>20s}")
for tf, obs, p in zip(TARGET_FREQS, obs_vals, pvals):
    print(f"{tf:12.5f} {obs:20.4f} {p:20.4f}")

# Save surrogate histograms
import os
os.makedirs('cross_coherence_surrogates', exist_ok=True)
for k, tf in enumerate(TARGET_FREQS):
    plt.figure(figsize=(6,4))
    plt.hist(sur_coh[:,k], bins=40, color='gray', alpha=0.8, edgecolor='black', zorder=1)
    plt.axvline(obs_vals[k], color='red', lw=2, label=f'Observed Coherence ({obs_vals[k]:.4f})', zorder=2)
    plt.title(f"Surrogate coherence distribution at f$\approx${tf:.5f}")
    plt.xlabel('Coherence Magnitude'); plt.ylabel('Count')
    plt.legend(); plt.grid(True, ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'cross_coherence_surrogate_hist_f{tf:.5f}.png', dpi=200)
    print(f"Saved surrogate histogram for f≈{tf:.5f}: 'cross_coherence_surrogates/cross_coherence_surrogate_hist_f{tf:.5f}.png'")