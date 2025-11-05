#!/usr/bin/env python3
# Consolidated script for Prime-Wave analysis robustness checks.

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import periodogram, welch, get_window, detrend, firwin, filtfilt
from scipy.stats import norm
from scipy.integrate import trapezoid
from numpy.fft import rfft, irfft

# ---------------------------
# CONFIG
# ---------------------------
SAMPLE_LIMITS = [20_000, 200_000, 2_000_000]
TARGET_MODES = {
    "Fast Mode (T=2.089)": 0.47871,
    "Mid Mode (T=2.571)": 0.38895,
    "Slow Mode (T=2.845)": 0.35153,
}
FFT_WINDOW = "hann"
RANDOM_SEED = 2025 # Using 2025 for consistency across all calls
np.random.seed(RANDOM_SEED)

# Global variables initialized after sieve
primes = None
samples = []

# ---------------------------
# CORE HELPERS (from prime_wave_modes.py)
# ---------------------------
def get_primes_up_to_n(n: int) -> np.ndarray:
    """Vectorized Sieve of Eratosthenes; returns primes ≤ n as a NumPy array."""
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

# ---------------------------
# PSD & NULL TEST HELPERS (from the user's provided cells)
# ---------------------------
USE_WELCH = True
WELCH_NPERSEG = 4096
F_BAND = (0.10, 0.55)

def psd_for(signal: np.ndarray):
    """Generates PSD using periodogram or Welch, removes DC."""
    signal = detrend(signal, type='constant') # Ensure explicit detrending for consistency
    if USE_WELCH:
        # Use overlap=None to make null test simpler/faster, matching some of your usage
        f, Pxx = welch(signal, fs=1.0, window='hann',
                       nperseg=min(WELCH_NPERSEG, len(signal)),
                       noverlap=None, detrend=False, return_onesided=True)
    else:
        # Use periodogram with the global FFT_WINDOW
        f, Pxx = periodogram(signal, fs=1.0, window=get_window(FFT_WINDOW, signal.size, fftbins=True))
    if Pxx.size:
        Pxx[0] = 0.0
    return f, Pxx

def nearest_bin(fgrid: np.ndarray, f0: float) -> int:
    return int(np.argmin(np.abs(fgrid - f0)))

def run_null_test(da_signal: np.ndarray, N_PERM=200, save_fig=True):
    """Permutation null: shuffle Δα, compute PSD, evaluate target bins."""
    target_labels = list(TARGET_MODES.keys())
    target_freqs = np.array(list(TARGET_MODES.values()), dtype=float)

    # 1. PSD of the real series
    f, Pxx = psd_for(da_signal)
    target_idx = np.array([nearest_bin(f, tf) for tf in target_freqs], dtype=int)
    obs = Pxx[target_idx]

    # 2. Permutation loop
    null_vals = np.zeros((N_PERM, len(target_idx)), dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    for b in range(N_PERM):
        shuffled = da_signal.copy()
        rng.shuffle(shuffled)
        _, Pxx_null = psd_for(shuffled)
        null_vals[b, :] = Pxx_null[target_idx]

    # 3. Stats calculation
    mu = null_vals.mean(axis=0)
    sd = null_vals.std(axis=0, ddof=1)
    z = (obs - mu) / np.where(sd > 0, sd, np.inf)
    p = 2.0 * (1.0 - norm.cdf(np.abs(z))) # two-sided p from z

    print("\n--- Permutation Null (Δα shuffled) ---")
    print(f"{'Mode':28s} {'f_target':>9s} {'obs':>12s} {'μ_null':>12s} {'σ_null':>10s} {'z':>8s} {'p':>10s}")
    for k, label in enumerate(target_labels):
        print(f"{label:28s} {target_freqs[k]:9.5f} {obs[k]:12.5e} {mu[k]:12.5e} {sd[k]:10.5e} {z[k]:8.2f} {p[k]:10.2e}")

    # 4. Optional: overlay null PSDs
    if save_fig:
        fmin, fmax = F_BAND
        plt.figure(figsize=(12, 7))
        # plot a handful (e.g., 30) null spectra for appearance
        rng = np.random.default_rng(RANDOM_SEED) # Reset rng for plotting consistency
        show_m = min(30, N_PERM)
        for b in range(show_m):
            shuffled = da_signal.copy(); rng.shuffle(shuffled)
            f_null, Pxx_null = psd_for(shuffled)
            plt.plot(f_null, Pxx_null, color='#bbbbbb', alpha=0.25, linewidth=0.8)
        plt.plot(f, Pxx, color='#1f77b4', linewidth=1.8, label='Real Δα PSD')
        peak_colors = ['#d62728', '#9467bd', '#8c564b']
        for col, (lab, tf) in zip(peak_colors, TARGET_MODES.items()):
            idx = nearest_bin(f, tf)
            plt.axvline(f[idx], color=col, linestyle='--', alpha=0.7)
        plt.xlim(fmin, fmax)
        plt.ylim(0, np.percentile(Pxx[(f>=fmin)&(f<=fmax)], 99.9)*1.2)
        plt.title("Null Overlay: shuffled Δα PSDs (gray) vs real (blue)")
        plt.xlabel("Frequency (f)"); plt.ylabel("PSD")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figure_null_overlay.png", dpi=150)
        print("Saved 'figure_null_overlay.png'")

# ---------------------------
# SURROGATE HELPERS (from Cell C)
# ---------------------------
def phase_randomize(x, rng):
    X = rfft(x)
    mag = np.abs(X)
    ph  = np.angle(X)
    k   = np.arange(1, len(X)-1)
    new_ph = rng.uniform(0, 2*np.pi, size=k.size)
    Y = np.zeros_like(X, dtype=complex)
    Y[0] = X[0]
    if len(X) > 1:
        Y[-1] = X[-1] if len(x) % 2 == 0 else 0.0 # Nyquist phase is either 0/pi or doesn't exist for odd N
    Y[1:-1] = mag[1:-1] * np.exp(1j*new_ph)
    y = irfft(Y, n=len(x))
    return y

def psd_at_targets(x):
    # Use detrend=constant explicitly for consistency with null/Welch in your logic
    f, Pxx = welch(x, fs=1.0, window='hann',
                   nperseg=min(4096, len(x)), noverlap=None, detrend='constant')
    if Pxx.size: Pxx[0] = 0.0
    targets = list(TARGET_MODES.values())
    idxs = [np.argmin(np.abs(f - ft)) for ft in targets]
    return f, Pxx, np.array(Pxx[idxs], dtype=float), np.array(idxs, dtype=int)

# ---------------------------
# OTHER ANALYSIS HELPERS
# ---------------------------
# From Cell 1
def psd_many(x):
    outs = []
    # Periodogram with different windows
    for w in ["hann","hamming","blackman"]:
        f, P = periodogram(x, fs=1.0, window=get_window(w, len(x)))
        P[0]=0; outs.append((f,P,f"periodogram-{w}"))
    # Welch with different nperseg
    for nseg in [1024, 2048, 4096, 8192]:
        f, P = welch(x, fs=1.0, window="hann", nperseg=min(nseg,len(x)))
        P[0]=0; outs.append((f,P,f"welch-{nseg}"))
    return outs

# From Cell 4
def delta_alpha_variant(p_list):
    # swap ratio: arctan(p_{n+1}/p_n) instead of p_n/p_{n+1}
    alpha = np.arctan(p_list[1:] / p_list[:-1])
    return np.diff(alpha)

# From Cell 5
def band_power(f, Pxx, f0, hw):
    m = (f >= (f0 - hw)) & (f <= (f0 + hw))
    return trapezoid(Pxx[m], f[m]) if np.any(m) else np.nan

# From Cell 8
def acf(x, L=4000): # Changed L to 4000 to match user's plot request
    x = (x - x.mean())
    c = np.correlate(x, x, mode='full')[len(x)-1:len(x)-1+L]
    return c / c[0]

# From Cell 9
def conditioned_psd(primes, nmax, cls):
    end = np.searchsorted(primes, nmax, 'right')
    p = primes[:end]
    nxt = p[1:]
    mask = (nxt % 6 == cls)
    sel = np.where(mask)[0]
    # build Δα only where both pairs exist
    idx = sel[(sel < len(p)-2)]
    # Re-calculate alpha and diff based on masked indices
    alpha = np.arctan(p[idx] / p[idx+1])
    da = np.diff(alpha)
    return psd_for(da)

# From Cell 10
def notch_fir(f0, bw=0.01, numtaps=2001, fs=1.0):
    f1, f2 = max(0, f0-bw/2), min(0.5, f0+bw/2)
    # keep everything except [f1,f2]
    return firwin(numtaps, [f1, f2], pass_zero='bandstop', fs=fs)


# ---------------------------
# INITIAL DATA GENERATION (From prime_wave_modes.py)
# ---------------------------
print("--- Phase 1: Data Initialization ---")
MAX_PRIME_VALUE = max(SAMPLE_LIMITS)
print(f"Starting analysis for primes up to {MAX_PRIME_VALUE:,} ...")
primes = get_primes_up_to_n(MAX_PRIME_VALUE)
print(f"Primes found: {len(primes):,}.")

palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i, limit in enumerate(SAMPLE_LIMITS):
    end_idx = np.searchsorted(primes, limit, side="right")
    p_list = primes[:end_idx]
    da = calculate_delta_alpha(p_list)
    samples.append({
        "limit": limit,
        "count": len(p_list),
        "da": da,
        "color": palette[i % len(palette)],
        "label": rf"Primes $\leq$ {limit:,} ($N={len(p_list):,}$)"
    })
    print(f"Sample {i+1}/{len(SAMPLE_LIMITS)} up to {limit:,}: {len(da):,} Δα values.")

# The plotting functions from the original script (Figure 2 and 3)
def plot_global_drift_decay(samples_data):
    plt.figure(figsize=(10, 6))
    max_abs_avg = 0.0
    for s in samples_data:
        da = s["da"]
        csum = np.cumsum(da)
        n = np.arange(1, da.size + 1, dtype=float)
        ravg = csum / n
        max_abs_avg = max(max_abs_avg, np.max(np.abs(ravg)))
        plt.plot(n, ravg, color=s["color"], label=s["label"], linewidth=1.2)
    plt.axhline(0.0, color="red", linestyle="--", alpha=0.6, label="Asymptotic Ideal (Zero Drift)")
    plt.title("Figure 2: Global Determination — Decay of Systematic Bias (Global Drift)")
    plt.xlabel("Prime Index ($n$)")
    plt.ylabel(r"Global Drift (Running Average of $\Delta\alpha$)")
    ylim = max(1e-12, max_abs_avg) * 1.1
    plt.ylim(-0.1*ylim, ylim)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9)
    fn = "figure_2_global_drift_decay_3samples.png"
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    print(f"Generated Figure 2: '{fn}'")

def plot_fixed_wave_modes(samples_data):
    base = samples_data[-1]
    da_base = base["da"]
    f_base, Pxx_base = psd_for(da_base)
    psds = []
    max_visible = 0.0
    fmin, fmax = 0.10, 0.55
    i0 = np.searchsorted(f_base, fmin); i1 = np.searchsorted(f_base, fmax)
    for s in samples_data:
        f, Pxx = psd_for(s["da"])
        psds.append((s, f, Pxx))
        mask = (f >= fmin) & (f <= fmax)
        if mask.any(): max_visible = max(max_visible, float(Pxx[mask].max()))

    identified = []
    for label, target_f in TARGET_MODES.items():
        idx = int(np.argmin(np.abs(f_base - target_f)))
        found_f = float(f_base[idx])
        found_mag = float(Pxx_base[idx])
        identified.append((label, target_f, found_f, found_mag))

    plt.figure(figsize=(12, 7))
    for s, f, Pxx in psds:
        alpha = 1.0 if s is base else 0.5; lw = 1.6 if s is base else 1.0
        plt.plot(f, Pxx, color=s["color"], alpha=alpha, linewidth=lw, label=s["label"])

    y_max = max_visible * 1.25 if max_visible > 0 else (Pxx_base[1:].max() * 1.1)
    plt.xlim(fmin, fmax); plt.ylim(0, y_max)
    peak_colors = ["#d62728", "#9467bd", "#8c564b"]
    y_text = y_max * 0.55
    for (pc, (label, target_f, found_f, found_mag)) in zip(peak_colors, identified):
        plt.axvline(found_f, color=pc, linestyle="--", alpha=0.6, lw=1.0)
        period_txt = ""
        if "(T=" in label:
            head, tail = label.split("(T="); period_txt = "(T=" + tail
            mode_name = head.strip()
        else: mode_name = label
        text = f"{mode_name}\n$f={found_f:.5f}$\n{period_txt}".strip()
        plt.annotate(
            text, xy=(found_f, min(found_mag * 1.05, y_max * 0.98)),
            xytext=(found_f, y_text), textcoords="data", ha="center", fontsize=9, color=pc,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.0", color=pc, lw=1.4))

    plt.title(f"Figure 3: Local Coherence — Invariant Spectral Modes (Overlay of {len(samples_data)} Samples)")
    plt.xlabel("Frequency ($f$)"); plt.ylabel("Power Spectral Density (PSD)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    fn = "figure_3_fixed_wave_modes_3samples.png"
    plt.savefig(fn, dpi=150)
    print(f"Generated Figure 3: '{fn}'")
    print("\n--- Mode Check (base sample) ---")
    print(f"{'Mode':28s} {'target f':>10s} {'found f':>10s} {'Δf':>10s}")
    for (label, target_f, found_f, _) in identified:
        print(f"{label:28s} {target_f:10.5f} {found_f:10.5f} {found_f - target_f:10.5f}")

# Execute original plots
plot_global_drift_decay(samples)
plot_fixed_wave_modes(samples)

# ---------------------------
# ANALYSIS CELLS EXECUTION
# ---------------------------
targets = list(TARGET_MODES.values())
labels = list(TARGET_MODES.keys())
da = samples[-1]['da']

print("\n--- Phase 2: Estimator/Window Robustness (Cell 1 & 2) ---")
# Cell 1: Estimator/window robustness
outs = psd_many(da)
plt.figure(figsize=(12,6))
for f,P,label in outs:
    plt.plot(f, P, alpha=0.5, linewidth=1, label=label)
peak_colors = ['#d62728','#9467bd','#8c564b']
for col, ft in zip(peak_colors, TARGET_MODES.values()):
    plt.axvline(ft, color=col, linestyle='--', alpha=0.6)
plt.xlim(0.10,0.55)
# Set Y-limit based on a high percentile of the last Welch PSD
Pxx_lim = outs[-1][1][(outs[-1][0]>=0.10)&(outs[-1][0]<=0.55)]
ylim_max = np.percentile(Pxx_lim, 99.9)*1.2 if Pxx_lim.size > 0 else None
plt.ylim(0, ylim_max)
plt.title("Estimator/window robustness")
plt.xlabel("Frequency (f)"); plt.ylabel("PSD")
plt.grid(True, linestyle=':', alpha=0.6); plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("figure_estimator_robustness.png", dpi=150)
print("Saved 'figure_estimator_robustness.png'")

# Cell 2: Zoom each target ±0.02
f, Pxx = psd_for(da)
for lab, ft, col in zip(labels, targets, peak_colors):
    w = 0.02
    m = (f >= ft-w) & (f <= ft+w)
    plt.figure(figsize=(7,4))
    plt.semilogy(f[m], Pxx[m], color=col)
    plt.axvline(ft, color=col, linestyle='--')
    plt.title(f"{lab} neighborhood (f≈{ft:.5f})")
    plt.xlabel("f"); plt.ylabel("PSD (log scale)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"figure_zoom_f{ft:.5f}.png", dpi=150)
    print(f"Saved 'figure_zoom_f{ft:.5f}.png'")

print("\n--- Phase 3: Phase-Randomized Surrogate Test (Cell 3) ---")
# Cell 3: Phase-randomized surrogate test
rng = np.random.default_rng(RANDOM_SEED)
f, Pxx, obs_bins, idxs = psd_at_targets(da)
M = 200  # number of surrogates
null_bins = np.zeros((M, len(targets)))
for m in range(M):
    y = phase_randomize(da, rng)
    _, _, vals, _ = psd_at_targets(y)
    null_bins[m,:] = vals

mu = null_bins.mean(axis=0)
sd = null_bins.std(axis=0, ddof=1)
z  = (obs_bins - mu) / np.where(sd>0, sd, np.inf)

print("\n--- Phase-randomized surrogate test results ---")
print(f"{'Mode':28s} {'f_target':>9s} {'obs':>12s} {'μ_null':>12s} {'σ_null':>10s} {'z':>8s}")
for lab, ft, z_k, obs_k, mu_k, sd_k in zip(labels, targets, z, obs_bins, mu, sd):
    print(f"{lab:28s} f={ft:.5f}  obs={obs_k:12.5e}  μ={mu_k:12.5e}  σ={sd_k:10.5e}  z={z_k:8.2f}")

# Simple overlay visualization
plt.figure(figsize=(12,6))
rng = np.random.default_rng(RANDOM_SEED) # Reset RNG for plot
for m in range(min(30,M)):
    y = phase_randomize(da, rng)
    f_s, Pxx_s = welch(y, fs=1.0, window='hann',
                       nperseg=min(4096, len(y)), noverlap=None, detrend='constant')
    if Pxx_s.size: Pxx_s[0] = 0.0
    plt.plot(f_s, Pxx_s, color='#bbbbbb', alpha=0.25, linewidth=0.8)
plt.plot(f, Pxx, color='#1f77b4', linewidth=1.8, label='Real Δα PSD')
for col, ft in zip(peak_colors, targets):
    plt.axvline(ft, color=col, linestyle='--', alpha=0.7)
plt.xlim(0.10, 0.55)
Pxx_lim = Pxx[(f>=0.10)&(f<=0.55)]
plt.ylim(0, np.percentile(Pxx_lim, 99.9)*1.2 if Pxx_lim.size > 0 else None)
plt.title("Phase-randomized surrogate overlay (gray) vs real (blue)")
plt.xlabel("Frequency (f)"); plt.ylabel("PSD")
plt.grid(True, linestyle=':', alpha=0.6); plt.legend()
plt.tight_layout()
plt.savefig('figure_phase_surrogates.png', dpi=150)
print("Saved 'figure_phase_surrogates.png'")


print("\n--- Phase 4: Alternate Mappings & Permutation Nulls (Cell 4 & 6) ---")
# Cell 4: Alternate ∆α mapping + null test
end = np.searchsorted(primes, 2_000_000, side='right')
da_alt = delta_alpha_variant(primes[:end])
print("\nRunning null test on alternate Δα mapping (arctan(p_n+1 / p_n))")
run_null_test(da_alt, save_fig=False) # Skip saving the figure to avoid overwriting

# Cell 6: Run null test on original Δα
print("\nRunning null test on original Δα mapping (arctan(p_n / p_n+1))")
run_null_test(da) # This saves 'figure_null_overlay.png'

print("\n--- Phase 5: Suppression Index vs Sample Size (Cell 5) ---")
# Cell 5: Band-suppression vs sample size
band_hw = 0.003
sizes = [20000, 200000, 500000, 1_000_000, 2_000_000]

suppress_idx = {lab: [] for lab in labels}
norm_idx     = {lab: [] for lab in labels}

for nmax in sizes:
    end = np.searchsorted(primes, nmax, side='right')
    da_nmax = calculate_delta_alpha(primes[:end])
    f, Pxx = psd_for(da_nmax)

    ref_mask = (f >= 0.10) & (f <= 0.55)
    ref_med  = np.median(Pxx[ref_mask]) if np.any(ref_mask) else np.nan

    for lab, f0 in zip(labels, targets):
        bp = band_power(f, Pxx, f0, band_hw)
        suppress_idx[lab].append(bp)
        norm_idx[lab].append(bp / ref_med if ref_med > 0 else np.nan)

# Plot normalized suppression
plt.figure(figsize=(9,5))
for lab, col in zip(labels, peak_colors):
    plt.plot(sizes, norm_idx[lab], marker='o', label=lab, color=col)
plt.xscale('log')
plt.xlabel('Max prime value (log scale)')
plt.ylabel('Band power / median power (0.10–0.55)')
plt.title('Suppression index vs. sample size (lower = stronger suppression)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('figure_suppression_vs_size.png', dpi=150)
print("Saved 'figure_suppression_vs_size.png'")

print("\n--- Phase 6: Autocorrelation (Cell 8) ---")
# Cell 8: Autocorrelation
lags = 4000
ac = acf(da, L=lags)
plt.figure(figsize=(9,4)); plt.plot(ac); plt.title(r"$\Delta\alpha$ autocorrelation"); plt.xlabel("Lag"); plt.grid(True,linestyle=':',alpha=0.6)
plt.tight_layout()
plt.savefig("figure_autocorrelation.png", dpi=150)
print("Saved 'figure_autocorrelation.png'")

print("\n--- Phase 7: Conditional PSD (Cell 9) ---")
# Cell 9: Conditioned by next-prime mod-6 class
plt.figure(figsize=(12,5))
for cls, name, col in [(1,'6k+1','#1f77b4'), (5,'6k-1','#ff7f0e')]:
    # We use the full prime list up to 2M for this
    f_c, P_c = conditioned_psd(primes, 2_000_000, cls)
    plt.plot(f_c, P_c, label=name, color=col, alpha=0.8)
for col, ft in zip(peak_colors, TARGET_MODES.values()):
    plt.axvline(ft, color=col, linestyle='--', alpha=0.6)
plt.xlim(0.10,0.55);
Pxx_lim = P_c[(f_c>=0.10)&(f_c<=0.55)]
plt.ylim(0, np.percentile(Pxx_lim, 99.9)*1.2 if Pxx_lim.size > 0 else None)
plt.title(r"$\Delta\alpha$ PSD conditioned on next-prime mod-6 class")
plt.xlabel("Frequency (f)"); plt.ylabel("PSD")
plt.grid(True, linestyle=':', alpha=0.6); plt.legend()
plt.tight_layout()
plt.savefig("figure_conditional_psd.png", dpi=150)
print("Saved 'figure_conditional_psd.png'")


print("\n--- Phase 8: Notch Filter Test (Cell 10) ---")
# Cell 10: Quick FIR bandstop at each target, then PSD of residual
x = da.astype(float)
y = x.copy()
for f0 in TARGET_MODES.values():
    b = notch_fir(f0, bw=0.01, numtaps=2001)
    # The filter must be short enough relative to the signal to avoid boundary artifacts
    y = filtfilt(b, [1.0], y)

f_x, P_x = psd_for(x); f_y, P_y = psd_for(y)
plt.figure(figsize=(12,5))
plt.plot(f_x, P_x, label='original', alpha=0.5)
plt.plot(f_y, P_y, label='notch-filtered', linewidth=1.6)
for col, ft in zip(peak_colors, TARGET_MODES.values()):
    plt.axvline(ft, color=col, linestyle='--', alpha=0.6)
plt.xlim(0.10,0.55)
plt.ylim(0, np.percentile(P_x[(f_x>=0.10)&(f_x<=0.55)], 99.9)*1.2)
plt.title("PSD before/after removing the three notches")
plt.xlabel("Frequency (f)"); plt.ylabel("PSD")
plt.grid(True, linestyle=':', alpha=0.6); plt.legend(); plt.tight_layout()
plt.savefig("figure_notch_filter_psd.png", dpi=150)
print("Saved 'figure_notch_filter_psd.png'")