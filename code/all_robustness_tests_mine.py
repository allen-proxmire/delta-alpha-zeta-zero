From scipy.signal import get_window, welch, periodogram

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



outs = psd_many(samples[-1]['da'])

plt.figure(figsize=(12,6))

for f,P,label in outs:

plt.plot(f, P, alpha=0.5, linewidth=1, label=label)

for col, ft in zip(['#d62728','#9467bd','#8c564b'], TARGET_MODES.values()):

plt.axvline(ft, color=col, linestyle='--', alpha=0.6)

plt.xlim(0.10,0.55)

plt.ylim(0, np.percentile(outs[-1][1][(outs[-1][0]>=0.10)&(outs[-1][0]<=0.55)], 99.9)*1.2)

plt.title("Estimator/window robustness")

plt.grid(True, linestyle=':', alpha=0.6); plt.legend(ncol=2, fontsize=8)

plt.tight_layout(); plt.show()_____# Zoom each target ±0.02 and show notch depth

f, Pxx = psd_for(samples[-1]['da'])

for lab, ft, col in zip(TARGET_MODES.keys(), TARGET_MODES.values(),

['#d62728','#9467bd','#8c564b']):

w = 0.02

m = (f >= ft-w) & (f <= ft+w)

plt.figure(figsize=(7,4))

plt.semilogy(f[m], Pxx[m], color=col)

plt.axvline(ft, color=col, linestyle='--')

plt.title(f"{lab} neighborhood (f≈{ft:.5f})")

plt.xlabel("f"); plt.ylabel("PSD (log scale)")

plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

plt.show()_____# === Cell C: Phase-randomized surrogate test at target bins ===

from numpy.fft import rfft, irfft, rfftfreq



def phase_randomize(x, rng):

X = rfft(x)

# keep DC and Nyquist; randomize interior phases

mag = np.abs(X)

ph = np.angle(X)

k = np.arange(1, len(X)-1)

new_ph = rng.uniform(0, 2*np.pi, size=k.size)

Y = np.zeros_like(X, dtype=complex)

Y[0] = X[0]

if len(X) > 1:

Y[-1] = X[-1]

Y[1:-1] = mag[1:-1] * np.exp(1j*new_ph)

y = irfft(Y, n=len(x))

return y



def psd_at_targets(x):

f, Pxx = welch(x, fs=1.0, window='hann',

nperseg=min(4096, len(x)), noverlap=None, detrend='constant')

if Pxx.size: Pxx[0] = 0.0

idxs = [np.argmin(np.abs(f - ft)) for ft in targets]

return f, Pxx, np.array(Pxx[idxs], dtype=float), np.array(idxs, dtype=int)



rng = np.random.default_rng(2025)

f, Pxx, obs_bins, idxs = psd_at_targets(samples[-1]['da'])



M = 200 # number of surrogates

null_bins = np.zeros((M, len(targets)))

for m in range(M):

y = phase_randomize(samples[-1]['da'], rng)

_, Pxx_s, vals, _ = psd_at_targets(y)

null_bins[m,:] = vals



mu = null_bins.mean(axis=0)

sd = null_bins.std(axis=0, ddof=1)

z = (obs_bins - mu) / np.where(sd>0, sd, np.inf)



print("\n--- Phase-randomized surrogate test ---")

for lab, ft, z_k, obs_k, mu_k, sd_k in zip(labels, targets, z, obs_bins, mu, sd):

print(f"{lab:28s} f={ft:.5f} obs={obs_k:.3e} μ={mu_k:.3e} σ={sd_k:.3e} z={z_k:.2f}")



# Simple overlay visualization

plt.figure(figsize=(12,6))

for m in range(min(30,M)):

y = phase_randomize(samples[-1]['da'], rng)

f_s, Pxx_s = welch(y, fs=1.0, window='hann',

nperseg=min(4096, len(y)), noverlap=None, detrend='constant')

if Pxx_s.size: Pxx_s[0] = 0.0

plt.plot(f_s, Pxx_s, color='#bbbbbb', alpha=0.25, linewidth=0.8)

plt.plot(f, Pxx, color='#1f77b4', linewidth=1.8, label='Real Δα PSD')

for col, ft in zip(['#d62728','#9467bd','#8c564b'], targets):

plt.axvline(ft, color=col, linestyle='--', alpha=0.7)

plt.xlim(0.10, 0.55)

plt.ylim(0, np.percentile(Pxx[(f>=0.10)&(f<=0.55)], 99.9)*1.2)

plt.title("Phase-randomized surrogate overlay (gray) vs real (blue)")

plt.xlabel("Frequency (f)"); plt.ylabel("PSD")

plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

plt.tight_layout()

plt.savefig('figure_phase_surrogates.png', dpi=150)

print("Saved 'figure_phase_surrogates.png'")____# === Cell B: Alternate ∆α mapping + null test ===

def delta_alpha_variant(p_list):

# swap ratio: arctan(p_{n+1}/p_n) instead of p_n/p_{n+1}

alpha = np.arctan(p_list[1:] / p_list[:-1])

return np.diff(alpha)



end = np.searchsorted(primes, 2_000_000, side='right')

da_alt = delta_alpha_variant(primes[:end])



# Reuse the null test helper you already defined:

run_null_test(da_alt)____# === Cell A: Band-suppression vs sample size ===

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import welch



targets = list(TARGET_MODES.values()) # [0.47871, 0.38895, 0.35153]

labels = list(TARGET_MODES.keys())

band_hw = 0.003 # half-width of band around each target

sizes = [20000, 200000, 500000, 1_000_000, 2_000_000] # tweak if needed



def psd(signal):

f, Pxx = welch(signal, fs=1.0, window='hann',

nperseg=min(4096, len(signal)), noverlap=None, detrend='constant')

if Pxx.size: Pxx[0] = 0.0

return f, Pxx



from scipy.integrate import trapezoid

# ...

def band_power(f, Pxx, f0, hw):

m = (f >= (f0 - hw)) & (f <= (f0 + hw))

return trapezoid(Pxx[m], f[m]) if np.any(m) else np.nan



# reuse your primes array if it covers max(sizes); else re-sieve

if primes[-1] < max(sizes):

primes = get_primes_up_to_n(max(sizes))



suppress_idx = {lab: [] for lab in labels} # absolute band power

norm_idx = {lab: [] for lab in labels} # band power / median power in [0.1,0.55]



for nmax in sizes:

end = np.searchsorted(primes, nmax, side='right')

da = calculate_delta_alpha(primes[:end])

f, Pxx = psd(da)



ref_mask = (f >= 0.10) & (f <= 0.55)

ref_med = np.median(Pxx[ref_mask]) if np.any(ref_mask) else np.nan



for lab, f0 in zip(labels, targets):

bp = band_power(f, Pxx, f0, band_hw)

suppress_idx[lab].append(bp)

norm_idx[lab].append(bp / ref_med if ref_med>0 else np.nan)



# Plot normalized suppression

plt.figure(figsize=(9,5))

for lab, col in zip(labels, ['#d62728','#9467bd','#8c564b']):

plt.plot(sizes, norm_idx[lab], marker='o', label=lab, color=col)

plt.xscale('log')

plt.xlabel('Max prime value (log scale)')

plt.ylabel('Band power / median power (0.10–0.55)')

plt.title('Suppression index vs. sample size (lower = stronger suppression)')

plt.grid(True, linestyle=':', alpha=0.6)

plt.legend()

plt.tight_layout()

plt.savefig('figure_suppression_vs_size.png', dpi=150)

print("Saved 'figure_suppression_vs_size.png'")_____run_null_test(samples[-1]["da"])_____from scipy.signal import welch



USE_WELCH = True # set False to use periodogram like before

N_PERM = 200 # number of shuffles for the null test

F_BAND = (0.10, 0.55) # visible band for plotting / scaling

WELCH_NPERSEG = 4096 # Welch segment length; tweak if RAM-limited



def psd_for(signal: np.ndarray):

if USE_WELCH:

f, Pxx = welch(signal, fs=1.0, window='hann',

nperseg=min(WELCH_NPERSEG, len(signal)),

noverlap=None, detrend='constant', return_onesided=True)

else:

f, Pxx = periodogram(signal, fs=1.0, window='hann')

if Pxx.size:

Pxx[0] = 0.0

return f, Pxx



def nearest_bin(fgrid: np.ndarray, f0: float) -> int:

return int(np.argmin(np.abs(fgrid - f0)))



def run_null_test(da_signal: np.ndarray):

"""Permutation null: shuffle Δα, compute PSD, evaluate target bins.

Prints z-scores and p-values; optionally saves a null overlay figure."""

# PSD of the real series

f, Pxx = psd_for(da_signal)



# Prepare storage for null power at each target bin

target_labels = list(TARGET_MODES.keys())

target_freqs = np.array(list(TARGET_MODES.values()), dtype=float)

target_idx = np.array([nearest_bin(f, tf) for tf in target_freqs], dtype=int)



null_vals = np.zeros((N_PERM, len(target_idx)), dtype=float)



rng = np.random.default_rng(1337)

for b in range(N_PERM):

shuffled = da_signal.copy()

rng.shuffle(shuffled)

_, Pxx_null = psd_for(shuffled)

null_vals[b, :] = Pxx_null[target_idx]



mu = null_vals.mean(axis=0)

sd = null_vals.std(axis=0, ddof=1)

obs = Pxx[target_idx]

z = (obs - mu) / np.where(sd > 0, sd, np.inf)

# two-sided p from z

from scipy.stats import norm

p = 2.0 * (1.0 - norm.cdf(np.abs(z)))



print("\n--- Permutation Null (Δα shuffled) ---")

print(f"{'Mode':28s} {'f_target':>9s} {'obs':>12s} {'μ_null':>12s} {'σ_null':>10s} {'z':>8s} {'p':>10s}")

for k, label in enumerate(target_labels):

print(f"{label:28s} {target_freqs[k]:9.5f} {obs[k]:12.5e} {mu[k]:12.5e} {sd[k]:10.5e} {z[k]:8.2f} {p[k]:10.2e}")



# Optional: overlay null PSDs

fmin, fmax = F_BAND

plt.figure(figsize=(12, 7))

# plot a handful (e.g., 30) null spectra for appearance

show_m = min(30, N_PERM)

for b in range(show_m):

# regenerate quickly to avoid storing all

shuffled = da_signal.copy(); rng.shuffle(shuffled)

f_null, Pxx_null = psd_for(shuffled)

plt.plot(f_null, Pxx_null, color='#bbbbbb', alpha=0.25, linewidth=0.8)

# real PSD on top

plt.plot(f, Pxx, color='#1f77b4', linewidth=1.8, label='Real Δα PSD')

for col, (lab, tf) in zip(['#d62728', '#9467bd', '#8c564b'], TARGET_MODES.items()):

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

print("Saved 'figure_null_overlay.png'")________#!/usr/bin/env python3

# prime_wave_modes.py

import numpy as np

import math

from scipy.signal import periodogram, get_window

from time import time

import sys

import matplotlib.pyplot as plt

from datetime import datetime



# ---------------------------

# CONFIG

# ---------------------------

# How far to sieve primes for each sample overlay:

SAMPLE_LIMITS = [20_000, 200_000, 2_000_000]



# Hypothesized fixed-mode frequencies (cycles/sample) and labels

TARGET_MODES = {

"Fast Mode (T=2.089)": 0.47871,

"Mid Mode (T=2.571)": 0.38895,

"Slow Mode (T=2.845)": 0.35153,

}



# FFT window to use (scipy.signal.get_window name)

FFT_WINDOW = "hann" # same as your original

RANDOM_SEED = 1337 # for any future null tests you might add



# ---------------------------

# PRIMES & Δα

# ---------------------------

def get_primes_up_to_n(n: int) -> np.ndarray:

"""Vectorized Sieve of Eratosthenes; returns primes ≤ n as a NumPy array."""

if n < 2:

return np.array([], dtype=np.int64)

is_prime = np.ones(n + 1, dtype=bool)

is_prime[:2] = False

limit = int(np.sqrt(n))

for p in range(2, limit + 1):

if is_prime[p]:

is_prime[p*p:n+1:p] = False

return np.flatnonzero(is_prime).astype(np.int64)



def calculate_delta_alpha(p_list: np.ndarray) -> np.ndarray:

"""

∆α = α(n+1) - α(n), with α(n) = arctan(p_n / p_{n+1}) as in your scripts.

Returns length len(p_list)-2.

"""

# α_n defined on adjacent prime pairs

alpha = np.arctan(p_list[:-1] / p_list[1:])

# first difference of α

return np.diff(alpha)



# ---------------------------

# BUILD SAMPLES

# ---------------------------

MAX_PRIME_VALUE = max(SAMPLE_LIMITS)

print(f"Starting analysis for primes up to {MAX_PRIME_VALUE:,} ...")

t0 = time()

primes = get_primes_up_to_n(MAX_PRIME_VALUE)

print(f"Primes found: {len(primes):,}. Time: {time() - t0:.2f}s.")



palette = ["#1f77b4", "#ff7f0e", "#2ca02c"] # blue, orange, green

samples = []



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



# ---------------------------

# FIGURE 2 — GLOBAL DRIFT

# ---------------------------

def plot_global_drift_decay(samples_data):

plt.figure(figsize=(10, 6))

max_abs_avg = 0.0



for s in samples_data:

da = s["da"]

# running mean of ∆α

csum = np.cumsum(da)

n = np.arange(1, da.size + 1, dtype=float)

ravg = csum / n

max_abs_avg = max(max_abs_avg, np.max(np.abs(ravg)))

plt.plot(n, ravg, color=s["color"], label=s["label"], linewidth=1.2)



plt.axhline(0.0, color="red", linestyle="--", alpha=0.6, label="Asymptotic Ideal (Zero Drift)")

plt.title("Figure 2: Global Determination — Decay of Systematic Bias (Global Drift)")

plt.xlabel("Prime Index ($n$)")

plt.ylabel("Global Drift (Running Average of $\\Delta\\alpha$)")

ylim = max(1e-12, max_abs_avg) * 1.1

plt.ylim(-0.1*ylim, ylim)

plt.grid(True, linestyle=":", alpha=0.6)

plt.legend(fontsize=9)

fn = "figure_2_global_drift_decay_3samples.png"

plt.tight_layout()

plt.savefig(fn, dpi=150)

print(f"Generated Figure 2: '{fn}'")



# ---------------------------

# FIGURE 3 — FIXED MODES

# ---------------------------

def psd_for(signal: np.ndarray):

"""

One-shot PSD with DC removed. Returns (freq, Pxx).

fs=1.0 since the sample index is the natural coordinate in your mapping.

"""

f, Pxx = periodogram(signal, fs=1.0, window=get_window(FFT_WINDOW, signal.size, fftbins=True))

if Pxx.size:

Pxx[0] = 0.0 # kill DC explicitly so scaling isn't dominated by it

return f, Pxx



def plot_fixed_wave_modes(samples_data):

# base sample = largest (last in list)

base = samples_data[-1]

da_base = base["da"]

f_base, Pxx_base = psd_for(da_base)



# Precompute PSDs for all samples once (avoid recomputation)

psds = []

max_visible = 0.0

fmin, fmax = 0.10, 0.55 # your viewing window

i0 = np.searchsorted(f_base, fmin)

i1 = np.searchsorted(f_base, fmax)



for s in samples_data:

f, Pxx = psd_for(s["da"])

psds.append((s, f, Pxx))

# track visible max (non-DC) within window:

# make sure frequency bins align—if not, compute local max via mask

mask = (f >= fmin) & (f <= fmax)

if mask.any():

max_visible = max(max_visible, float(Pxx[mask].max()))



# Identify nearest-bin “peaks” to each target using the base PSD

identified = []

for label, target_f in TARGET_MODES.items():

idx = int(np.argmin(np.abs(f_base - target_f)))

found_f = float(f_base[idx])

found_mag = float(Pxx_base[idx])

identified.append((label, target_f, found_f, found_mag))



# Plot

plt.figure(figsize=(12, 7))

for s, f, Pxx in psds:

alpha = 1.0 if s is base else 0.5

lw = 1.6 if s is base else 1.0

plt.plot(f, Pxx, color=s["color"], alpha=alpha, linewidth=lw, label=s["label"])



# Nice y-limits in visible band

y_max = max_visible * 1.25 if max_visible > 0 else (Pxx_base[1:].max() * 1.1)

plt.xlim(fmin, fmax)

plt.ylim(0, y_max)



# Annotate the three modes

peak_colors = ["#d62728", "#9467bd", "#8c564b"] # red, purple, brown

y_text = y_max * 0.55

for (pc, (label, target_f, found_f, found_mag)) in zip(peak_colors, identified):

# vertical guide

plt.axvline(found_f, color=pc, linestyle="--", alpha=0.6, lw=1.0)

# arrowed annotation from mid-height to near the peak

period_txt = ""

# if label contains (T=...), keep it for readability

if "(T=" in label:

head, tail = label.split("(T=")

period_txt = "(T=" + tail

mode_name = head.strip()

else:

mode_name = label



text = f"{mode_name}\n$f={found_f:.5f}$\n{period_txt}".strip()

plt.annotate(

text,

xy=(found_f, min(found_mag * 1.05, y_max * 0.98)),

xytext=(found_f, y_text),

textcoords="data",

ha="center",

fontsize=9,

color=pc,

arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.0", color=pc, lw=1.4)

)



plt.title(f"Figure 3: Local Coherence — Invariant Spectral Modes (Overlay of {len(samples_data)} Samples)")

plt.xlabel("Frequency ($f$)")

plt.ylabel("Power Spectral Density (PSD)")

plt.grid(True, linestyle=":", alpha=0.6)

plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

plt.tight_layout()

fn = "figure_3_fixed_wave_modes_3samples.png"

plt.savefig(fn, dpi=150)

print(f"Generated Figure 3: '{fn}'")



# Console report: target vs found

print("\n--- Mode Check (base sample) ---")

print(f"{'Mode':28s} {'target f':>10s} {'found f':>10s} {'Δf':>10s}")

for (label, target_f, found_f, _) in identified:

print(f"{label:28s} {target_f:10.5f} {found_f:10.5f} {found_f - target_f:10.5f}")



# ---------------------------

# MAIN

# ---------------------------

if __name__ == "__main__":

np.random.seed(RANDOM_SEED)



plot_global_drift_decay(samples)

plot_fixed_wave_modes(samples)

plt.show()____ # (a) Autocorrelation around lag region set by f (lag ≈ 1/f in samples)

def acf(x, L=2000):

x = (x - x.mean())

c = np.correlate(x, x, mode='full')[len(x)-1:len(x)-1+L]

return c / c[0]

da = samples[-1]['da']

lags = 4000

ac = acf(da, L=lags)

plt.figure(figsize=(9,4)); plt.plot(ac); plt.title("Δα autocorrelation"); plt.xlabel("lag"); plt.grid(True,linestyle=':',alpha=0.6)

plt.tight_layout(); plt.show()______# (b) Condition by next-prime mod-6 class

def conditioned_psd(primes, nmax, cls):

end = np.searchsorted(primes, nmax, 'right')

p = primes[:end]

nxt = p[1:]

mask = (nxt % 6 == cls)

sel = np.where(mask)[0]

# build Δα only where both pairs exist

idx = sel[(sel < len(p)-2)]

alpha = np.arctan(p[:-1] / p[1:])

da = np.diff(alpha)[idx]

return psd_for(da)



for cls, name, col in [(1,'6k+1','#1f77b4'), (5,'6k-1','#ff7f0e')]:

f_c, P_c = conditioned_psd(primes, 2_000_000, cls)

plt.plot(f_c, P_c, label=name, color=col, alpha=0.8)

for col, ft in zip(['#d62728','#9467bd','#8c564b'], TARGET_MODES.values()):

plt.axvline(ft, color=col, linestyle='--', alpha=0.6)

plt.xlim(0.10,0.55); plt.ylim(0, np.percentile(P_c[(f_c>=0.10)&(f_c<=0.55)], 99.9)*1.2)

plt.title("Δα PSD conditioned on next-prime mod-6 class")

plt.grid(True, linestyle=':', alpha=0.6); plt.legend()

plt.tight_layout(); plt.show()_______# Quick FIR bandstop at each target, then PSD of residual

from scipy.signal import firwin, filtfilt



fs = 1.0

def notch_fir(f0, bw=0.01, numtaps=1001):

f1, f2 = max(0, f0-bw/2), min(0.5, f0+bw/2)

# keep everything except [f1,f2]

bands = [0, f1, f2, 0.5]

desired = [1, 0, 1]

# use firwin2-like via remez? Keep it simple: cascade two narrow bandstops

return firwin(numtaps, [f1, f2], pass_zero='bandstop', fs=fs)



x = samples[-1]['da'].astype(float)

y = x.copy()

for f0 in TARGET_MODES.values():

b = notch_fir(f0, bw=0.01, numtaps=2001)

y = filtfilt(b, [1.0], y)



f_x, P_x = psd_for(x); f_y, P_y = psd_for(y)

plt.figure(figsize=(12,5))

plt.plot(f_x, P_x, label='original', alpha=0.5)

plt.plot(f_y, P_y, label='notch-filtered', linewidth=1.6)

plt.xlim(0.10,0.55)

plt.title("PSD before/after removing the three notches")

plt.grid(True, linestyle=':', alpha=0.6); plt.legend(); plt.tight_layout(); plt.show()