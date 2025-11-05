Synthetic gap control tests for Prime-Wave project.



Generates four synthetic sequences:

- Uniform random gaps

- Poisson (exponential) gaps

- Random odd sequence (same count as real primes)

- Gaussian-perturbed real primes



For each sequence:

- compute Δα = diff(arctan(p[:-1] / p[1:]))

- detrend Δα

- compute Welch PSD (nperseg=2048, hann)

- extract peak power near target frequencies

- save overlay plots and a CSV summary



Outputs saved to ./synthetic_outputs/

"""

import os

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import welch, detrend

import pandas as pd

from math import ceil



# ---------------- CONFIG ----------------

MAX_PRIME_VAL = 2_000_000

TARGET_FREQS = [0.35153, 0.38895, 0.47871]

NPERSEG = 2048

NOVERLAP = NPERSEG // 2

OUTDIR = "synthetic_outputs"

os.makedirs(OUTDIR, exist_ok=True)



# ---------------- prime generation (sieve) ----------------

def get_primes_up_to_n(n):

is_prime = np.ones(n + 1, dtype=bool)

is_prime[:2] = False

r = int(n**0.5)

for p in range(2, r+1):

if is_prime[p]:

is_prime[p*p:n+1:p] = False

return np.where(is_prime)[0]



print("Generating real primes up to", MAX_PRIME_VAL)

primes = get_primes_up_to_n(MAX_PRIME_VAL)

primes = primes[primes >= 3] # start at 3 to match previous behavior



# ---------------- helpers ----------------

def compute_da_from_sequence(seq):

# seq: increasing integer array of "primes"

if len(seq) < 4:

return np.array([])

alpha = np.arctan(seq[:-1] / seq[1:])

da = np.diff(alpha)

return detrend(da)



def find_peak_near(f, P, target, window=0.02):

mask = (f >= target - window) & (f <= target + window)

if not np.any(mask):

return (None, None)

idx_rel = np.argmax(P[mask])

idx = np.where(mask)[0][idx_rel]

return (f[idx], P[idx])



# ---------------- synthetic generators ----------------

rng = np.random.default_rng(2025)



def uniform_random_gaps(max_val, mean_gap):

max_gap = max(3, int(2 * mean_gap))

# generate many small gaps, make them odd

gaps = rng.integers(1, max_gap+1, size=ceil(max_val/max_gap)*3)

gaps = gaps | 1

seq = np.cumsum(gaps)

seq = seq[seq <= max_val]

seq = np.unique(np.insert(seq, 0, 3))

return seq



def poisson_gaps(max_val, mean_gap):

# exponential interarrival times

gaps = rng.exponential(scale=mean_gap, size=ceil(max_val/mean_gap)*3)

gaps = np.round(gaps).astype(int)

gaps = np.maximum(gaps, 1)

gaps = gaps | 1

seq = np.cumsum(gaps)

seq = seq[seq <= max_val]

seq = np.unique(np.insert(seq, 0, 3))

return seq



def random_odd_sequence(max_val, n_terms):

odds = np.arange(3, max_val+1, 2)

choose = rng.choice(odds, size=n_terms, replace=False)

seq = np.sort(choose)

seq = np.unique(np.insert(seq, 0, 3))

return seq



def gaussian_perturb_primes(primes_arr, sigma=2.0):

noise = rng.normal(loc=0.0, scale=sigma, size=len(primes_arr))

pert = np.round(primes_arr + noise).astype(int)

pert = np.maximum(pert, 3)

pert = np.sort(pert)

for i in range(1, len(pert)):

if pert[i] <= pert[i-1]:

pert[i] = pert[i-1] + 2

pert = (pert | 1)

pert = pert[pert <= MAX_PRIME_VAL]

return np.unique(pert)



# ---------------- build sequences ----------------

mean_gap = np.mean(np.diff(primes))

seqs = {

"real_primes": primes,

"uniform_gaps": uniform_random_gaps(MAX_PRIME_VAL, mean_gap),

"poisson_gaps": poisson_gaps(MAX_PRIME_VAL, mean_gap),

"random_odd": random_odd_sequence(MAX_PRIME_VAL, len(primes)),

"gauss_perturb": gaussian_perturb_primes(primes, sigma=2.0)

}



# ---------------- compute PSDs and peaks ----------------

rows = []

psd_store = {}

for name, seq in seqs.items():

da = compute_da_from_sequence(seq)

if da.size == 0:

continue

f, P = welch(da, fs=1.0, window='hann', nperseg=NPERSEG, noverlap=NOVERLAP)

psd_store[name] = (f, P)

for t in TARGET_FREQS:

pkf, pkp = find_peak_near(f, P, t)

rows.append({

"sequence": name,

"n_terms": len(seq),

"target_freq": t,

"peak_freq": pkf,

"peak_power": pkp

})



df = pd.DataFrame(rows)

csv_path = os.path.join(OUTDIR, "synthetic_psd_peaks.csv")

df.to_csv(csv_path, index=False)

print("Saved CSV:", csv_path)



# ---------------- overlay plot ----------------

plt.figure(figsize=(12,6))

for name, (f, P) in psd_store.items():

if name == "real_primes":

plt.semilogy(f, P, lw=2, color='k', label=name)

else:

plt.semilogy(f, P, alpha=0.8, label=name)

for t in TARGET_FREQS:

plt.axvline(t, color='red', ls='--', alpha=0.6)

plt.xlim(0.1, 0.55)

plt.xlabel("Frequency f")

plt.ylabel("PSD (log scale)")

plt.title("PSD Overlay: Real primes vs Synthetic Controls")

plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))

plt.tight_layout()

plt.savefig(os.path.join(OUTDIR, "synthetic_psd_overlay.png"), dpi=200)

plt.close()



# ---------------- peak bar charts ----------------

for t in TARGET_FREQS:

vals = []

names = []

for name in psd_store:

f, P = psd_store[name]

pkf, pkp = find_peak_near(f, P, t)

vals.append(pkp if pkp is not None else 0.0)

names.append(name)

plt.figure(figsize=(9,4))

plt.bar(names, vals)

plt.yscale('log')

plt.ylabel("Peak Power (log scale)")

plt.title(f"Peak Power near f={t:.5f} across sequences")

plt.tight_layout()

plt.savefig(os.path.join(OUTDIR, f"synthetic_peakbar_f{t:.5f}.png"), dpi=200)

plt.close()



print("Saved overlay and bar charts to folder:", OUTDIR)

print("All done. Inspect the CSV and PNGs in", OUTDIR)