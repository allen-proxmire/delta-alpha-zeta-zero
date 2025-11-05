import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, welch
import pandas as pd

# CONFIG
MAX_PRIME_VALUE = 10_000_000 # Scaling up 5x from 2M
TARGET_FREQS = np.array([0.35153, 0.38895, 0.47871])
FS = 1.0
NPERSEG = 4096 # Increased resolution for larger N
NOVERLAP = NPERSEG // 2

# Helper functions for data generation
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

# --- Generate Data and Run PSD ---
print(f"Generating primes up to {MAX_PRIME_VALUE:,}...")
primes = get_primes_up_to_n(MAX_PRIME_VALUE)
full_da = detrend(calculate_delta_alpha(primes))
N = len(full_da)
print(f"Total detrended Δα signal length (N): {N:,}")

# Compute Power Spectral Density (PSD) using Welch's method
f, Pxx = welch(full_da, fs=FS, window='hann', nperseg=NPERSEG, noverlap=NOVERLAP, scaling='density')

# --- Peak Finding and Comparison ---
def find_nearest_peak(f_grid, Pxx_grid, target_f, search_width=0.01):
    """Finds the maximum PSD near a target frequency."""
    target_idx = np.argmin(np.abs(f_grid - target_f))
    
    # Define search window indices
    search_points = int(search_width / (f_grid[1] - f_grid[0]))
    start_idx = max(0, target_idx - search_points)
    end_idx = min(len(f_grid), target_idx + search_points)
    
    search_slice = Pxx_grid[start_idx:end_idx]
    search_f_slice = f_grid[start_idx:end_idx]
    
    if len(search_slice) == 0:
        return {'Target_f': target_f, 'Observed_f': np.nan, 'Power': np.nan, 'Shift_ppm': np.nan}
        
    # Find the peak within the slice
    peak_relative_idx = np.argmax(search_slice)
    peak_f = search_f_slice[peak_relative_idx]
    peak_power = search_slice[peak_relative_idx]
    
    shift_ppm = (peak_f - target_f) / target_f * 1e6
    
    return {
        'Target_f (2M)': target_f,
        'Observed_f (10M)': peak_f,
        'Power': peak_power,
        'Shift (ppm)': shift_ppm
    }

results = [find_nearest_peak(f, Pxx, tf) for tf in TARGET_FREQS]
df_results = pd.DataFrame(results)

# --- Plotting the Spectrum ---
plt.figure(figsize=(10, 6))
plt.plot(f, Pxx, color='#1E88E5', alpha=0.8, label='PSD (Primes $\leq 10M$)')

# Mark the observed peaks
for index, row in df_results.iterrows():
    plt.axvline(row['Observed_f (10M)'], color='red', ls='-', alpha=0.7, linewidth=1.5,
                label=f"f={row['Observed_f (10M)']:.5f}")
    
plt.xlim(0.3, 0.55)
plt.ylim(0, np.percentile(Pxx[f>0.3], 99)) # Zoom in on the area of interest
plt.xlabel('Frequency $f$'); plt.ylabel('Power Spectral Density (PSD)')
plt.title('PSD of $\Delta\alpha$ up to $10,000,000$ Primes')
plt.grid(True, ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig('extended_scaling_psd.png', dpi=200)
print("\nSaved 'extended_scaling_psd.png'")

print("\n--- Scaling Analysis Results (Primes $\leq 10,000,000$) ---")
print(df_results.to_string(index=False))