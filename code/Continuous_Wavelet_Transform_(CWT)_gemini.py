import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, cwt, morlet
from numpy.fft import fftfreq

# --- Initial Data Setup (Recreating environment from previous turns) ---
MAX_PRIME_VALUE = 2_000_000
TARGET_FREQS = np.array([0.35153, 0.38895, 0.47871])
FS = 1.0 # Sampling Frequency (dt = 1)
# Center frequency for the Morlet wavelet, often 1/2pi * w0, where w0=6. We use the SciPy default.
MORLET_CENTER_FREQ = 0.8125 

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

# Generate the full Δα signal
primes = get_primes_up_to_n(MAX_PRIME_VALUE)
full_da = detrend(calculate_delta_alpha(primes))
N = len(full_da)
print(f"Total detrended Δα signal length (N): {N:,}")
# ---------------------------------------------------------------------

# --- CWT Setup ---

# 1. Define Scale Grid (Widths)
# CWT scale 'a' relates to frequency 'f' approximately as: a = Fc / f * dt
# We need to map our target frequencies to the corresponding scales.
target_scales = MORLET_CENTER_FREQ / TARGET_FREQS * (1/FS)

# Create a sequence of scales that covers the relevant frequency range (up to Nyquist, f=0.5)
# Scales for f=0.5 (highest relevant freq) to f=0.01 (very low freq)
min_scale = MORLET_CENTER_FREQ / 0.5 * (1/FS)
max_scale = MORLET_CENTER_FREQ / 0.01 * (1/FS)

# Use 100 scales, log-spaced for better resolution at high frequencies
scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 100)
scales = scales[::-1] # Reverse to align from low scale (high freq) to high scale (low freq)

# 2. Convert scales back to a frequency array for plotting
cwt_freqs = MORLET_CENTER_FREQ / scales * FS

# 3. Perform CWT
# The 'morlet' function in scipy is a good default wavelet for spectral analysis
wavelet_coefficients = cwt(full_da, morlet, scales, w=6) # w=6 is a standard Morlet parameter

# Calculate the Power Spectrum (Magnitude Squared)
power_spectrum = np.abs(wavelet_coefficients)**2

print(f"CWT Power Spectrum computed for {len(scales)} scales.")

# --- Plotting the Scalogram ---

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the CWT Scalogram (time-frequency power map)
# The power is often log-transformed for better visual contrast
im = ax.pcolormesh(
    np.arange(N), # Time/Index axis
    cwt_freqs, # Frequency axis
    power_spectrum,
    cmap='inferno',
    shading='gouraud',
    vmax=np.percentile(power_spectrum, 95) # Cap max power for visualization
)
cbar = fig.colorbar(im, ax=ax, label='Wavelet Power')

# Mark the locations of the three coherent modes
for tf in TARGET_FREQS:
    ax.axhline(tf, color='lime', linestyle='--', alpha=0.9, linewidth=1.5,
               label=f'Target Mode f={tf:.5f}')

ax.set_title('Continuous Wavelet Transform (CWT) Scalogram of $\Delta\alpha$')
ax.set_xlabel('Prime Index $n$ (from 1 to 148,931)')
ax.set_ylabel('Frequency $f$')
ax.set_ylim(0.25, 0.55) # Focus on the relevant high-frequency band
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('wavelet_scalogram.png', dpi=200)
print("Saved 'wavelet_scalogram.png'")

# --- Quantitative Check (Mean Power across the sequence) ---
print("\n--- Quantitative Power Check ---")

# Find the scale index closest to each target frequency
def nearest_scale_idx(f_array, val):
    return np.argmin(np.abs(f_array - val))

power_summary = []
for tf in TARGET_FREQS:
    idx = nearest_scale_idx(cwt_freqs, tf)
    # Average power over the entire sequence length (N)
    mean_power = np.mean(power_spectrum[idx, :])
    # Standard Deviation of power (measures amplitude fluctuation over time)
    std_power = np.std(power_spectrum[idx, :])
    
    power_summary.append({
        'f': tf,
        'f_actual': cwt_freqs[idx],
        'Mean_Power': mean_power,
        'Power_StdDev': std_power,
        'Coeff_of_Variation': std_power / mean_power
    })

df_power = pd.DataFrame(power_summary)
print(df_power.to_string(index=False))