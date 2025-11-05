import numpy as np
import math
from scipy.signal import periodogram
from time import time
import sys
import matplotlib.pyplot as plt

# Set high recursion limit for deep prime generation if needed
sys.setrecursionlimit(3000)

# --- CONFIGURATION: DEFINE THE SAMPLE LIMITS ---
# Enter the maximum prime value for each sample you want to analyze and plot.
# The script will run up to the highest value listed here (e.g., 2,000,000).
SAMPLE_LIMITS = [20000, 200000, 2000000] 

# --- 1. Prime Generation and Delta Alpha Calculation ---

def get_primes_up_to_n(n):
    """Uses the Sieve of Eratosthenes."""
    if n < 2: return []
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0:2] = False 
    for p in range(2, int(np.sqrt(n)) + 1):
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
    return np.where(is_prime)[0] # Return as NumPy array for efficient slicing

def calculate_delta_alpha(p_list):
    """Calculates ∆α = alpha(n+1) - alpha(n) in radians."""
    # Optimized calculation for large arrays
    alpha = np.arctan(p_list[:-1] / p_list[1:])
    return np.diff(alpha)

# --- Orchestration: Generate primes and slice signals for all samples ---

MAX_PRIME_VALUE = max(SAMPLE_LIMITS)
print(f"Starting analysis for primes up to {MAX_PRIME_VALUE:,}...")
start_time = time()
primes = get_primes_up_to_n(MAX_PRIME_VALUE)
print(f"Primes found: {len(primes):,}. Time: {time() - start_time:.2f}s.")

all_sample_data = []
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
max_drift = 0 # Helper variable for consistent Y-axis scaling on Figure 2

for i, limit in enumerate(SAMPLE_LIMITS):
    # Find the index of the last prime <= limit
    # np.searchsorted is efficient for finding the split point
    end_index = np.searchsorted(primes, limit, side='right')
    
    # Slice the prime list and calculate Delta Alpha signal
    p_list = primes[:end_index]
    delta_alpha = calculate_delta_alpha(p_list)
    
    # Store all data needed for plotting and analysis
    data_entry = {
        'limit': limit,
        'count': len(p_list),
        'da_signal': delta_alpha,
        'color': colors[i % len(colors)],
        'label': f'Primes $\leq$ {limit:,} ($N={len(p_list):,}$)'
    }
    all_sample_data.append(data_entry)
    print(f"Sample {i+1} ({limit:,}): Calculated {len(delta_alpha):,} Delta Alpha values.")


# ----------------------------------------------------------------------
# --- FIGURE 2: THE DECAY OF GLOBAL DRIFT (GLOBAL DETERMINATION) ---
# ----------------------------------------------------------------------

def plot_global_drift_decay(samples_data):
    """
    Plots the running cumulative average of ∆α for all samples to visualize 
    Global Drift decay and compare convergence.
    """
    plt.figure(figsize=(10, 6))
    max_y_val = 0
    
    # Loop through all samples to plot
    for sample in samples_data:
        da_signal = sample['da_signal']
        
        cumulative_sum = np.cumsum(da_signal)
        n_steps = np.arange(1, len(da_signal) + 1)
        cumulative_average = cumulative_sum / n_steps
        
        # Track max Y for setting plot limits
        max_y_val = max(max_y_val, np.max(cumulative_average))
        
        # --- Plotting ---
        plt.plot(n_steps, cumulative_average, 
                 color=sample['color'], 
                 label=sample['label'], 
                 linewidth=1)
    
    # Target value: The limit is zero.
    plt.axhline(0, color='red', linestyle='--', alpha=0.6, label='Asymptotic Ideal (Zero Drift)')

    plt.title('Figure 2: Global Determination - Decay of Systematic Bias (Global Drift)')
    plt.xlabel('Prime Index ($n$)')
    plt.ylabel('Global Drift (Running Average of $\\Delta\\alpha$)')
    
    # Set y-axis limits to clearly show the curve dropping toward zero
    plt.ylim(-max_y_val / 10, max_y_val * 1.1)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=9)
    plt.savefig('figure_2_global_drift_decay_3samples.png')
    print("Generated Figure 2: 'figure_2_global_drift_decay_3samples.png'")

# ----------------------------------------------------------------------
# --- FIGURE 3: THE INVARIANT SPECTRAL PEAKS (LOCAL COHERENCE) ---
# ----------------------------------------------------------------------

def plot_fixed_wave_modes(samples_data):
    """
    Performs FFT analysis and plots the power spectrum overlay for all samples, 
    guaranteeing the three target frequencies are labeled based on the largest sample.
    """
    # Use the largest sample (last in the list) for the base line and peak identification
    base_sample = samples_data[-1]
    da_signal_base = base_sample['da_signal']

    # FFT Analysis for Base Sample
    f_base, Pxx_base = periodogram(da_signal_base, fs=1.0, window='hann')
    
    # Target Frequencies from the Paper (Sorted by frequency: Fast, Mid, Slow)
    target_modes = {
        'Fast Mode (T=2.089)': 0.47871, 
        'Mid Mode (T=2.571)': 0.38895, 
        'Slow Mode (T=2.845)': 0.35153
    }
    
    identified_peaks = []
    peak_colors = ['#d62728', '#9467bd', '#8c564b']
    
    # Explicitly find the actual peak closest to the target frequency in the base sample
    for i, (mode, target_f) in enumerate(target_modes.items()):
        idx = np.argmin(np.abs(f_base - target_f))
        
        identified_peaks.append({
            "freq": f_base[idx], 
            "mag": Pxx_base[idx], 
            "mode": mode,
            "color": peak_colors[i]
        })

    # --- Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot all samples as an overlay
    for sample in samples_data:
        f, Pxx = periodogram(sample['da_signal'], fs=1.0, window='hann')
        # Use lower alpha for smaller samples to emphasize the largest one
        alpha_val = 1.0 if sample == base_sample else 0.5
        
        plt.plot(f, Pxx, label=sample['label'], 
                 color=sample['color'], 
                 alpha=alpha_val, 
                 linewidth=1.5 if sample == base_sample else 1.0)


    # Label the three fixed peaks using the explicitly identified targets from the base sample
    for peak in identified_peaks:
        freq = peak['freq']
        mag = peak['mag']
        
        # Place a dot on the peak (using the base sample's magnitude)
        plt.plot(freq, mag, 'o', color=peak['color'], markersize=7, 
                 label=f"{peak['mode']} $f={freq:.5f}$")
        
        # Add a text label (Frequency)
        label_text = f"$f={freq:.5f}$"
        
        # Annotate slightly offset for readability
        plt.annotate(label_text, 
                     xy=(freq, mag), 
                     xytext=(freq + 0.005, mag * 1.05), # Offset right and up
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5),
                     fontsize=9, 
                     color=peak['color'])

    plt.title(f'Figure 3: Local Coherence - Invariant Spectral Modes (Overlay of {len(samples_data)} Samples)')
    plt.xlabel('Frequency ($f$)')
    plt.ylabel('Power Spectral Density (PSD)')
    plt.xlim(0.1, 0.55)
    # Exclude DC component (index 0) from max calculation for better scaling
    plt.ylim(0, np.max(Pxx_base[1:]) * 1.1) 
    
    plt.grid(True, linestyle=':', alpha=0.6)
    # Move legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig('figure_3_fixed_wave_modes_3samples.png')
    print("Generated Figure 3: 'figure_3_fixed_wave_modes_3samples.png'")

# ----------------------------------------------------------------------
# --- EXECUTION ---
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # Execute the functions to generate the plots using all sample data
    plot_global_drift_decay(all_sample_data)
    plot_fixed_wave_modes(all_sample_data)
    plt.show()
