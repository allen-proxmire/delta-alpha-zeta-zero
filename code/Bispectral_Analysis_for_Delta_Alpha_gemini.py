import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, detrend
from numpy.fft import fft
import pandas as pd

# --- Initial Data Setup (Recreating environment from previous turns) ---
MAX_PRIME_VALUE = 2_000_000
TARGET_FREQS = np.array([0.35153, 0.38895, 0.47871])
FS = 1.0 # Sampling Frequency

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
print(f"Total detrended Δα signal length: {N:,}")
# ---------------------------------------------------------------------

# --- Bispectrum Configuration ---
# NPERSEG must be a power of 2 for FFT efficiency and must be a divisor of N for non-overlapping segments.
# We'll use 4096 segments. N = 148,931. Let's use 4096 and take a non-overlapping average.
NPERSEG = 4096
# Calculate number of non-overlapping segments
N_SEGMENTS = int(N // NPERSEG)
# Trim the signal to an exact multiple of NPERSEG
signal = full_da[:N_SEGMENTS * NPERSEG]

print(f"Using {N_SEGMENTS} non-overlapping segments of length {NPERSEG}.")

# The frequency grid based on NPERSEG
freq_grid = np.linspace(0, FS/2, NPERSEG // 2 + 1)
# Find the indices corresponding to our target frequencies
def nearest_idx(fgrid, val):
    return np.argmin(np.abs(fgrid - val))

# Get indices for the three modes A, B, C
idx_A = nearest_idx(freq_grid, TARGET_FREQS[0])
idx_B = nearest_idx(freq_grid, TARGET_FREQS[1])
idx_C = nearest_idx(freq_grid, TARGET_FREQS[2])

# Use the actual frequencies corresponding to the FFT indices
f_A = freq_grid[idx_A]; f_B = freq_grid[idx_B]; f_C = freq_grid[idx_C]
print(f"Actual frequencies used: f_A={f_A:.5f}, f_B={f_B:.5f}, f_C={f_C:.5f}")


def compute_bicoherence(x, nperseg, n_segments, fs):
    """
    Computes the squared bicoherence (b^2) for a signal x.
    b^2(f1, f2) measures quadratic phase coupling (QPC).
    """
    segment_length = nperseg
    window = windows.hann(segment_length)

    # Initialize averages for the numerator and denominator terms
    B_num = np.zeros((segment_length, segment_length), dtype=complex)
    B_den_term1 = np.zeros((segment_length, segment_length), dtype=float)
    B_den_term2 = np.zeros((segment_length), dtype=float)

    for k in range(n_segments):
        # 1. Segment and apply window
        segment = x[k * segment_length: (k + 1) * segment_length] * window

        # 2. Compute FFT
        X = fft(segment)
        # Shift the zero frequency to the start for easier indexing
        # Note: We compute for all frequencies but only plot the positive-positive quadrant

        # 3. Calculate terms for the numerator: X(f1) * X(f2) * X*(f1+f2)
        # Outer product for X(f1) * X(f2)
        X1X2 = np.outer(X, X)

        # Iterate over frequency pairs (f1, f2)
        for i in range(segment_length):
            for j in range(segment_length):
                # Calculate the index k for the sum frequency f3 = f1 + f2
                idx_k = (i + j) % segment_length
                
                # Update numerator (Averaging the Bispectrum)
                B_num[i, j] += X1X2[i, j] * np.conjugate(X[idx_k])

                # Update denominator term 1: |X(f1)X(f2)|^2
                B_den_term1[i, j] += np.abs(X1X2[i, j])**2
        
        # Update denominator term 2: |X(f3)|^2 (used for normalization)
        B_den_term2 += np.abs(X)**2

    # 4. Final Bicoherence Calculation (b^2)
    B_num /= n_segments # Final Bispectrum estimate
    B_den_term1 /= n_segments
    B_den_term2 /= n_segments

    # Initialize bicoherence array
    b_squared = np.zeros_like(B_num, dtype=float)

    for i in range(segment_length):
        for j in range(segment_length):
            idx_k = (i + j) % segment_length
            
            denominator = B_den_term1[i, j] * B_den_term2[idx_k]
            
            # Use a small epsilon to avoid division by zero
            epsilon = 1e-10
            if denominator > epsilon:
                b_squared[i, j] = np.abs(B_num[i, j])**2 / denominator

    # Return only the positive-positive frequency quadrant
    # f1 and f2 indices go from 0 up to (NPERSEG/2)
    limit = segment_length // 2 + 1
    
    # Calculate frequencies for the axes (up to Nyquist)
    f_axis = np.linspace(0, fs / 2, limit)

    return b_squared[:limit, :limit], f_axis

# --- Execute Analysis ---
b_squared, f_axis = compute_bicoherence(signal, NPERSEG, N_SEGMENTS, FS)
print("\nBispectral analysis complete.")

# --- Plotting ---
plt.figure(figsize=(10, 8))

# Use the logarithm of b_squared for better visualization contrast
# The scale ranges from 0 (no coupling) to 1 (perfect QPC)
plt.pcolormesh(f_axis, f_axis, b_squared, shading='nearest', cmap='viridis', vmin=0, vmax=0.01) # Set max to 0.01 for visibility
cbar = plt.colorbar(label='Squared Bicoherence ($b^2$)')

# Overlay the key combination points
plt.scatter([f_A, f_B, f_C], [f_A, f_B, f_C], color='red', marker='x', s=100, label='Diagonal ($f_1 = f_2$)')

# Check for a diagonal line crossing the target frequencies (i.e., self-coupling: 2f -> 2f)
# This is mainly a visual check for symmetry, as 2f is out of band

# Plot the most interesting difference interactions as a guide
# (f_A, f_B) -> f_B - f_A (Low frequency f_L is f_2 - f_1)
plt.scatter(f_B, f_A, color='white', marker='o', s=150, edgecolors='black', label='$f_B, f_A$ Pair')
plt.scatter(f_C, f_B, color='yellow', marker='o', s=150, edgecolors='black', label='$f_C, f_B$ Pair')
plt.scatter(f_C, f_A, color='magenta', marker='o', s=150, edgecolors='black', label='$f_C, f_A$ Pair')

plt.xlabel('$f_1$ (Frequency)'); plt.ylabel('$f_2$ (Frequency)')
plt.title('Squared Bicoherence ($b^2$) of $\Delta\alpha$ (Primes $\leq 2M$)')
plt.xlim(0, 0.5); plt.ylim(0, 0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('bispectral_analysis.png', dpi=200)
print("\nSaved 'bispectral_analysis.png'")

# --- Quantitative Check of Key Combinations (QPC) ---

# Check the six points corresponding to (f_i, f_j) pairs for DIFFERENCE coupling (f_j - f_i)
# Note: Since the modes are high, sum coupling (f_i + f_j) is out of band (> 0.5)
print("\n--- Quantitative Check of Difference Coupling (f_j - f_i) ---")
print("These points check if the difference frequency is generated nonlinearly.")

# Index lookup function for b_squared array
def lookup_bicoherence(f1, f2):
    idx1 = nearest_idx(f_axis, f1)
    idx2 = nearest_idx(f_axis, f2)
    return b_squared[idx1, idx2]

coupling_results = []

# f_B vs f_A
b_BA = lookup_bicoherence(f_A, f_B)
coupling_results.append({
    'f1': f_A, 'f2': f_B, 'Target_f3': np.abs(f_B - f_A),
    'b_squared': b_BA, 'Type': 'Diff'
})

# f_C vs f_B
b_CB = lookup_bicoherence(f_B, f_C)
coupling_results.append({
    'f1': f_B, 'f2': f_C, 'Target_f3': np.abs(f_C - f_B),
    'b_squared': b_CB, 'Type': 'Diff'
})

# f_C vs f_A
b_CA = lookup_bicoherence(f_A, f_C)
coupling_results.append({
    'f1': f_A, 'f2': f_C, 'Target_f3': np.abs(f_C - f_A),
    'b_squared': b_CA, 'Type': 'Diff'
})

df_results = pd.DataFrame(coupling_results)

# Format the output table
print(f"{'f1 (Mode)':<10} {'f2 (Mode)':<10} {'|f2 - f1|':<10} {'Bicoherence (b^2)':<20} {'Interpretation':<20}")
print("-" * 70)
for index, row in df_results.iterrows():
    if row['b_squared'] > 0.005:
        interp = "Possible Coupling"
    else:
        interp = "Weak Coupling"
    print(f"{row['f1']:<10.5f} {row['f2']:<10.5f} {row['Target_f3']:<10.5f} {row['b_squared']:<20.8f} {interp:<20}")