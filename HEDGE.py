#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.special import gammaincc
from scipy.stats import chi2, chisquare
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
def approximate_entropy_test(byte_data):
    """
    Approximate Entropy Test (NIST SP 800-22)
     Optimized with NumPy.
    
    Args:
        byte_data: A list or numpy array of integers (0-255).
                   Example: [255, 0, 128, ...]
    
    Returns:
        (success, p, None)
    """
    # Convert bytes (0-255) to a bit array (0-1) efficiently
    # np.unpackbits unpacks uint8 elements into bits (big-endian order)
    data = np.array(byte_data, dtype=np.uint8)
    bits = np.unpackbits(data)
    n = len(bits)
    
    # Calculate m based on the length of the bit stream
    m = int(np.floor(np.log2(n))) - 6
    if m < 2:
        m = 2
    if m > 3:
        m = 3
    
    phi_m = []
    
    # Loop for block sizes m and m+1
    for iterm in range(m, m + 2):
        # Step 1: Pad the bits (Circular buffer logic)
        # We append the first iterm-1 bits to the end
        padded_bits = np.concatenate((bits, bits[:iterm-1]))
        
        # Step 2: Vectorized Pattern Counting
        # Instead of iterating and converting substrings to ints in a loop,
        # we construct the integer values for all positions at once using bitwise shifts.
        
        # We use a container for the integer values of the patterns
        # Since m is capped at 3, patterns are max 4 bits (values 0-15), so int8 is sufficient.
        vals = np.zeros(n, dtype=np.int8)
        
        # Construct the integers for every window of length 'iterm'
        for k in range(iterm):
            vals <<= 1
            # Add the bit at the specific offset. 
            # Slice padded_bits from k to n+k to get the bit at that position for every window
            vals |= padded_bits[k : n + k]
            
        # Count occurrences of each pattern (0 to 2^iterm - 1)
        # bincount is extremely fast compared to a python loop
        counts = np.bincount(vals, minlength=2**iterm)
        
        # Step 3: Calculate Probabilities (Ci)
        Ci = counts.astype(np.float64) / n
        
        # Step 4: Calculate Phi
        # Formula: sum( Ci * log(Ci / 10.0) )
        # We only compute log for non-zero probabilities to avoid -inf
        valid_Ci = Ci[Ci > 0]
        
        # Note: The original code divides by 10.0 inside the log.
        # While mathematically this constant cancels out in the final subtraction,
        # we keep it to strictly match the original code's operation order.
        phi_val = np.sum(valid_Ci * np.log(valid_Ci / 10.0))
        phi_m.append(phi_val)
        
    # Step 6
    appen_m = phi_m[0] - phi_m[1]
    chisq = 2 * n * (np.log(2) - appen_m)
    
    # Step 7
    # Using scipy.special.gammaincc (Regularized upper incomplete gamma function)
    # This matches the statistical definition used in the NIST test.
    p = gammaincc(2**(m-1), (chisq / 2.0))
    
    success = (p >= 0.01)
    return (success, p, None)


def frequency_within_block_test(byte_data):
    """
    Frequency (Monobit) Test within a Block
    NIST SP 800-22
    Optimized with NumPy
    Input: byte array (values 0–255)
    Output: (success, p, None)
    """

    # Convert bytes → bits (0/1)
    data = np.asarray(byte_data, dtype=np.uint8)
    bits = np.unpackbits(data)
    n = bits.size

    # Original NIST constraint
    if n < 100:
        return False, 1.0, None

    # Block size selection (same logic as original)
    M = 20
    N = n // M
    if N > 99:
        N = 99
        M = n // N

    # Truncate to exact multiple
    bits = bits[:N * M]

    # Reshape into blocks: shape (N, M)
    blocks = bits.reshape((N, M))

    # Count ones per block (vectorized)
    ones = np.sum(blocks, axis=1)

    # Proportion of ones per block
    proportions = ones / float(M)

    # Chi-square calculation (exact same formula)
    chisq = np.sum(4.0 * M * (proportions - 0.5) ** 2)

    # P-value
    p = gammaincc(N / 2.0, chisq / 2.0)

    success = (p >= 0.01)
    return (success, p, None)


def normcdf(x):
    """
    Normal CDF using erfc (same as original)
    """
    return 0.5 * math.erfc(-x * math.sqrt(0.5))


def p_value(n, z):
    """
    NIST SP800-22 cumulative sums p-value computation
    Preserves original summation limits and formula
    """
    sqrt_n = math.sqrt(n)

    # Sum A
    sum_a = 0.0
    startk = int(math.floor(((-n / z) + 1.0) / 4.0))
    endk   = int(math.floor((( n / z) - 1.0) / 4.0))

    for k in range(startk, endk + 1):
        c1 = ((4.0 * k + 1.0) * z) / sqrt_n
        c2 = ((4.0 * k - 1.0) * z) / sqrt_n
        sum_a += normcdf(c1) - normcdf(c2)

    # Sum B
    sum_b = 0.0
    startk = int(math.floor(((-n / z) - 3.0) / 4.0))
    endk   = int(math.floor((( n / z) - 1.0) / 4.0))

    for k in range(startk, endk + 1):
        c1 = ((4.0 * k + 3.0) * z) / sqrt_n
        c2 = ((4.0 * k + 1.0) * z) / sqrt_n
        sum_b += normcdf(c1) - normcdf(c2)

    return 1.0 - sum_a + sum_b


def cumulative_sums_test(byte_data):
    """
    Cumulative Sums (Cusum) Test
    NIST SP 800-22
    Optimized with NumPy
    Input: byte array (0–255)
    Output: (success, None, [p_forward, p_backward])
    """

    # Convert bytes → bits
    data = np.asarray(byte_data, dtype=np.uint8)
    bits = np.unpackbits(data)
    n = bits.size

    # Step 1: Convert {0,1} → {-1,+1}
    x = 2 * bits - 1

    # Steps 2 & 3: cumulative sums
    cumsum_forward = np.cumsum(x)
    forward_max = int(np.max(np.abs(cumsum_forward)))

    cumsum_backward = np.cumsum(x[::-1])
    backward_max = int(np.max(np.abs(cumsum_backward)))

    # Step 4: p-values
    p_forward  = p_value(n, forward_max)
    p_backward = p_value(n, backward_max)

    success = (p_forward >= 0.01) and (p_backward >= 0.01)
    plist = [p_forward, p_backward]

    return (success, None, plist)

def chi2_uniform(row, Length=4096):
    """
    Computes the chi-square statistic for a given array of bytes
    to test if it follows a uniform distribution.
    
    Parameters:
        row (numpy.ndarray): A numpy array of byte values (0-255).
        Length (int): Total number of bytes in the fragment.
    
    Returns:
        float: The chi-square statistic.
    """
    # Expected frequency for uniform distribution
    Ei = Length / 256
    
    # Count occurrences of each byte value (0-255)
    observed_counts = np.bincount(row, minlength=256)
    
    # Compute chi-square statistic using vectorized operations
    chi2_stat = np.sum(((observed_counts - Ei) ** 2) / Ei)
    
    # Degrees of freedom (df = 256 - 1)
    df = 255

    # Compute p-value using chi-square survival function (1 - CDF)
    p_value = 1 - chi2.cdf(chi2_stat, df)

        
    return chi2_stat, p_value

def chi_test_absolute(chi2_uniform_stat, gamma=2, length=4096):
    if length==1024:
        success = (1 if (255.02-gamma*22.57)<=chi2_uniform_stat<=(255.02+gamma*22.57) else 0)
    elif length==2048:
        success = (1 if (254.98-gamma*22.57)<=chi2_uniform_stat<=(254.98+gamma*22.57) else 0)
    elif length==4096:
        success = (1 if (255.04-gamma*22.60)<=chi2_uniform_stat<=(255.04+gamma*22.60) else 0)
    elif length==8192:
        success = (1 if (255.09-gamma*22.54)<=chi2_uniform_stat<=(255.09+gamma*22.54) else 0)
    elif length==16.384:
        success = (1 if (254.96-gamma*22.76)<=chi2_uniform_stat<=(254.96+gamma*22.76) else 0)  
    elif length==32.768:
        success = (1 if (255.08-gamma*22.68)<=chi2_uniform_stat<=(255.08+gamma*22.68) else 0)     
    elif length==65.536:
        success = (1 if (255.37-gamma*22.82)<=chi2_uniform_stat<=(255.37+gamma*22.82) else 0)            
    return success

def chi_test_confidence(chi2_uniform_pvalue):
    
    success = (0 if chi2_uniform_pvalue<0.01 or chi2_uniform_pvalue>0.99 else 1)
    return success

def HEDGE_test(byte_data):
    """

    Input: byte array (0–255)
    Output: array of 0 or 1 (o means unecrypted and 1 means encrypted)
    """
    row = np.array(byte_data, dtype=np.uint8)

    r1 = approximate_entropy_test(row)[0]
    r2 = frequency_within_block_test(row)[0]
    r3 = cumulative_sums_test(row)[0]
    r4 = chi_test_absolute(chi2_uniform(row)[0])
    r5 = chi_test_confidence(chi2_uniform(row)[1])
    result = (r1 == True) and (r2 == True) and (r3 == True) and (r4 == True) and (r5==True)
    return result

if __name__ == "__main__":
    input_addr = input("input csv dataset(each row containing bytes eg[234, 2, ...]) address: ")
    output_addr = input("address of csv file you want the results save to ( 0 means unecrypted, 1 means encrypted): ")
    print("loading dataset ...")
    dataset = pd.read_csv(input_addr, header=None).values
    total = len(dataset)

    start = time.perf_counter()

    results = np.empty(total, dtype=np.uint8)

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(HEDGE_test, dataset[i]): i
            for i in range(total)
        }

        for future in tqdm(
            as_completed(futures),
            total=total,
            desc="Processing",
            unit="rows"
        ):
            idx = futures[future]
            results[idx] = future.result()

    end = time.perf_counter()

    print(f"\nExecution time: {end - start:.6f} seconds")
    print("wait it is saving")

    pd.DataFrame(results).to_csv(output_addr, index=False, header=False)