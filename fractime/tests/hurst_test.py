from fractime.core.hurst import hurst_exponent
import numpy as np

# Test 1 - Random walk (rendimenti indipendenti)
np.random.seed(42)
random_returns = np.random.randn(2000)
H_random = hurst_exponent(random_returns)
print(f"Random walk H: {H_random}")

# Test 2 - Serie mean-reverting (rendimenti che si invertono)
np.random.seed(42)
mean_rev = np.zeros(2000)
for i in range(1, 2000):
    mean_rev[i] = -0.5 * mean_rev[i-1] + np.random.randn()
H_mean_rev = hurst_exponent(mean_rev)
print(f"Mean reverting H: {H_mean_rev}")