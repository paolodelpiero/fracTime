import numpy as np

def calculate_rs(returns, n):
    number_of_points = len(returns)
    number_of_points = number_of_points/n
    RS_Means = []
    for i in range(1, int(number_of_points)+1):
         segment = returns[(i-1)*n : i*n]
         # R/S 
         R = np.max(np.cumsum(segment - np.mean(segment))) - np.min(np.cumsum(segment - np.mean(segment)))
         S = np.std(segment)
         if S == 0:
             continue
         RS_Means.append(R / S)
    
    RS_Mean = np.mean(RS_Means)
    return RS_Mean
        

def hurst_exponent(returns):
    n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    RS_values = []
    for n in n_values:
        RS_values.append(calculate_rs(returns, n))

    log_RS = np.log(RS_values)
    log_n = np.log(n_values)

    # linear regression
    A = np.vstack([log_n, np.ones(len(log_n))]).T
    hurst_exponent, _ = np.linalg.lstsq(A, log_RS, rcond=None)[0]

    return hurst_exponent