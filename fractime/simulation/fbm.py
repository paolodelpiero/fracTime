import numpy as np

def fbm_covariance_matrix(n_steps, H):
    cov = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            cov[i, j] = 0.5 * ((i+1)**(2*H) + (j+1)**(2*H) - abs(i-j)**(2*H))
    return cov

def generate_fbm_path(n_steps, H, seed=None):
    if seed is not None:
        np.random.seed(seed)
    cov_matrix = fbm_covariance_matrix(n_steps, H)
    cholesky = np.linalg.cholesky(cov_matrix)
    random_vector = np.random.randn(n_steps)
    path = cholesky @ random_vector
    return path

def generate_price_scenarios(current_price, H, volatility, n_steps, n_scenarios=100):
    scenarios = np.zeros((n_scenarios, n_steps))
    
    for i in range(n_scenarios):
        fbm_path = generate_fbm_path(n_steps, H)
        fbm_increments = np.diff(fbm_path, prepend=0)
        fbm_increments = fbm_increments / np.std(fbm_increments)
        scaled_returns = fbm_increments * volatility
        prices = current_price * np.exp(np.cumsum(scaled_returns))
        scenarios[i] = prices
    
    return scenarios

def calculate_risk_metrics(scenarios, confidence_level=0.95):
    final_prices = scenarios[:, -1]
    initial_price = scenarios[0, 0] 
    total_returns = (final_prices - initial_price) / initial_price
    var = np.percentile(total_returns, (1 - confidence_level) * 100)
    cvar = np.mean(total_returns[total_returns <= var])
    return {
        'VaR': var,
        'CVaR': cvar,
        'Mean Return': np.mean(total_returns),
        'Std Dev Return': np.std(total_returns)
    }