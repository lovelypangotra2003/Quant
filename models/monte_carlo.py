# monte_carlo.py (new file)
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from utils.calculators import PortfolioCalculators

class MonteCarloOptimizer:
    """
    Monte Carlo simulation for portfolio optimization.
    Generates random portfolios and finds optimal ones based on various criteria.
    """
    
    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, 
                 risk_free_rate: float = 0.02):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        self.tickers = expected_returns.index.tolist()
    
    def run_simulation(self, n_portfolios: int = 10000, 
                      constraints: Dict = None) -> pd.DataFrame:
        """Run Monte Carlo simulation with constraints."""
        if constraints is None:
            constraints = {}
            
        no_shorting = constraints.get('no_shorting', True)
        max_weight = constraints.get('max_weight', 1.0)
        
        return PortfolioCalculators.monte_carlo_simulation(
            self.expected_returns, self.cov_matrix, n_portfolios,
            self.risk_free_rate, no_shorting, max_weight
        )
    
    def find_optimal_portfolios(self, simulation_results: pd.DataFrame) -> Dict:
        """Find optimal portfolios from simulation results."""
        # Max Sharpe portfolio
        max_sharpe_idx = simulation_results['sharpe_ratio'].idxmax()
        max_sharpe_portfolio = {
            'weights': simulation_results.loc[max_sharpe_idx, 'weights'],
            'return': simulation_results.loc[max_sharpe_idx, 'return'],
            'volatility': simulation_results.loc[max_sharpe_idx, 'volatility'],
            'sharpe_ratio': simulation_results.loc[max_sharpe_idx, 'sharpe_ratio']
        }
        
        # Min Volatility portfolio
        min_vol_idx = simulation_results['volatility'].idxmin()
        min_vol_portfolio = {
            'weights': simulation_results.loc[min_vol_idx, 'weights'],
            'return': simulation_results.loc[min_vol_idx, 'return'],
            'volatility': simulation_results.loc[min_vol_idx, 'volatility'],
            'sharpe_ratio': simulation_results.loc[min_vol_idx, 'sharpe_ratio']
        }
        
        # Max Return portfolio
        max_return_idx = simulation_results['return'].idxmax()
        max_return_portfolio = {
            'weights': simulation_results.loc[max_return_idx, 'weights'],
            'return': simulation_results.loc[max_return_idx, 'return'],
            'volatility': simulation_results.loc[max_return_idx, 'volatility'],
            'sharpe_ratio': simulation_results.loc[max_return_idx, 'sharpe_ratio']
        }
        
        return {
            'max_sharpe': max_sharpe_portfolio,
            'min_volatility': min_vol_portfolio,
            'max_return': max_return_portfolio,
            'all_portfolios': simulation_results
        }
    
    def get_efficient_frontier(self, simulation_results: pd.DataFrame, 
                             n_points: int = 50) -> pd.DataFrame:
        """Extract efficient frontier from simulation results."""
        # Sort by volatility and find non-dominated portfolios
        sorted_results = simulation_results.sort_values('volatility')
        efficient_portfolios = []
        max_return_so_far = -np.inf
        
        for _, portfolio in sorted_results.iterrows():
            if portfolio['return'] > max_return_so_far:
                efficient_portfolios.append(portfolio)
                max_return_so_far = portfolio['return']
        
        efficient_df = pd.DataFrame(efficient_portfolios)
        
        # Interpolate to get smooth frontier
        if len(efficient_df) > 2:
            min_vol = efficient_df['volatility'].min()
            max_vol = efficient_df['volatility'].max()
            target_volatilities = np.linspace(min_vol, max_vol, n_points)
            
            frontier_returns = np.interp(target_volatilities, 
                                       efficient_df['volatility'], 
                                       efficient_df['return'])
            
            frontier_data = []
            for vol, ret in zip(target_volatilities, frontier_returns):
                frontier_data.append({
                    'volatility': vol,
                    'return': ret,
                    'sharpe_ratio': (ret - self.risk_free_rate) / vol if vol > 0 else 0
                })
            
            return pd.DataFrame(frontier_data)
        
        return efficient_df