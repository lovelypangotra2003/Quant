import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, List
import warnings
import streamlit as st
from utils.calculators import PortfolioCalculators
from sklearn.covariance import LedoitWolf

class EnhancedMarkowitzOptimizer:
    """
    Enhanced Markowitz optimization with regularization and multiple objective functions.
    """
    
    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, 
                 risk_free_rate: float = 0.02):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        self.tickers = expected_returns.index.tolist()
        
    def portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        
        port_return = np.dot(weights, self.expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        # Realistic bounds
        port_return = np.clip(port_return, -0.50, 1.00)  # Max 100% annual return
        port_volatility = np.clip(port_volatility, 0.05, 0.80)  # 5% to 80% vol

        sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility if port_volatility > 0 else 0
        
        return {
            'return': float(port_return),
            'volatility': float(port_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'weights': weights
        }
    
    def optimize(self, objective: str = 'max_sharpe', 
                 constraints: Dict = None,
                 regularization: float = 0.0,
                 **kwargs) -> Dict[str, float]:
        """
        Enhanced optimization with regularization and multiple objectives.
        
        Parameters:
        -----------
        objective : str
            One of: 'max_sharpe', 'min_volatility', 'max_return', 
                    'max_diversification', 'risk_parity', 'max_decorrelation'
        constraints : Dict
            Optimization constraints
        regularization : float
            L2 regularization strength (0 = no regularization)
        """
        if constraints is None:
            constraints = {}
            
        # Default constraints
        default_constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Set bounds
        no_shorting = constraints.get('no_shorting', True)
        max_weight = constraints.get('max_weight', 0.4)
        
        if no_shorting:
            bounds = tuple((0, max_weight) for _ in range(self.n_assets))
        else:
            bounds = tuple((-max_weight, max_weight) for _ in range(self.n_assets))
            
        # Initial guess (equal weighting)
        x0 = np.array([1.0 /self.n_assets] * self.n_assets)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)
        if x0.sum() <= 0:
            x0 = (lb + ub) / 2.0
        x0 = x0 / x0.sum()
        
        # Define objective functions with regularization
        if objective == 'max_sharpe':
            def objective_function(weights):
                metrics = self.portfolio_metrics(weights)
                reg_penalty = regularization * np.sum((weights - 1/self.n_assets) ** 2)
                return -metrics['sharpe_ratio'] + reg_penalty
                
        elif objective == 'min_volatility':
            def objective_function(weights):
                metrics = self.portfolio_metrics(weights)
                reg_penalty = regularization * np.sum((weights - 1/self.n_assets) ** 2)
                return metrics['volatility'] + reg_penalty
                
        elif objective == 'max_return':
            def objective_function(weights):
                metrics = self.portfolio_metrics(weights)
                reg_penalty = regularization * np.sum((weights - 1/self.n_assets) ** 2)
                return -metrics['return'] + reg_penalty
                
        elif objective == 'max_diversification':
            def objective_function(weights):
                # Maximum Diversification Ratio
                individual_vols = np.sqrt(np.diag(self.cov_matrix))
                weighted_vol = np.sum(weights * individual_vols)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                
                if portfolio_vol > 0:
                    diversification_ratio = weighted_vol / portfolio_vol
                else:
                    diversification_ratio = 0
                    
                reg_penalty = regularization * np.sum((weights - 1/self.n_assets) ** 2)
                return -diversification_ratio + reg_penalty  # Maximize diversification
                
        elif objective == 'risk_parity':
            def objective_function(weights):
                # Equal risk contribution
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                marginal_risk = np.dot(self.cov_matrix, weights) / portfolio_vol
                risk_contributions = weights * marginal_risk
                
                # Objective: minimize variance of risk contributions
                risk_parity_deviation = np.std(risk_contributions)
                reg_penalty = regularization * np.sum((weights - 1/self.n_assets) ** 2)
                return risk_parity_deviation + reg_penalty
                
        elif objective == 'max_decorrelation':
            def objective_function(weights):
                # Maximum Decorrelation (uses correlation matrix instead of covariance)
                vols = np.sqrt(np.diag(self.cov_matrix))
                corr_matrix = self.cov_matrix / np.outer(vols, vols)
                portfolio_variance = np.dot(weights.T, np.dot(corr_matrix, weights))
                reg_penalty = regularization * np.sum((weights - 1/self.n_assets) ** 2)
                return portfolio_variance + reg_penalty
                
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Perform optimization
        result = minimize(
            fun=objective_function,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=default_constraints,
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
            return self.portfolio_metrics(x0)
        
        # Clip and renormalize final weights (numerical safety)
        weights = np.clip(result.x, lb, ub)
        if weights.sum() == 0:
            weights = x0
        else:
            weights = weights / weights.sum()
        
        # Check concentration and warn if needed
        if np.max(weights) > 0.5:
            concentrated_asset = self.tickers[np.argmax(weights)]
            st.warning(f"⚠️ High concentration in {objective}: {np.max(weights):.1%} in {concentrated_asset}")
        
        return self.portfolio_metrics(weights)
    
    def robust_optimization(self, returns_data: pd.DataFrame, 
                          constraints: Dict = None,
                          n_bootstraps: int = 50,
                          regularization: float = 0.1) -> Dict:
        """
        Robust optimization using bootstrap resampling.
        """
        if constraints is None:
            constraints = {}
            
        bootstrap_weights = []
        
        for i in range(n_bootstraps):
            # Bootstrap sample with replacement
            sample_returns = returns_data.sample(n=len(returns_data), replace=True)
            sample_er = sample_returns.mean() * 252
            sample_cov = sample_returns.cov() * 252
            
            # Optimize on bootstrap sample
            try:
                temp_optimizer = EnhancedMarkowitzOptimizer(sample_er, sample_cov, self.risk_free_rate)
                result = temp_optimizer.optimize('max_sharpe', constraints, regularization)
                bootstrap_weights.append(result['weights'])
            except Exception as e:
                continue
        
        if bootstrap_weights:
            # Average weights across bootstrap samples
            avg_weights = np.mean(bootstrap_weights, axis=0)
            avg_weights = avg_weights / np.sum(avg_weights)  # Re-normalize
            
            # Calculate final metrics using original parameters
            return self.portfolio_metrics(avg_weights)
        else:
            # Fallback to standard optimization
            return self.optimize('max_sharpe', constraints, regularization)

    def efficient_frontier(self, n_points: int = 50, constraints: Dict = None) -> pd.DataFrame:
        """Generate efficient frontier with constraints."""
        if constraints is None:
            constraints = {}
            
        # Get minimum variance portfolio
        min_vol_result = self.optimize('min_volatility', constraints)
        min_ret = min_vol_result['return']
        
        # Get maximum return portfolio  
        max_ret_result = self.optimize('max_return', constraints)
        max_ret = max_ret_result['return']
        
        # Generate target returns along efficient frontier
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_data = []
        
        for target in target_returns:
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.expected_returns) - target}
            ]
            
            # Apply bounds from constraints
            no_shorting = constraints.get('no_shorting', True)
            max_weight = constraints.get('max_weight', 0.4)
            
            if no_shorting:
                bounds = [(0, max_weight) for _ in range(self.n_assets)]
            else:
                bounds = [(-max_weight, max_weight) for _ in range(self.n_assets)]
                
            x0 = np.ones(self.n_assets) / self.n_assets
            
            result = minimize(
                fun=lambda x: np.sqrt(x.T @ self.cov_matrix @ x),
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                vol = np.sqrt(result.x.T @ self.cov_matrix @ result.x)
                sharpe = (target - self.risk_free_rate) / vol if vol > 0 else 0
                frontier_data.append({
                    'return': target,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    'weights': result.x
                })
        
        return pd.DataFrame(frontier_data)