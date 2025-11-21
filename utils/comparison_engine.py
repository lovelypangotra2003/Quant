import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from .calculators import PortfolioCalculators

class PortfolioComparisonEngine:
    """
    Engine to generate comprehensive portfolio comparisons with key metrics.
    """
    
    def __init__(self, returns_data: pd.DataFrame, expected_returns: pd.Series, 
                 cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02):
        self.returns_data = returns_data
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        
    def calculate_portfolio_metrics(self, weights: np.ndarray, portfolio_name: str) -> Dict:
        """Calculate comprehensive metrics for a portfolio."""
        # Basic metrics
        port_return, port_vol = PortfolioCalculators.portfolio_metrics(
            weights, self.expected_returns, self.cov_matrix
        )
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Risk metrics from DAILY returns for accuracy
        portfolio_daily_returns = self.returns_data.dot(weights)
        
        # Use geometric annualization for return-based metrics
        geometric_annual_return = PortfolioCalculators.geometric_annualized_return(portfolio_daily_returns)

        var_95 = PortfolioCalculators.calculate_value_at_risk(self.returns_data, weights, 0.05)
        cvar_95 = PortfolioCalculators.calculate_cvar(self.returns_data, weights, 0.05)
        max_drawdown = PortfolioCalculators.calculate_max_drawdown(self.returns_data, weights)
        # diversification = PortfolioCalculators.calculate_diversification_ratio(weights, self.cov_matrix)
        sortino = PortfolioCalculators.calculate_sortino_ratio(self.returns_data, weights, self.risk_free_rate)
        
        # Monte Carlo simulation for forward-looking estimates
        mc_stats = self._monte_carlo_forward_look(weights, n_simulations=1000)
        
        # Loss probabilities
        loss_probabilities = self._calculate_loss_probabilities(portfolio_daily_returns)
        
        return {
            'portfolio_name': portfolio_name,
            'sharpe_ratio': sharpe,
            'volatility': port_vol,
            'expected_return': port_return,
            'geometric_annual_return': geometric_annual_return,  # Geometric annualized
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sortino_ratio': sortino,
            # 'diversification_ratio': diversification,
            'mc_90_low': mc_stats['percentile_5'],
            'mc_90_high': mc_stats['percentile_95'],
            'loss_prob_1_year': loss_probabilities['1_year_loss_prob'],
            'loss_prob_5_percent': loss_probabilities['prob_below_5'],
            'weights': weights,
            'concentration': self._calculate_concentration(weights),
            'return_consistency_check': f"Sample: {port_return:.1%}, Geometric: {geometric_annual_return:.1%}"
        }
    
    def _monte_carlo_forward_look(self, weights: np.ndarray, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation for forward-looking return distribution."""
        # Use historical parameters for simulation
        portfolio_returns = self.returns_data.dot(weights)
        mean_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Simulate annual returns
        simulated_returns = np.random.normal(mean_return, volatility, n_simulations)
        
        return {
            'percentile_5': np.percentile(simulated_returns, 5),
            'percentile_25': np.percentile(simulated_returns, 25),
            'percentile_75': np.percentile(simulated_returns, 75),
            'percentile_95': np.percentile(simulated_returns, 95),
            'mean': np.mean(simulated_returns),
            'std': np.std(simulated_returns)
        }
    
    def _calculate_loss_probabilities(self, portfolio_returns: pd.Series) -> Dict:
        """Calculate various loss probabilities."""
        # Probability of negative return in 1 year
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        if annual_vol > 0:
            z_score_0 = (0 - annual_return) / annual_vol
            prob_negative_1yr = stats.norm.cdf(z_score_0)
        else:
            prob_negative_1yr = 0.5 if annual_return == 0 else (0 if annual_return > 0 else 1)
        
        # Probability of return below -5% in 1 year
        z_score_5 = (-0.05 - annual_return) / annual_vol if annual_vol > 0 else 0
        prob_below_5 = stats.norm.cdf(z_score_5)
        
        return {
            '1_year_loss_prob': prob_negative_1yr,
            'prob_below_5': prob_below_5
        }
    
    def _calculate_concentration(self, weights: np.ndarray) -> float:
        """Calculate portfolio concentration (Herfindahl index)."""
        return np.sum(weights ** 2)
    
    # def generate_all_portfolios(self, constraints: Dict) -> Dict:
    #     """Generate all portfolio strategies for comparison."""
    #     portfolios = {}
        
    #     # 1. Equal-weighted portfolio
    #     equal_weights = np.ones(self.n_assets) / self.n_assets
    #     portfolios['equal_weight'] = self.calculate_portfolio_metrics(equal_weights, "Equal-Weighted")
        
    #     # 2. Markowitz optimal (max Sharpe)
    #     from models.markowitz import MarkowitzOptimizer
    #     markowitz_opt = MarkowitzOptimizer(self.expected_returns, self.cov_matrix, self.risk_free_rate)
    #     sharpe_result = markowitz_opt.optimize('max_sharpe', constraints)
    #     portfolios['markowitz_sharpe'] = self.calculate_portfolio_metrics(
    #         sharpe_result['weights'], "Markowitz (Optimal)"
    #     )
        
    #     # 3. Minimum volatility portfolio
    #     min_vol_result = markowitz_opt.optimize('min_volatility', constraints)
    #     portfolios['min_volatility'] = self.calculate_portfolio_metrics(
    #         min_vol_result['weights'], "Minimum Volatility"
    #     )
        
    #     # 4. Maximum diversification portfolio
    #     max_div_weights = self._max_diversification_portfolio(constraints)
    #     portfolios['max_diversification'] = self.calculate_portfolio_metrics(
    #         max_div_weights, "Max Diversification"
    #     )
        
    #     # 5. Risk parity portfolio (optional)
    #     try:
    #         risk_parity_weights = self._risk_parity_portfolio(constraints)
    #         portfolios['risk_parity'] = self.calculate_portfolio_metrics(
    #             risk_parity_weights, "Risk Parity"
    #         )
    #     except:
    #         pass
        
    #     return portfolios


    def generate_all_portfolios(self, constraints: Dict, regularization: float = 0.1) -> Dict:
        """Generate all portfolio strategies with proper regularization."""
        portfolios = {}
        
        # 1. Equal-weighted portfolio (baseline)
        equal_weights = np.ones(self.n_assets) / self.n_assets
        portfolios['equal_weight'] = self.calculate_portfolio_metrics(equal_weights, "Equal-Weighted")
        
        # 2. Traditional Markowitz (for comparison) - WITH REGULARIZATION
        from models.markowitz import EnhancedMarkowitzOptimizer
        markowitz_opt = EnhancedMarkowitzOptimizer(self.expected_returns, self.cov_matrix, self.risk_free_rate)
        
         # Use STRONGER regularization
        strong_regularization = min(regularization + 0.1, 0.3)

        # Apply regularization to ALL optimizations
        sharpe_result = markowitz_opt.optimize('max_sharpe', constraints, regularization=strong_regularization)
        portfolios['markowitz_regularized'] = self.calculate_portfolio_metrics(
            sharpe_result['weights'], f"Markowitz (Reg={regularization})"
        )
        
        # 3. Bootstrap robust optimization (most reliable)
        # bootstrap_result = markowitz_opt.robust_optimization(
        #     self.returns_data, constraints, regularization=regularization
        # )
        # portfolios['bootstrap_robust'] = self.calculate_portfolio_metrics(
        #     bootstrap_result['weights'], "Bootstrap Robust"
        # )
        
        # # 4. Maximum diversification (often more stable)
        # try:
        #     max_div_result = markowitz_opt.optimize('max_diversification', constraints, regularization=regularization)
        #     portfolios['max_diversification'] = self.calculate_portfolio_metrics(
        #         max_div_result['weights'], "Maximum Diversification"
        #     )
        # except:
        #     # Fallback
        #     portfolios['max_diversification'] = portfolios['bootstrap_robust']

        # 3. Minimum volatility portfolio
        min_vol_result = markowitz_opt.optimize('min_volatility', constraints, regularization=regularization)
        portfolios['min_volatility'] = self.calculate_portfolio_metrics(
            min_vol_result['weights'], "Minimum Volatility"
        )
        
        return portfolios

    def _create_markowitz_sharpe(self, constraints: Dict) -> np.ndarray:
        from models.markowitz import EnhancedMarkowitzOptimizer
        optimizer = EnhancedMarkowitzOptimizer(self.expected_returns, self.cov_matrix, self.risk_free_rate)
        result = optimizer.optimize('max_sharpe', constraints)
        return result['weights']
    
    def _max_diversification_portfolio(self, constraints: Dict) -> np.ndarray:
        """Calculate maximum diversification portfolio."""
        from scipy.optimize import minimize
        
        # Objective: maximize diversification ratio
        def objective(weights):
            div_ratio = PortfolioCalculators.calculate_diversification_ratio(weights, self.cov_matrix)
            return -div_ratio  # Minimize negative diversification
        
        # Constraints
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds
        if constraints.get('no_shorting', True):
            bounds = [(0, constraints.get('max_weight', 1.0)) for _ in range(self.n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(self.n_assets)]
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return result.x if result.success else np.ones(self.n_assets) / self.n_assets
    
    def _risk_parity_portfolio(self, constraints: Dict) -> np.ndarray:
        """Calculate risk parity portfolio."""
        from scipy.optimize import minimize
        
        # Risk contributions
        volatilities = np.sqrt(np.diag(self.cov_matrix))
        inverse_vol_weights = 1 / volatilities
        risk_parity_weights = inverse_vol_weights / np.sum(inverse_vol_weights)
        
        return risk_parity_weights
    
    def create_comparison_table(self, portfolios: Dict) -> pd.DataFrame:
        """Create a comprehensive comparison table."""
        table_data = []
        
        for portfolio in portfolios.values():
            table_data.append({
                'Portfolio': portfolio['portfolio_name'],
                'Sharpe Ratio': f"{portfolio['sharpe_ratio']:.2f}",
                'Volatility': f"{portfolio['volatility']:.1%}",
                'Expected Return': f"{portfolio['expected_return']:.1%}",
                'Max Drawdown': f"{portfolio['max_drawdown']:.1%}",
                'Sortino Ratio': f"{portfolio['sortino_ratio']:.2f}",
                '90% CI Range': f"{portfolio['mc_90_low']:.1%} to {portfolio['mc_90_high']:.1%}",
                'Loss Probability': f"{portfolio['loss_prob_5_percent']:.0%} chance of < -5%",
                'Diversification': f"{portfolio['diversification_ratio']:.2f}",
                'Concentration': f"{portfolio['concentration']:.3f}"
            })
        
        return pd.DataFrame(table_data)