import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class PortfolioCalculators:
    """Various portfolio calculation utilities for portfolio analysis and
    risk/return metrics.
    """

    @staticmethod
    def calculate_diversification_ratio(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """Calculate portfolio diversification ratio.

        Diversification ratio = sum(w_i * sigma_i) / portfolio_volatility
        where sigma_i is the individual asset volatility (sqrt of diagonal of cov).
        """
        weighted_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sum_weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
        return float(sum_weighted_vol / weighted_vol) if weighted_vol > 0 else 0.0

    @staticmethod
    def calculate_value_at_risk(returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR) for the portfolio at given confidence level.

        Returns the percentile of portfolio returns corresponding to the confidence level.
        """
        portfolio_returns = returns.dot(weights)
        return float(np.percentile(portfolio_returns, confidence_level * 100))

    @staticmethod
    def calculate_cvar(returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR / Expected Shortfall)."""
        portfolio_returns = returns.dot(weights)
        var = PortfolioCalculators.calculate_value_at_risk(returns, weights, confidence_level)
        tail = portfolio_returns[portfolio_returns <= var]
        return float(tail.mean()) if not tail.empty else float(var)

    @staticmethod
    def calculate_max_drawdown(returns: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate maximum drawdown for the portfolio returns series."""
        portfolio_returns = returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return float(drawdown.min())

    @staticmethod
    def calculate_sortino_ratio(returns: pd.DataFrame, weights: np.ndarray, risk_free_rate: float = 0.02, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return).

        Uses annualized return and annualized downside deviation.
        """
        # portfolio_returns = returns.dot(weights)
        # excess_returns = portfolio_returns - target_return
        # downside_returns = excess_returns[excess_returns < 0]
        # if downside_returns.empty:
        #     return float('inf')

        # downside_deviation = float(np.std(downside_returns, ddof=1) * np.sqrt(252))
        # annualized_return = float(portfolio_returns.mean() * 252)
        # if downside_deviation == 0:
        #     return float('inf')
        # return float((annualized_return - risk_free_rate) / downside_deviation)
        portfolio_returns = returns.dot(weights)
    
        # Use geometric annualization for return
        annualized_return = PortfolioCalculators.geometric_annualized_return(portfolio_returns)
        
        # Calculate downside deviation from daily returns
        excess_returns = portfolio_returns - target_return/252  # Convert target to daily
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty or len(downside_returns) < 10:
            return float('inf') if annualized_return > risk_free_rate else 0.0

        downside_deviation = float(np.std(downside_returns, ddof=1) * np.sqrt(252))
        
        if downside_deviation == 0:
            return float('inf')
        
        return float((annualized_return - risk_free_rate) / downside_deviation)

    @staticmethod
    def calculate_beta_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02) -> Tuple[float, float]:
        """Calculate portfolio beta and alpha relative to a benchmark using sample estimators (ddof=1)."""
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if aligned.shape[0] == 0:
            return 0.0, 0.0

        port = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

        # use pandas sample covariance/variance (ddof=1) for consistency
        covariance = float(port.cov(bench, ddof=1))
        bench_variance = float(bench.var(ddof=1))
        beta = float(covariance / bench_variance) if bench_variance != 0 else 0.0

        port_annual = float(port.mean() * 252)
        bench_annual = float(bench.mean() * 252)
        alpha = float(port_annual - (risk_free_rate + beta * (bench_annual - risk_free_rate)))
        return beta, alpha

    @staticmethod
    def calculate_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error (annualized std of active returns) using sample std (ddof=1)."""
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if aligned.shape[0] == 0:
            return 0.0
        active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        return float(active.std(ddof=1) * np.sqrt(252))

    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio = annualized active return / tracking error."""
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if aligned.shape[0] == 0:
            return 0.0
        active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        avg_active = float(active.mean() * 252)
        te = PortfolioCalculators.calculate_tracking_error(portfolio_returns, benchmark_returns)
        return float(avg_active / te) if te != 0 else 0.0

    @staticmethod
    def calculate_calmar_ratio(returns: pd.DataFrame, weights: np.ndarray, lookback_years: int = 3) -> float:
        """Calculate Calmar ratio = annualized return / max drawdown over lookback."""
        portfolio_returns = returns.dot(weights)
        if len(portfolio_returns) > lookback_years * 252:
            portfolio_returns = portfolio_returns[-lookback_years * 252:]
        annual_return = float(portfolio_returns.mean() * 252)
        max_drawdown = PortfolioCalculators.calculate_max_drawdown(returns, weights)
        return float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else float('inf')

    @staticmethod
    def calculate_ulcer_performance_index(returns: pd.DataFrame, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Ulcer Performance Index (UPI).

        UPI = (annualized excess return) / ulcer_index where ulcer_index is RMS of drawdowns.
        """
        portfolio_returns = returns.dot(weights)
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ulcer_index = float(np.sqrt(np.mean(drawdown ** 2)))
        excess_return = float(portfolio_returns.mean() * 252 - risk_free_rate)
        return float(excess_return / ulcer_index) if ulcer_index != 0 else float('inf')

    @staticmethod
    def calculate_treynor_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Treynor ratio = (annualized excess return) / beta."""
        beta, _ = PortfolioCalculators.calculate_beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate)
        excess_return = float(portfolio_returns.mean() * 252 - risk_free_rate)
        return float(excess_return / beta) if beta != 0 else float('inf')

    @staticmethod
    def monte_carlo_simulation(expected_returns: pd.Series, cov_matrix: pd.DataFrame, n_portfolios: int = 10000,
                               risk_free_rate: float = 0.02, no_shorting: bool = True, max_weight: float = 1.0) -> pd.DataFrame:
        """Run Monte Carlo simulation to generate random portfolios.

        Returns a DataFrame with columns: 'weights', 'return', 'volatility', 'sharpe_ratio'.
        """
        n_assets = len(expected_returns)
        results: List[Dict] = []

        for _ in range(n_portfolios):
            if no_shorting:
                weights = np.random.random(n_assets)
                weights /= weights.sum()
                # Enforce max weight by resampling until constraint satisfied
                if max_weight < 1.0:
                    while np.any(weights > max_weight):
                        weights = np.random.random(n_assets)
                        weights /= weights.sum()
            else:
                weights = np.random.normal(0, 1, n_assets)
                weights /= np.abs(weights).sum()

            # Use the annualized metrics function
            port_return, port_vol = PortfolioCalculators.portfolio_metrics(weights, expected_returns, cov_matrix)
            sharpe = float((port_return - risk_free_rate) / port_vol) if port_vol > 0 else 0.0

            results.append({
                'weights': weights,
                'return': port_return,
                'volatility': port_vol,
                'sharpe_ratio': sharpe
            })

        return pd.DataFrame(results)
    
    @staticmethod
    def geometric_annualized_return(returns: pd.Series) -> float:
        """Return CAGR-style annualized return for a return series (decimal)."""
        r = returns.dropna()
        if r.empty or len(r) < 5:
            return 0.0
        # return float((1 + r).prod() ** (252 / len(r)) - 1)
        # For backtest returns (already daily)
        days = len(r)
        total_return = (1 + r).prod() - 1
        
        if days > 0:
            # Geometric annualization
            annualized = (1 + total_return) ** (252 / days) - 1
            return float(np.clip(annualized, -0.50, 5.0))
        return 0.0

    @staticmethod
    def portfolio_metrics(weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float]:
        """Calculate portfolio metrics - handles both numpy and pandas inputs."""
        # Convert to numpy arrays for consistent computation
        if hasattr(expected_returns, 'values'):
            er_array = expected_returns.values
        else:
            er_array = expected_returns
            
        if hasattr(cov_matrix, 'values'):
            cov_array = cov_matrix.values
        else:
            cov_array = cov_matrix
        
        port_return = float(np.dot(weights, er_array))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_array, weights))))
        return port_return, port_vol