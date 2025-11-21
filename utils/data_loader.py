import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict
import streamlit as st

class DataLoader:
    """Handles data loading and preprocessing for portfolio optimization."""
    
    def __init__(self):
        self.available_tickers = []
        
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def load_ticker_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical price data for given tickers."""
        try:
            data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, timeout = 30)
            returns = data['Close'].pct_change().dropna()
            return returns
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
  
    def calculate_metrics(self, returns: pd.DataFrame, shrinkage: float = 0.5, risk_free_rate: float = 0.02) -> Dict:
        """Calculate financial metrics with CONSISTENT annualization."""
        if returns.empty:
            return {}
    
        returns = returns.dropna(axis=1, thresh=len(returns)*0.8)
    
        
        # DAILY returns (input)
        daily_returns = returns
        
        # CONSISTENT ANNUALIZATION METHODOLOGY
        # Method 1: Geometric annualized returns (most accurate for returns)
        # geometric_annual_returns = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
        arithmetic_annual_returns = daily_returns.mean() * 252
        # Method 2: Sample-based annualized covariance (standard for volatility)
        annual_cov_matrix = daily_returns.cov() * 252
        
        # Apply shrinkage to covariance only (returns shrinkage is problematic)
        if shrinkage > 0:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf(shrinkage=shrinkage).fit(daily_returns.values)
                shrunk_cov = lw.covariance_ * 252.0
            except Exception:
                # Fallback to simple diagonal shrinkage
                n_assets = len(returns.columns)
                average_variance = np.mean(np.diag(annual_cov_matrix))
                target_cov = np.eye(n_assets) * average_variance
                shrunk_cov = (1 - shrinkage) * annual_cov_matrix + shrinkage * target_cov
        else:
            shrunk_cov = annual_cov_matrix
        
        # Use geometric returns (more realistic) with mild shrinkage toward equal-weight
        market_return = arithmetic_annual_returns.mean()
        shrunk_returns = (1 - shrinkage) * arithmetic_annual_returns + (shrinkage) * market_return
        
        # Cap individual asset returns at realistic levels
        # Individual stocks realistically: -30% to +40% annual expected
        shrunk_returns = np.clip(shrunk_returns, -0.20, 0.30)

        # Ensure proper pandas Series/DataFrame types
        shrunk_returns = pd.Series(shrunk_returns, index=returns.columns)
        shrunk_cov = pd.DataFrame(shrunk_cov, index=returns.columns, columns=returns.columns)

        # Realistic volatility bounds
        volatilities = np.sqrt(np.diag(shrunk_cov))
        volatilities = np.clip(volatilities, 0.05, 0.80)  # 5% to 80% annual vol
        
        sharpe_ratios = (shrunk_returns - risk_free_rate) / (volatilities + 1e-12)
        sharpe_ratios = np.clip(sharpe_ratios, -3, 3)  # Cap Sharpe ratios

        # REALITY CHECK - warn about unrealistic metrics
        if shrunk_returns.max() >0.40:  # >40% annual return
            st.warning(f"⚠️ Unrealistic returns detected (max: {shrunk_returns.max():.1%}). Check data quality.")
        
        return {
            'expected_returns': shrunk_returns,  # GEOMETRIC ANNUALIZED
            'covariance_matrix': shrunk_cov,     # ANNUALIZED COVARIANCE
            'sharpe_ratios': sharpe_ratios,
            'volatilities': volatilities,
            'shrinkage_applied': shrinkage,
            'annualization_method': 'realistic_bounded'
        }
    # def calculate_metrics(self, returns: pd.DataFrame, shrinkage: float = 0.5, risk_free_rate: float = 0.02) -> Dict:
    #     """Calculate financial metrics with shrinkage to prevent overfitting."""
    #     if returns.empty:
    #         return {}
        
    #     # Traditional calculations (what you have now)
    #     # annualized_returns = returns.mean() * 252
    #     # annualized_cov = returns.cov() * 252

    #     # DAILY returns first
    #     daily_returns = returns
    #     days = len(daily_returns)
    #     if days <= 0:
    #         return {}
        
    #     # Then annualize properly
    #     annualized_returns = (1 + daily_returns).prod(axis = 0) ** (252.0 / days) - 1.0
    #     # OR more simply:
    #     # annualized_returns = daily_returns.mean() * 252  # This is approximate but standard
        
    #     annualized_cov = daily_returns.cov() * 252.0
        
    #     # SHRINKAGE: Pull extreme values toward the average
    #     if shrinkage > 0:
    #         # Shrink returns toward equal-weighted market return
    #         # market_return = annualized_returns.mean()
    #         # shrunk_returns = (1 - shrinkage) * annualized_returns + shrinkage * market_return
            
    #         # # Shrink covariance matrix toward diagonal (less correlation)
    #         # n_assets = len(returns.columns)
    #         # average_vol = np.mean(np.diag(annualized_cov))
    #         # target_cov = np.eye(n_assets) * average_vol
    #         # shrunk_cov = (1 - shrinkage) * annualized_cov + shrinkage * target_cov
    #         try:
    #             # sklearn's LedoitWolf expects returns (not prices)
    #             from sklearn.covariance import LedoitWolf
    #             lw = LedoitWolf().fit(daily_returns.values)
    #             shrunk_cov = lw.covariance_ * 252.0
    #         except Exception:
    #             # fallback to simple diagonal shrinkage toward average variance
    #             n_assets = len(returns.columns)
    #             average_variance = np.mean(np.diag(annualized_cov))
    #             target_cov = np.eye(n_assets) * average_variance
    #             shrunk_cov = (1 - shrinkage) * annualized_cov + shrinkage * target_cov
    #         #  Shrink returns toward equal-weighted market return (James-Stein style)
    #         market_return = annualized_returns.mean()
    #         shrunk_returns = (1 - shrinkage) * annualized_returns + shrinkage * market_return
    #     else:
    #         shrunk_returns = annualized_returns
    #         shrunk_cov = annualized_cov
        
    #     sharpe_ratios = (shrunk_returns - risk_free_rate) / (np.sqrt(np.diag(shrunk_cov))+ 1e-12)
        
    #     # print("Mean returns:\n", returns.mean().head())
    #     # print("Covariance matrix:\n", returns.cov().head())
    #     # print("Shrinked covariance:\n", shrunk_cov.head())

    #     return {
    #         'expected_returns': shrunk_returns,
    #         'covariance_matrix': shrunk_cov,
    #         'sharpe_ratios': sharpe_ratios,
    #         'volatilities': np.sqrt(np.diag(shrunk_cov)),
    #         'shrinkage_applied': shrinkage
    #     }
        
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """Validate ticker symbols and return valid ones."""
        valid_tickers = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if info.get('regularMarketPrice') is not None:
                    valid_tickers.append(ticker)
            except:
                st.warning(f"Ticker {ticker} not found or invalid.")
        return valid_tickers