# validation/simulated_forward_tester.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import yfinance as yf

# class SimulatedForwardTestingEngine:
#     """Simulates forward testing of portfolio strategies using historical data."""
    
#     def __init__(self):
#         self.test_results = {}
#         self.logger = logging.getLogger(__name__)
    
#     def run_forward_test(self, strategies_weights: Dict, test_period_days: int = 30, 
#                         initial_capital: float = 10000, tickers: List[str] = None) -> Dict:
#         """
#         Run simulated forward test using historical data.
        
#         Args:
#             strategies_weights: Dictionary of strategy names to weight dictionaries
#             test_period_days: Number of days to simulate
#             initial_capital: Initial capital for each strategy
#             tickers: List of tickers to get data for
        
#         Returns:
#             Dictionary with performance results for each strategy
#         """
#         self.logger.info(f"Starting simulated forward test for {len(strategies_weights)} strategies")
        
#         if not tickers:
#             # Extract tickers from strategies
#             all_tickers = set()
#             for weights in strategies_weights.values():
#                 all_tickers.update(weights.keys())
#             tickers = list(all_tickers)
        
#         # Get historical data for the test period
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=test_period_days + 10)  # Buffer for lookback
        
#         try:
#             price_data = self._get_price_data(tickers, start_date, end_date)
#             returns_data = price_data.pct_change().dropna()
            
#             # Run simulation for each strategy
#             performance_results = {}
#             for strategy_name, weights in strategies_weights.items():
#                 performance = self._simulate_strategy_performance(
#                     strategy_name, weights, returns_data, initial_capital
#                 )
#                 performance_results[strategy_name] = performance
            
#             # Store results
#             test_id = f"sim_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#             self.test_results[test_id] = {
#                 'strategies': list(strategies_weights.keys()),
#                 'period_days': test_period_days,
#                 'initial_capital': initial_capital,
#                 'performance': performance_results
#             }
            
#             return performance_results
            
#         except Exception as e:
#             self.logger.error(f"Error in simulated forward test: {e}")
#             return {name: {'error': str(e)} for name in strategies_weights.keys()}
    
#     def _get_price_data(self, tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
#         """Get price data for simulation."""
#         try:
#             # Convert to string dates for yfinance
#             start_str = start_date.strftime('%Y-%m-%d')
#             end_str = end_date.strftime('%Y-%m-%d')
            
#             data = yf.download(tickers, start=start_str, end=end_str, auto_adjust=True)
#             return data['Close']
#         except Exception as e:
#             self.logger.error(f"Error getting price data: {e}")
#             raise
    
#     def _simulate_strategy_performance(self, strategy_name: str, weights: Dict, 
#                                      returns_data: pd.DataFrame, initial_capital: float) -> Dict:
#         """Simulate strategy performance over the test period."""
#         try:
#             # Create weight vector aligned with returns data columns
#             weight_vector = np.array([weights.get(ticker, 0) for ticker in returns_data.columns])
            
#             # Calculate portfolio returns
#             portfolio_returns = returns_data.dot(weight_vector)
            
#             # Calculate cumulative performance
#             cumulative_returns = (1 + portfolio_returns).cumprod()
#             portfolio_values = initial_capital * cumulative_returns
            
#             # Calculate metrics
#             total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
#             # annualized_return = total_return * (252 / len(portfolio_returns))
#             # total_days = len(portfolio_returns)
#             # annualized_return = (1 + total_return) ** (252 / total_days) - 1
#             total_days = len(portfolio_returns)
#             if total_days > 0:
#                 # Method 1: Direct geometric annualization
#                 annualized_return = (1 + total_return) ** (252 / total_days) - 1
                
#                 # Method 2: Using daily returns (more precise for volatile returns)
#                 daily_geometric_mean = (1 + portfolio_returns).prod() ** (1 / total_days) - 1
#                 annualized_return_alt = (1 + daily_geometric_mean) ** 252 - 1
                
#                 # Use the more conservative estimate
#                 annualized_return = min(annualized_return, annualized_return_alt)
#             else:
#                 annualized_return = 0.0
#             # Volatility and risk metrics
#             volatility = portfolio_returns.std() * np.sqrt(252)
#             sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
#             # Drawdown calculation
#             running_max = portfolio_values.expanding().max()
#             drawdowns = (portfolio_values - running_max) / running_max
#             max_drawdown = drawdowns.min()
            
#             # Sortino ratio (downside risk)
#             downside_returns = portfolio_returns[portfolio_returns < 0]
#             downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
#             sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
            
#             # Win rate
#             positive_days = (portfolio_returns > 0).sum()
#             win_rate = positive_days / len(portfolio_returns)
            
#             return {
#                 'strategy_name': strategy_name,
#                 'actual_return': float(total_return),
#                 'annualized_return': float(annualized_return),
#                 'volatility': float(volatility),
#                 'actual_sharpe': float(sharpe_ratio),
#                 'max_drawdown': float(max_drawdown),
#                 'sortino_ratio': float(sortino_ratio),
#                 'win_rate': float(win_rate),
#                 'total_days': int(len(portfolio_returns)),
#                 'positive_days': int(positive_days),
#                 'final_value': float(portfolio_values.iloc[-1]),
#                 'portfolio_values': portfolio_values.tolist(),
#                 'daily_returns': portfolio_returns.tolist()
#             }
            
#         except Exception as e:
#             self.logger.error(f"Error simulating strategy {strategy_name}: {e}")
#             return {
#                 'strategy_name': strategy_name,
#                 'error': str(e),
#                 'actual_return': 0.0,
#                 'annualized_return': 0.0,
#                 'volatility': 0.0,
#                 'actual_sharpe': 0.0,
#                 'max_drawdown': 0.0,
#                 'sortino_ratio': 0.0,
#                 'win_rate': 0.0,
#                 'total_days': 0,
#                 'positive_days': 0,
#                 'final_value': initial_capital
#             }
    
#     def get_test_history(self) -> Dict:
#         """Get history of all simulated tests."""
#         return self.test_results

class WalkForwardValidationEngine:
    """Walk-forward validation with multiple rolling windows."""
    
    def __init__(self):
        self.test_results = {}
        self.logger = logging.getLogger(__name__)

    def _get_price_data(self, tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get price data for simulation."""
        try:
            # Convert to string dates for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            self.logger.info(f"Downloading price data for {len(tickers)} tickers from {start_str} to {end_str}")
            data = yf.download(tickers, start=start_str, end=end_str, auto_adjust=True, progress=False)
            
            if data.empty:
                raise ValueError("No data returned from yfinance")
                
            return data['Close']
        except Exception as e:
            self.logger.error(f"Error getting price data: {e}")
            raise
    
    # def run_walk_forward_validation(self, strategies_weights: Dict, 
    #                             train_days: int = 252,
    #                             test_days: int = 63,
    #                             step_days: int = 21,
    #                             initial_capital: float = 10000,
    #                             tickers: List[str] = None) -> Dict:
    #     """
    #     Run walk-forward validation across multiple rolling windows.
    #     """
    #     if not tickers:
    #         all_tickers = set()
    #         for weights in strategies_weights.values():
    #             all_tickers.update(weights.keys())
    #         tickers = list(all_tickers)
        
    #     # validate input windows
    #     if train_days <= 0 or test_days <= 0 or step_days <= 0:
    #         self.logger.error("Invalid window sizes: train_days, test_days and step_days must be > 0")
    #         return {name: {'error': 'invalid window sizes'} for name in strategies_weights.keys()}
        
    #     # Get full historical data
    #     end_date = datetime.now()
    #     start_date = end_date - timedelta(days=train_days + test_days + step_days * 10)
        
    #     try:
    #         price_data = self._get_price_data(tickers, start_date, end_date)
    #         returns_data = price_data.pct_change().dropna()
            
    #         # Generate rolling windows
    #         windows = self._generate_rolling_windows(returns_data, train_days, test_days, step_days)
            
    #         if not windows:
    #             self.logger.warning("No rolling windows generated (insufficient data for the chosen window sizes).")
    #             return {name: {'error': 'insufficient data for windows'} for name in strategies_weights.keys()}
            
    #         # Test each strategy across all windows
    #         strategy_performance = {}
    #         for strategy_name, weights in strategies_weights.items():
    #             performance = self._evaluate_strategy_walk_forward(
    #                 strategy_name, weights, windows, initial_capital
    #             )
    #             strategy_performance[strategy_name] = performance
            
    #         return strategy_performance
            
    #     except Exception as e:
    #         self.logger.error(f"Error in walk-forward validation: {e}")
    #         return {name: {'error': str(e)} for name in strategies_weights.keys()}


    def run_walk_forward_validation(self, strategies_weights: Dict, 
                                train_days: int = 504,
                                test_days: int = 126,
                                step_days: int = 63,
                                initial_capital: float = 10000,
                                tickers: List[str] = None,
                                desired_windows: int  = 20) -> Dict:
        """
        Run walk-forward validation across multiple rolling windows.
        """
        if not tickers:
            all_tickers = set()
            for weights in strategies_weights.values():
                all_tickers.update(weights.keys())
            # Make ordering deterministic for reproducibility
            tickers = sorted(list(all_tickers))
        
        # validate input windows
        if train_days <= 0 or test_days <= 0 or step_days <= 0:
            self.logger.error("Invalid window sizes: train_days, test_days and step_days must be > 0")
            return {name: {'error': 'invalid window sizes'} for name in strategies_weights.keys()}
        
        # Get full historical data
        end_date = datetime.now() - timedelta(days=90)
        # start_date = end_date - timedelta(days=train_days + test_days + step_days * 10)
        # compute required history based on desired windows (use buffer)
        required_days = train_days + test_days + step_days * max(10, desired_windows)
        # add small calendar buffer (1.1x) to account for weekends/holidays
        start_date = end_date - timedelta(days=int(required_days * 1.1))
        
        try:
            price_data = self._get_price_data(tickers, start_date, end_date)
            returns_data = price_data.pct_change().dropna()
            self.logger.info(f"returns_data shape: rows={len(returns_data)}, cols={len(returns_data.columns)}")
            
             # If the initial fetch did not return enough trading rows for even a single window,
            # retry with a much larger lookback (fallback).
            needed_rows = train_days + test_days + step_days * max(5, desired_windows//2)
            if len(returns_data) < needed_rows:
                self.logger.warning(
                    f"Insufficient trading rows ({len(returns_data)}) for requested windows; retrying with extended history"
                )
                # Try an extended lookback (~11 years)
                alt_start = end_date - timedelta(days=4000)
                self.logger.info(f"Retrying fetch from {alt_start.date()} to {end_date.date()}")
                price_data = self._get_price_data(tickers, alt_start, end_date)
                returns_data = price_data.pct_change().dropna()
                self.logger.info(f"After retry returns_data rows={len(returns_data)}")
            # Generate rolling windows
            windows = self._generate_rolling_windows(returns_data, train_days, test_days, step_days)
            
            if not windows:
                self.logger.warning("No rolling windows generated (insufficient data for the chosen window sizes).")
                return {name: {'error': 'insufficient data for windows'} for name in strategies_weights.keys()}
            
            # Test each strategy across all windows
            strategy_performance = {}
            for strategy_name, weights in strategies_weights.items():
                performance = self._evaluate_strategy_walk_forward(
                    strategy_name, weights, windows, initial_capital
                )
                strategy_performance[strategy_name] = performance
            
            return strategy_performance
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward validation: {e}")
            return {name: {'error': str(e)} for name in strategies_weights.keys()}
    
    # def _generate_rolling_windows(self, returns_data: pd.DataFrame, 
    #                             train_days: int, test_days: int, step_days: int) -> List[Dict]:
    #     """Generate rolling train/test windows."""
    #     windows = []
    #     total_days = len(returns_data)
        
    #     start_idx = 0
    #     while start_idx + train_days + test_days <= total_days:
    #         train_start = start_idx
    #         train_end = start_idx + train_days
    #         test_start = train_end
    #         test_end = test_start + test_days
            
    #         windows.append({
    #             'train_returns': returns_data.iloc[train_start:train_end],
    #             'test_returns': returns_data.iloc[test_start:test_end],
    #             'window_id': len(windows)
    #         })
            
    #         start_idx += step_days
        
    #     self.logger.info(f"Generated {len(windows)} rolling windows")
    #     return windows

    def _generate_rolling_windows(self, returns_data: pd.DataFrame, 
                            train_days: int, test_days: int, step_days: int) -> List[Dict]:
        """Generate rolling train/test windows with proper indexing."""
        windows = []
        total_days = len(returns_data)

        # Calculate expected window count BEFORE generating
        expected_windows = max(1, (total_days - train_days - test_days) // step_days + 1)
        self.logger.info(f"Window generation: total={total_days}d, train={train_days}d, test={test_days}d, step={step_days}d")
        self.logger.info(f"Expected windows: {expected_windows}")
        
        # Ensure we have enough data
        if total_days < train_days + test_days:
            self.logger.warning(f"Insufficient data: {total_days} days for train {train_days} + test {test_days}")
            return windows
        
        # start_idx = 0
        # window_count = 0
        
        # while start_idx + train_days + test_days <= total_days:
        #     train_start = start_idx
        #     train_end = start_idx + train_days
        #     test_start = train_end
        #     test_end = min(test_start + test_days, total_days)  # Don't exceed data bounds
            
        #     # Ensure we have valid test period
        #     if test_end - test_start < test_days * 0.5:  # At least 50% of desired test period
        #         break
                
        #     windows.append({
        #         'train_returns': returns_data.iloc[train_start:train_end],
        #         'test_returns': returns_data.iloc[test_start:test_end],
        #         'window_id': window_count,
        #         'train_period': f"{returns_data.index[train_start].date()} to {returns_data.index[train_end-1].date()}",
        #         'test_period': f"{returns_data.index[test_start].date()} to {returns_data.index[test_end-1].date()}"
        #     })
            
        #     window_count += 1
        #     start_idx += step_days
        # Deterministic index-based generation to ensure expected window count
        # window_count = 0
        # last_start = total_days - (train_days + test_days)
        # if last_start < 0:
        #     return windows

        # for start_idx in range(0, last_start + 1, step_days):
        #     train_start = start_idx
        #     train_end = start_idx + train_days
        #     test_start = train_end
        #     test_end = test_start + test_days
        #     # clamp final window test_end
        #     if test_end > total_days:
        #         test_end = total_days

        #     windows.append({
        #         'train_returns': returns_data.iloc[train_start:train_end],
        #        'test_returns': returns_data.iloc[test_start:test_end],
        #         'window_id': window_count,
        #         'train_period': f"{returns_data.index[train_start].date()} to {returns_data.index[train_end-1].date()}",
        #         'test_period': f"{returns_data.index[test_start].date()} to {returns_data.index[test_end-1].date()}"
        #     })
        #     window_count += 1
        
        # self.logger.info(f"Generated {len(windows)} rolling windows (train: {train_days}d, test: {test_days}d, step: {step_days}d)")
        # logging.info(f"✅ Generated {len(windows)} actual windows")
        # return windows
        # Generate windows with deterministic stepping
        window_id = 0
        start_idx = 0
        
        while start_idx + train_days + test_days <= total_days:
            train_start = start_idx
            train_end = train_start + train_days
            test_start = train_end
            test_end = test_start + test_days
            
            # Verify we have enough data in both periods
            if (train_end - train_start == train_days and 
                test_end - test_start >= test_days * 0.8):  # Allow 80% of test period
                
                train_returns = returns_data.iloc[train_start:train_end]
                test_returns = returns_data.iloc[test_start:test_end]
                
                # Ensure no NaN issues
                if (len(train_returns) >= train_days * 0.9 and 
                    len(test_returns) >= test_days * 0.8 and
                    not train_returns.isna().all().any() and
                    not test_returns.isna().all().any()):
                    
                    windows.append({
                        'train_returns': train_returns,
                        'test_returns': test_returns,
                        'window_id': window_id,
                        'train_period': f"{returns_data.index[train_start].date()} to {returns_data.index[train_end-1].date()}",
                        'test_period': f"{returns_data.index[test_start].date()} to {returns_data.index[test_end-1].date()}"
                    })
                    window_id += 1
            
            start_idx += step_days
        
        self.logger.info(f"✅ Generated {len(windows)} valid windows")
        return windows
    
    # def _evaluate_strategy_walk_forward(self, strategy_name: str, weights: Dict,
    #                               windows: List[Dict], initial_capital: float) -> Dict:
    #     """Evaluate strategy across all walk-forward windows."""
    #     window_results = []
        
    #     for window in windows:
    #         try:
    #             # Align weights with test returns columns
    #             weight_vector = np.array([weights.get(ticker, 0) 
    #                                     for ticker in window['test_returns'].columns])
                
    #             # Calculate test period performance
    #             test_returns = window['test_returns'].dot(weight_vector)
                
    #             # FIX: Use geometric compounding for more accurate returns
    #             # cumulative_return = (1 + test_returns).prod() - 1
    #             # portfolio_value = initial_capital * (1 + cumulative_return)
                
    #             # # Calculate metrics for this window - FIXED CALCULATIONS
    #             # total_days = len(test_returns)
                
    #             # # More robust annualized return calculation
    #             # if total_days > 0:
    #             #     # Method 1: Geometric annualization
    #             #     annualized_return = (1 + cumulative_return) ** (252 / total_days) - 1
                    
    #             #     # Method 2: Using daily returns for more precision
    #             #     daily_returns_mean = test_returns.mean()
    #             #     daily_returns_std = test_returns.std()
                    
    #             #     # Compound annual growth rate (more accurate)
    #             #     cagr = (portfolio_value / initial_capital) ** (252 / total_days) - 1
                    
    #             #     # Use the most conservative estimate
    #             #     annualized_return = min(annualized_return, cagr)
                    
    #             #     # Cap unrealistic returns
    #             #     annualized_return = np.clip(annualized_return, -0.99, 5.0)  # Cap at 500% max
    #             # else:
    #             #     annualized_return = 0.0
                    
    #             # FIX: Use simple period return for short test periods
    #             cumulative_return = (1 + test_returns).prod() - 1
                
    #             # For annualization, be conservative
    #             if len(test_returns) >= 60:  # Only annualize if sufficient data
    #                 annualized_return = (1 + cumulative_return) ** (252 / len(test_returns)) - 1
    #             else:
    #                 annualized_return = cumulative_return  # Use period return for short tests
                
    #             # CAP unrealistic returns
    #             annualized_return = np.clip(annualized_return, -0.90, 5.0)  # Max 500% return
            
    #             # Volatility calculation
    #             volatility = test_returns.std() * np.sqrt(252)
                
    #             # Sharpe ratio with realistic bounds
    #             sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    #             sharpe_ratio = np.clip(sharpe_ratio, -5, 10)  # Realistic bounds
                
    #             # Drawdown calculation
    #             cumulative_returns = (1 + test_returns).cumprod()
    #             running_max = cumulative_returns.expanding().max()
    #             drawdowns = (cumulative_returns - running_max) / running_max
    #             max_drawdown = drawdowns.min()
                
    #             window_results.append({
    #                 'window_id': window['window_id'],
    #                 'total_return': float(cumulative_return),
    #                 'annualized_return': float(annualized_return),
    #                 'volatility': float(volatility),
    #                 'sharpe_ratio': float(sharpe_ratio),
    #                 'max_drawdown': float(max_drawdown),
    #                 'test_days': len(test_returns),
    #                 # 'final_value': float(portfolio_value)
    #             })
                
    #         except Exception as e:
    #             self.logger.warning(f"Window {window['window_id']} failed for {strategy_name}: {e}")
    #             continue
        
    #     # Aggregate results across all windows
    #     if not window_results:
    #         return {'error': 'No successful windows'}
        
    #     results_df = pd.DataFrame(window_results)
        
    #     # More robust aggregation with outlier handling
    #     def robust_mean(series):
    #         """Calculate mean with outlier removal."""
    #         Q1 = series.quantile(0.25)
    #         Q3 = series.quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 1.5 * IQR
    #         filtered = series[(series >= lower_bound) & (series <= upper_bound)]
    #         return filtered.mean() if len(filtered) > 0 else series.mean()
        
    #     return {
    #         'strategy_name': strategy_name,
    #         'window_count': len(window_results),
    #         'mean_annualized_return': float(robust_mean(results_df['annualized_return'])),
    #         'median_annualized_return': float(results_df['annualized_return'].median()),
    #         'std_annualized_return': float(results_df['annualized_return'].std()),
    #         'mean_sharpe_ratio': float(robust_mean(results_df['sharpe_ratio'])),
    #         'median_sharpe_ratio': float(results_df['sharpe_ratio'].median()),
    #         'win_rate': float((results_df['total_return'] > 0).mean()),
    #         'consistency_score': float(self._calculate_consistency_score(results_df)),
    #         'all_window_results': window_results,
    #         'percentile_25_return': float(results_df['annualized_return'].quantile(0.25)),
    #         'percentile_75_return': float(results_df['annualized_return'].quantile(0.75)),
    #         'min_return': float(results_df['annualized_return'].min()),
    #         'max_return': float(results_df['annualized_return'].max())
    #     }
    
    def _evaluate_strategy_walk_forward(self, strategy_name: str, weights: Dict,
                              windows: List[Dict], initial_capital: float) -> Dict:
        """Evaluate strategy across all walk-forward windows - CORRECTED ANNUALIZATION."""
        window_results = []
        rejected_reasons = []

        for window in windows:
            try:
                # Align weights with test returns columns
                weight_vector = np.array([weights.get(ticker, 0) 
                                        for ticker in window['test_returns'].columns])
                
                # Calculate test period performance
                test_returns = window['test_returns'].dot(weight_vector)
                
                # Annualize ONLY if we have reasonable data
                test_days = len(test_returns)
                if test_days < 21:
                    rejected_reasons.append((window['window_id'], 'too_short'))
                    continue

                # Period return (NOT annualized yet)
                total_period_return = (1 + test_returns).prod() - 1

                # Volatility (annualized)
                volatility = test_returns.std() * np.sqrt(min(252, test_days)) 
                volatility = float(np.clip(volatility, 0.005, 2.0))
                
                if test_days >= 63:  # At least 3 months
                    # Conservative annualization: (1 + R)^(252/days) - 1
                    annualized_return = (1 + total_period_return) ** (252 / test_days) - 1
                    # Cap unrealistic values
                    annualized_return = float(np.clip(annualized_return, -0.50, 0.50))

                else:
                    # For short periods, use period return scaled to annual (but capped)
                    annualized_return = total_period_return * (252 / test_days)
                    annualized_return = float(np.clip(annualized_return, -0.75, 0.75))

                # REALITY CHECKS
                if abs(annualized_return) > 1.5:  
                    self.logger.warning(f"EXTREME return {annualized_return:.1%} in window {window['window_id']}")
                    rejected_reasons.append((window['window_id'],'extreme_return'))
                    
                if volatility < 0 or np.isnan(volatility):
                    self.logger.warning(f"Bad volatility {volatility} in window {window['window_id']}; skipping")
                    rejected_reasons.append((window['window_id'], 'bad_vol'))
                    continue
                    
                # # volatility = np.clip(volatility, 0.01, 1.0)
                
                # Sharpe ratio
                sharpe_ratio = annualized_return / volatility if volatility > 0.005 else 0
                sharpe_ratio = np.clip(sharpe_ratio, -5, 10)
                
                # # Drawdown
                # cumulative_returns = (1 + test_returns).cumprod()
                # running_max = cumulative_returns.expanding().max()
                # drawdowns = (cumulative_returns - running_max) / running_max
                # max_drawdown = drawdowns.min()
                
                window_results.append({
                    'window_id': window['window_id'],
                    'total_return': float(total_period_return),
                    'annualized_return': float(annualized_return),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    # 'max_drawdown': float(max_drawdown),
                    'test_days': int(test_days)
                })
                
            except Exception as e:
                self.logger.warning(f"Window {window['window_id']} failed for {strategy_name}: {e}")
                rejected_reasons.append((window.get('window_id', -1), f'exception:{e}'))
                continue
        
        if not window_results:
            return {'error': 'No successful windows'}
        
        # min_windows_required = max(3, len(windows) * 0.3)  # At least 30% of windows
        # if len(window_results) < min_windows_required:
        #     return {'error': f'Insufficient valid windows: {len(window_results)}/{len(windows)}'}
        
        # At least 30% of windows (integer)
        min_windows_required = max(3, int(np.ceil(len(windows) * 0.3)))
        if len(window_results) < min_windows_required:
            return {'error': f'Insufficient valid windows: {len(window_results)}/{len(windows)}',
                    'accepted_windows': len(window_results),
                    'generated_windows': len(windows),
                    'rejected_reasons_sample': rejected_reasons[:10]}

        results_df = pd.DataFrame(window_results)
        
        # Remove outliers before aggregation
        Q1 = results_df['annualized_return'].quantile(0.25)
        Q3 = results_df['annualized_return'].quantile(0.75)
        IQR = Q3 - Q1
        mask = (results_df['annualized_return'] >= Q1 - 1.5*IQR) & (results_df['annualized_return'] <= Q3 + 1.5*IQR)
        filtered_df = results_df[mask]
        
        # Attach metadata about generation/filters
        meta = {
            'generated_windows': len(windows),
            'accepted_windows': len(window_results),
            'windows_after_outlier_removal': len(filtered_df),
            'rejected_reasons_sample': rejected_reasons[:10]
        }


        return {
            'strategy_name': strategy_name,
            'window_count': len(window_results),
            'mean_annualized_return': float(filtered_df['annualized_return'].mean()),
            'median_annualized_return': float(filtered_df['annualized_return'].median()),
            'std_annualized_return': float(filtered_df['annualized_return'].std()),
            'median_sharpe_ratio': float(filtered_df['sharpe_ratio'].median()),
            'win_rate': float((results_df['total_return'] > 0).mean()),
            'consistency_score': float(self._calculate_consistency_score(filtered_df)),
            'all_window_results': window_results,
            'windows_after_outlier_removal': len(filtered_df),
            'generation_meta': meta 
        }

    def _calculate_consistency_score(self, results_df: pd.DataFrame) -> float:
        """Calculate strategy consistency across windows."""
        if len(results_df) < 2:
            return 0.0
        
        # Score based on positive returns frequency and low volatility of returns
        win_rate = (results_df['total_return'] > 0).mean()
        return_stability = 1 - (results_df['annualized_return'].std() / 
                            (abs(results_df['annualized_return'].mean()) + 1e-8))
        
        return 0.6 * win_rate + 0.4 * min(1.0, return_stability)