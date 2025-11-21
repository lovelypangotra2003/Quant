# validation/forward_tester.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List
import logging
from .real_time_tracker import RealTimePortfolioTracker

class ForwardTestingEngine:
    """Handles forward testing of portfolio strategies in real-time."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.tracker = RealTimePortfolioTracker(api_key, secret_key)
        self.test_results = {}
        self.logger = logging.getLogger(__name__)
    
    def run_forward_test(self, strategies_weights: Dict, test_period_days: int = 30, 
                        initial_capital: float = 10000) -> Dict:
        """
        Run forward test of multiple strategies over specified period.
        
        Args:
            strategies_weights: Dictionary of strategy names to weight dictionaries
            test_period_days: Number of days to run the test
            initial_capital: Initial capital for each strategy
        
        Returns:
            Dictionary with performance results for each strategy
        """
        self.logger.info(f"Starting forward test for {len(strategies_weights)} strategies")
        
        # Initial strategy execution
        initial_results = self.tracker.execute_strategy_comparison(
            strategies_weights, initial_capital
        )
        
        # Daily performance monitoring
        performance_history = self._monitor_performance(
            list(strategies_weights.keys()), test_period_days
        )
        
        # Calculate final metrics
        final_results = self._calculate_final_metrics(performance_history)
        
        # Store results
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.test_results[test_id] = {
            'strategies': list(strategies_weights.keys()),
            'period_days': test_period_days,
            'initial_capital': initial_capital,
            'performance': final_results,
            'history': performance_history
        }
        
        self.debug_forward_test_results(performance_history)

        return final_results
    
    def _monitor_performance(self, strategy_names: List[str], period_days: int) -> Dict:
        """Monitor strategy performance daily."""
        performance_data = {}
        start_date = datetime.now()
        
        for day in range(period_days):
            daily_performance = {}
            current_date = start_date + timedelta(days=day)
            
            self.logger.info(f"Monitoring day {day + 1}/{period_days}")

            # Add small delay to allow orders to settle
            if day > 0:
                time.sleep(2)
            
            for strategy_name in strategy_names:
                try:
                    # Get current performance
                    current_perf = self.tracker.monitor_strategy_performance(strategy_name)
                    
                    if 'error' not in current_perf:
                        # Calculate additional metrics
                        daily_metrics = self._calculate_daily_metrics(
                            strategy_name, current_perf, day
                        )
                        daily_performance[strategy_name] = daily_metrics
                    else:
                        self.logger.warning(f"Could not monitor {strategy_name}: {current_perf['error']}")
                        daily_performance[strategy_name] = {
                            'error': current_perf['error'],
                            'day': day,
                            'timestamp': datetime.now()
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error monitoring {strategy_name} on day {day}: {e}")
                    daily_performance[strategy_name] = {
                        'error': str(e),
                        'day': day,
                        'timestamp': datetime.now()
                    }
            
            performance_data[day] = {
                'date': current_date,
                'performance': daily_performance
            }
            
            # Wait until next day (for real daily monitoring)
            # if day < period_days - 1:
            #     self._wait_until_next_day()
        
        return performance_data
    
    def _calculate_daily_metrics(self, strategy_name: str, current_perf: Dict, day: int) -> Dict:
        """Calculate comprehensive daily performance metrics."""
        try:
            # Basic metrics
            portfolio_value = current_perf.get('portfolio_value', 0)
            current_return = current_perf.get('current_return', 0)
            positions = current_perf.get('positions', {})
            
            # Calculate additional metrics
            daily_volatility = self._calculate_daily_volatility(strategy_name)
            max_drawdown = self._calculate_current_drawdown(strategy_name, portfolio_value)
            sharpe_ratio = self._calculate_daily_sharpe(current_return, daily_volatility)
            
            return {
                'portfolio_value': portfolio_value,
                'daily_return': current_return,
                'positions': positions,
                'volatility': daily_volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'day': day,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error calculating daily metrics for {strategy_name}: {e}")
            return {'error': str(e)}
    
    def _calculate_daily_volatility(self, strategy_name: str) -> float:
        """Calculate daily volatility based on historical performance."""
        try:
            history = self.tracker.get_daily_performance_history(strategy_name)
            if len(history) > 1 and 'daily_return' in history.columns:
                returns = history['daily_return'].dropna()
                return float(returns.std()) if len(returns) > 0 else 0.0
            return 0.0
        except Exception as e:
            self.logger.warning(f"Could not calculate volatility for {strategy_name}: {e}")
            return 0.0
        
    def _calculate_current_drawdown(self, strategy_name: str, current_value: float) -> float:
        """Calculate current drawdown from peak."""
        try:
            history = self.tracker.get_daily_performance_history(strategy_name)
            if len(history) > 0 and 'portfolio_value' in history.columns:
                peak_value = history['portfolio_value'].max()
                current_drawdown = (current_value - peak_value) / peak_value
                return float(current_drawdown)
            return 0.0
        except Exception as e:
            self.logger.warning(f"Could not calculate drawdown for {strategy_name}: {e}")
            return 0.0
    
    def _calculate_daily_sharpe(self, daily_return: float, volatility: float) -> float:
        """Calculate daily Sharpe ratio."""
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        if volatility > 0:
            return (daily_return - risk_free_rate) / volatility
        return 0.0
    
    def _wait_until_next_day(self):
        """Wait until next trading day (simplified - in production, would align with market hours)."""
        # For simulation purposes, we'll just wait a short time
        # In real implementation, this would wait until next market open
        time.sleep(2)  # Reduced for demo purposes
    
    def _calculate_final_metrics(self, performance_history: Dict) -> Dict:
        """Calculate final performance metrics for all strategies."""
        final_metrics = {}
        
        # Extract all strategy names from history
        strategy_names = set()
        for day_data in performance_history.values():
            strategy_names.update(day_data['performance'].keys())
        
        for strategy_name in strategy_names:
            try:
                strategy_data = self._extract_strategy_history(performance_history, strategy_name)
                
                if len(strategy_data) > 0:
                    metrics = self._calculate_strategy_final_metrics(strategy_data)
                    metrics['strategy_name'] = strategy_name
                    final_metrics[strategy_name] = metrics
                else:
                    final_metrics[strategy_name] = {
                    'strategy_name': strategy_name,
                    'error': 'No data available',
                    'actual_return': 0.0,
                    'annualized_return': 0.0,
                    'volatility': 0.0,
                    'actual_sharpe': 0.0,
                    'max_drawdown': 0.0
                }
                    
            except Exception as e:
                self.logger.error(f"Error calculating final metrics for {strategy_name}: {e}")
                final_metrics[strategy_name] = {
                'strategy_name': strategy_name,
                'error': str(e),
                'actual_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'actual_sharpe': 0.0,
                'max_drawdown': 0.0
            }
        
        return final_metrics
    
    def _extract_strategy_history(self, performance_history: Dict, strategy_name: str) -> List[Dict]:
        """Extract performance history for a specific strategy."""
        strategy_data = []
        
        for day, day_data in performance_history.items():
            if strategy_name in day_data['performance']:
                strategy_perf = day_data['performance'][strategy_name]
                if 'error' not in strategy_perf:
                    strategy_data.append(strategy_perf)
        
        return strategy_data
    
    def _calculate_strategy_final_metrics(self, strategy_data: List[Dict]) -> Dict:
        """Calculate comprehensive final metrics for a strategy."""
        if not strategy_data or len(strategy_data) < 2:
            return {
                'error': 'Insufficient data',
                'actual_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'actual_sharpe': 0.0,
                'max_drawdown': 0.0,
                'sortino_ratio': 0.0,
                'win_rate': 0.0,
                'total_days': len(strategy_data),
                'positive_days': 0,
                'final_value': 0.0
            }
        
        try:
            df = pd.DataFrame(strategy_data)
            
            # Basic metrics
            if 'portfolio_value' in df.columns and len(df) > 1:
                initial_value = df['portfolio_value'].iloc[0]
                final_value = df['portfolio_value'].iloc[-1]
                
                if initial_value > 0:
                    total_return = (final_value - initial_value) / initial_value
                else:
                    total_return = 0.0
                
                # Use geometric annualization on daily returns for consistency
                daily_returns = df['daily_return'].dropna()
                if len(daily_returns) > 0:
                    from utils.calculators import PortfolioCalculators
                    annualized_return = PortfolioCalculators.geometric_annualized_return(daily_returns)
                else:
                    annualized_return = total_return * (252 / len(df)) if len(df) > 0 else 0.0
            else:
                total_return = 0.0
                annualized_return = 0.0
                final_value = 0.0
            
            # Volatility
            if 'daily_return' in df.columns and len(df) > 1:
                volatility = df['daily_return'].std() * np.sqrt(252)
            else:
                volatility = 0.0
            
            # Sharpe ratio
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
            
            # Risk metrics
            max_drawdown = df['max_drawdown'].min() if 'max_drawdown' in df.columns else 0.0
            sortino_ratio = self._calculate_sortino_ratio(df) if 'daily_return' in df.columns else 0.0
            
            # Consistency metrics
            if 'daily_return' in df.columns:
                positive_days = (df['daily_return'] > 0).sum()
                win_rate = positive_days / len(df) if len(df) > 0 else 0.0
            else:
                positive_days = 0
                win_rate = 0.0
            
            return {
                'actual_return': float(total_return),
                'annualized_return': float(annualized_return),
                'volatility': float(volatility),
                'actual_sharpe': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'sortino_ratio': float(sortino_ratio),
                'win_rate': float(win_rate),
                'total_days': int(len(df)),
                'positive_days': int(positive_days),
                'final_value': float(final_value)
            }
        except Exception as e:
            self.logger.error(f"Error in _calculate_strategy_final_metrics: {e}")
            return {
                'error': str(e),
                'actual_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'actual_sharpe': 0.0,
                'max_drawdown': 0.0,
                'sortino_ratio': 0.0,
                'win_rate': 0.0,
                'total_days': 0,
                'positive_days': 0,
                'final_value': 0.0
            }
    
    def _calculate_sortino_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sortino ratio from strategy data."""
        try:
            returns = df['daily_return']
            downside_returns = returns[returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_volatility = downside_returns.std() * np.sqrt(252)
            annualized_return = returns.mean() * 252
            
            return annualized_return / downside_volatility if downside_volatility > 0 else 0
        except Exception as e:
            self.logger.warning(f"Could not calculate Sortino ratio: {e}")
            return 0.0
    
    def get_test_history(self) -> Dict:
        """Get history of all forward tests."""
        return self.test_results
    
    def debug_forward_test_results(self, performance_history: Dict):
        """Debug method to inspect forward test results."""
        self.logger.info("=== DEBUG FORWARD TEST RESULTS ===")
        
        for day, day_data in performance_history.items():
            self.logger.info(f"Day {day}: {day_data['date']}")
            for strategy, perf in day_data['performance'].items():
                if 'error' in perf:
                    self.logger.warning(f"  {strategy}: ERROR - {perf['error']}")
                else:
                    self.logger.info(f"  {strategy}: Value=${perf.get('portfolio_value', 0):.2f}, Return={perf.get('daily_return', 0):.2%}")