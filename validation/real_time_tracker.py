# validation/real_time_tracker.py - UPDATED VERSION
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import time

class RealTimePortfolioTracker:
    """Handles real-time portfolio tracking and order execution with Alpaca."""
    
    def __init__(self, api_key: str, secret_key: str, base_url='https://paper-api.alpaca.markets/'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/') + '/' 
        self.api = None
        self.portfolio_history = []
        self.active_strategies = {}
        self.setup_logging()
        self._initialize_api()
        
    def _initialize_api(self):
        """Initialize Alpaca API with proper error handling."""
        try:
            self.api = tradeapi.REST(
                self.api_key, 
                self.secret_key, 
                self.base_url, 
                api_version='v2'
            )
            # Test connection immediately
            account = self.api.get_account()
            self.logger.info(f"✅ Successfully connected to Alpaca. Account: {account.status}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Alpaca API: {e}")
            self.api = None
            return False
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def is_connected(self) -> bool:
        """Check if we have a valid connection to Alpaca."""
        if self.api is None:
            return False
        try:
            self.api.get_account()
            return True
        except:
            return False
    

    def _get_baseline_value(self) -> float:
        """Returns current equity as baseline for return tracking."""
        try:
            account = self.api.get_account()
            return float(account.equity)
        except Exception as e:
            self.logger.warning(f"Could not fetch baseline equity: {e}")
            return 0.0
        
    def execute_strategy_comparison(self, strategies_weights: Dict, initial_capital: float = 10000) -> Dict:
        """Execute multiple strategies simultaneously in paper trading."""
        if not self.is_connected():
            self.logger.error("Cannot execute strategies: Not connected to Alpaca")
            return {name: {'error': 'Not connected to Alpaca'} for name in strategies_weights.keys()}
        
        strategy_performance = {}
        
        for strategy_name, weights in strategies_weights.items():
            self.logger.info(f"Executing strategy: {strategy_name}")
            try:
                performance = self._execute_single_strategy(strategy_name, weights, initial_capital)
                strategy_performance[strategy_name] = performance
                # self.active_strategies[strategy_name] = performance
                self.active_strategies[strategy_name] = {
                    **performance,
                    "initial_value": performance.get("initial_value", self._get_baseline_value())
                }
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Failed to execute {strategy_name}: {e}")
                strategy_performance[strategy_name] = {'error': str(e)}
        
        return strategy_performance
    
    def _execute_single_strategy(self, strategy_name: str, weights: Dict, capital: float) -> Dict:
        """Execute a single strategy and track its performance."""
        if not self.is_connected():
            return {'status': 'failed', 'error': 'Not connected to Alpaca'}
            
        try:
            # Close any existing positions for this strategy
            self._close_existing_positions()
            
            # Calculate and execute orders
            orders = self._calculate_orders(weights, capital)
            executed_orders = self._place_orders(orders)
            
            initial_value = self._get_portfolio_value()
            # Record initial state
            initial_state = {
                'timestamp': datetime.now(),
                'strategy': strategy_name,
                'positions': self._get_current_positions(),
                'portfolio_value': initial_value,
                'cash': float(self.api.get_account().cash),
                'orders': executed_orders
            }
            self.portfolio_history.append(initial_state)
            
            return {
                'status': 'executed',
                'orders_placed': len(executed_orders),
                'initial_value': initial_value,
                'timestamp': initial_state['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"Strategy execution failed for {strategy_name}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _calculate_orders(self, weights: Dict, capital: float) -> List[Dict]:
        """Calculate orders based on target weights and capital."""
        orders = []
        
        # get fresh account snapshot for sizing
        try:
            account = self.api.get_account()
            available_cash = float(account.cash)
            buying_power = float(getattr(account, "buying_power", available_cash))
        except Exception:
            available_cash = capital
            buying_power = capital
        
        for ticker, weight in weights.items():
            if weight > 0.001:  # Only positions > 0.1%
                try:
                    # Get current price
                    current_price = self._get_current_price(ticker)
                    
                    if current_price > 0:
                        # Cap notional by available buying power
                        notional_amount = min(capital * weight, buying_power * 0.99)
                        shares = int(notional_amount / current_price)
                        
                        if shares > 0:
                            orders.append({
                                'symbol': ticker,
                                'qty': shares,
                                'side': 'buy',
                                'type': 'market',
                                'time_in_force': 'day'
                            })
                except Exception as e:
                    self.logger.warning(f"Could not calculate order for {ticker}: {e}")
        
        return orders
    
    def _place_orders(self, orders: List[Dict]) -> List:
            """Place orders with Alpaca, with retry/backoff for transient failures (e.g. wash trade)."""
            executed_orders = []
            
            for order in orders:
                max_attempts = 5
                attempt = 0
                while attempt < max_attempts:
                    attempt += 1
                    try:
                        # Add small delay to avoid rate limiting
                        time.sleep(0.3)
                        
                        # double-check available cash before submitting
                        try:
                            acct = self.api.get_account()
                            cash = float(acct.cash)
                        except Exception:
                            cash = None
                        
                        # if cash known and insufficient, skip
                        if cash is not None and cash < order['qty'] * max(0.01, self._get_current_price(order['symbol'])):
                            self.logger.warning(f"Skipping order for {order['symbol']}: insufficient cash ({cash})")
                            break
                        
                        result = self.api.submit_order(
                            symbol=order['symbol'],
                            qty=order['qty'],
                            side=order['side'],
                            type=order['type'],
                            time_in_force=order['time_in_force']
                        )
                        executed_orders.append(result)
                        self.logger.info(f"Order executed: {order['symbol']} {order['qty']} shares")
                        break  # success -> break retry loop
                        
                    except Exception as e:
                        msg = str(e).lower()
                        # If Alpaca indicates a wash trade or similar transient policy, wait and retry
                        if 'wash' in msg or 'wash trade' in msg or 'potential wash' in msg or 'temporary' in msg:
                            backoff = min(60 * attempt, 300)  # up to 5 minutes
                            self.logger.warning(f"Transient Alpaca error for {order['symbol']} (attempt {attempt}): {e}. Backing off {backoff}s and retrying.")
                            time.sleep(backoff)
                            continue
                        # For other errors, log and stop retrying this order
                        self.logger.error(f"Order failed for {order['symbol']}: {e}")
                        break
            
            return executed_orders
    
    def _close_existing_positions(self):
        """Close all existing positions and wait until Alpaca acknowledges closure."""
        try:
            self.api.close_all_positions()
            self.logger.info("Closed all existing positions (requested).")
            # Poll until positions list is empty or timeout
            timeout = 30  # seconds
            poll_interval = 1.0
            elapsed = 0.0
            while elapsed < timeout:
                try:
                    positions = self.api.list_positions()
                    if not positions:
                        self.logger.info("Positions cleared.")
                        return
                except Exception as e:
                    self.logger.warning(f"Error checking positions after close_all_positions: {e}")
                time.sleep(poll_interval)
                elapsed += poll_interval
            self.logger.warning("Timed out waiting for positions to clear; continuing anyway.")
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")

    def _get_current_price(self, ticker: str) -> float:
        """Get current price for a ticker using Alpaca's data API."""
        def get_price():
            try:
                latest_trade = self.api.get_latest_trade(ticker)
                return float(latest_trade.price)
            except:
                bars = self.api.get_bars(ticker, "1Min", limit=1)
                if hasattr(bars, "df") and not bars.df.empty:
                    return float(bars.df["close"].iloc[-1])
                elif hasattr(bars, "__iter__"):
                    bars_list = list(bars)
                    if len(bars_list) > 0:
                        return float(bars_list[-1].c)
                return 0.0
        
        try:
            return self._retry_api_call(get_price)
        except Exception as e:
            self.logger.warning(f"Could not get price for {ticker} after retries: {e}")
            return 0.0
    
    def _get_current_positions(self) -> Dict:
        """Get current positions."""
        try:
            positions = self.api.list_positions()
            return {pos.symbol: float(pos.market_value) for pos in positions}
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def _get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        try:
            account = self.api.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            self.logger.error(f"Error getting portfolio value: {e}")
            return 0.0
    
    def monitor_strategy_performance(self, strategy_name: str) -> Dict:
        """Monitor current performance of a strategy."""
        if not self.is_connected():
            return {'error': 'Not connected to Alpaca'}
            
        try:
            positions = self._get_current_positions()
            portfolio_value = self._get_portfolio_value()
            cash = float(self.api.get_account().cash)
            
            # Calculate current returns
            initial_value = (
                self.active_strategies.get(strategy_name, {}).get("initial_value")
                or next(
                    (h['portfolio_value'] for h in self.portfolio_history 
                    if h.get('strategy') == strategy_name), portfolio_value
                )
            )
            
            current_return = 0.0
            if initial_value > 0:
                current_return = (portfolio_value - initial_value) / initial_value
            
            # Record current state to history
            current_state = {
                'timestamp': datetime.now(),
                'strategy': strategy_name,
                'positions': positions,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'current_return': current_return
            }
            self.portfolio_history.append(current_state)
            
            return {
                'strategy': strategy_name,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions': positions,
                'current_return': current_return,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error monitoring strategy {strategy_name}: {e}")
            return {'error': str(e)}


    def get_daily_performance_history(self, strategy_name: str) -> pd.DataFrame:
        """Get historical daily performance for a strategy."""
        try:
            # Filter portfolio history for this strategy
            strategy_history = [
                entry for entry in self.portfolio_history 
                if entry.get('strategy') == strategy_name
            ]
            
            if not strategy_history:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(strategy_history)
            df = df[['timestamp', 'portfolio_value']].copy()
            df['daily_return'] = df['portfolio_value'].pct_change()
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Could not get performance history for {strategy_name}: {e}")
            return pd.DataFrame()
    def _retry_api_call(self, func, max_retries=3, delay=1):
        """Retry API calls with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay * (2 ** attempt))