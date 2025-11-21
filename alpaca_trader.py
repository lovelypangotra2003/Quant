# alpaca_trader.py
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st
from datetime import datetime, timedelta
import time

class AlpacaTrader:
    """Handles Alpaca API integration for paper trading simulation."""
    
    def __init__(self):
        self.api = None
        self.connected = False
        self.initialize_api()
    
    def initialize_api(self):
        """Initialize Alpaca API connection."""
        try:
            # You'll need to set these in your Streamlit secrets or environment variables
            api_key = st.secrets.get("ALPACA_API_KEY", "your_paper_api_key")
            api_secret = st.secrets.get("ALPACA_SECRET_KEY", "your_paper_secret")
            base_url = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            
            self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            
            # Test connection
            account = self.api.get_account()
            self.connected = True
            st.success(f"✅ Connected to Alpaca Paper Trading - Account: {account.account_number}")
            
        except Exception as e:
            st.warning(f"⚠️ Alpaca connection failed: {str(e)}")
            st.info("Please set ALPACA_API_KEY, ALPACA_SECRET_KEY in your Streamlit secrets")
            self.connected = False
    
    def get_account_info(self) -> Dict:
        """Get current account information."""
        if not self.connected:
            return {}
        
        try:
            account = self.api.get_account()
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': int(account.day_trade_count)
            }
        except Exception as e:
            st.error(f"Error getting account info: {e}")
            return {}
    
    def execute_portfolio_rebalance(self, target_weights: Dict, total_amount: float = None) -> Dict:
        """
        Execute portfolio rebalancing trades based on target weights.
        
        Parameters:
        -----------
        target_weights : Dict
            Dictionary of {ticker: target_weight}
        total_amount : float
            Total portfolio value to use (defaults to available buying power)
        """
        if not self.connected:
            return {'success': False, 'message': 'Not connected to Alpaca'}
        
        try:
            # Get current positions
            positions = self.api.list_positions()
            current_positions = {p.symbol: float(p.market_value) for p in positions}
            
            # Get total portfolio value
            account_info = self.get_account_info()
            if total_amount is None:
                total_amount = float(account_info.get('buying_power', 10000))
            
            # Calculate target dollar amounts
            target_amounts = {}
            for ticker, weight in target_weights.items():
                target_amounts[ticker] = total_amount * weight
            
            # Calculate trades needed
            orders_placed = []
            for ticker, target_amount in target_amounts.items():
                current_amount = current_positions.get(ticker, 0)
                trade_amount = target_amount - current_amount
                
                if abs(trade_amount) > 10:  # Minimum trade size
                    side = 'buy' if trade_amount > 0 else 'sell'
                    qty = self._calculate_quantity(ticker, abs(trade_amount))
                    
                    if qty > 0:
                        order = self.api.submit_order(
                            symbol=ticker,
                            qty=qty,
                            side=side,
                            type='market',
                            time_in_force='day'
                        )
                        orders_placed.append({
                            'symbol': ticker,
                            'side': side,
                            'quantity': qty,
                            'order_id': order.id
                        })
            
            return {
                'success': True,
                'orders_placed': orders_placed,
                'total_amount': total_amount,
                'target_weights': target_weights
            }
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _calculate_quantity(self, ticker: str, dollar_amount: float) -> int:
        """Calculate number of shares based on dollar amount."""
        try:
            # Get current price
            bars = self.api.get_bars(ticker, '1Min', limit=1)
            current_price = bars[ticker][0].c
            
            # Calculate quantity (round down to whole shares)
            quantity = int(dollar_amount / current_price)
            return quantity
        except:
            return 0
    
    def get_portfolio_performance(self) -> Dict:
        """Get current portfolio performance metrics."""
        if not self.connected:
            return {}
        
        try:
            positions = self.api.list_positions()
            account = self.api.get_account()
            
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            
            position_details = []
            total_pnl = 0
            
            for position in positions:
                pnl = float(position.unrealized_pl)
                total_pnl += pnl
                position_details.append({
                    'symbol': position.symbol,
                    'qty': int(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'unrealized_pl': pnl,
                    'unrealized_plpc': float(position.unrealized_plpc)
                })
            
            return {
                'portfolio_value': portfolio_value,
                'cash': cash,
                'total_unrealized_pl': total_pnl,
                'positions': position_details,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error getting portfolio performance: {e}")
            return {}
    
    def get_historical_portfolio_value(self, days: int = 30) -> pd.DataFrame:
        """Get historical portfolio values for performance tracking."""
        if not self.connected:
            return pd.DataFrame()
        
        try:
            # This would require storing historical data externally
            # For now, return empty DataFrame
            return pd.DataFrame()
        except:
            return pd.DataFrame()

class PaperTradingSimulator:
    """Simulates paper trading without actual API calls for demonstration."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []
    
    def execute_rebalance(self, target_weights: Dict, prices: Dict) -> Dict:
        """Simulate portfolio rebalancing."""
        total_value = self.current_capital
        
        # Calculate current positions value
        for ticker, position in self.positions.items():
            if ticker in prices:
                total_value += position['shares'] * prices[ticker]
        
        # Calculate target shares
        target_shares = {}
        for ticker, weight in target_weights.items():
            if ticker in prices and prices[ticker] > 0:
                target_value = total_value * weight
                target_shares[ticker] = target_value / prices[ticker]
        
        # Execute trades
        trades = []
        for ticker, target_share_count in target_shares.items():
            current_shares = self.positions.get(ticker, {}).get('shares', 0)
            share_diff = target_share_count - current_shares
            
            if abs(share_diff) >= 1:  # Minimum 1 share
                trade_value = share_diff * prices[ticker]
                
                if trade_value <= self.current_capital or share_diff < 0:
                    # Update position
                    if ticker not in self.positions:
                        self.positions[ticker] = {'shares': 0, 'avg_price': 0}
                    
                    if share_diff > 0:  # Buy
                        new_avg_price = (
                            (self.positions[ticker]['shares'] * self.positions[ticker]['avg_price'] + 
                             share_diff * prices[ticker]) / 
                            (self.positions[ticker]['shares'] + share_diff)
                        )
                        self.positions[ticker]['avg_price'] = new_avg_price
                        self.current_capital -= trade_value
                    else:  # Sell
                        self.current_capital += abs(trade_value)
                    
                    self.positions[ticker]['shares'] = target_share_count
                    
                    trades.append({
                        'symbol': ticker,
                        'shares': share_diff,
                        'price': prices[ticker],
                        'value': trade_value,
                        'type': 'BUY' if share_diff > 0 else 'SELL'
                    })
        
        # Record trade
        self.trade_history.extend(trades)
        
        # Update portfolio history
        portfolio_value = self.current_capital
        for ticker, position in self.positions.items():
            if ticker in prices:
                portfolio_value += position['shares'] * prices[ticker]
        
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'cash': self.current_capital
        })
        
        return {
            'success': True,
            'trades': trades,
            'portfolio_value': portfolio_value
        }
    
    def get_portfolio_summary(self, current_prices: Dict) -> Dict:
        """Get current portfolio summary."""
        total_value = self.current_capital
        position_details = []
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                market_value = position['shares'] * current_prices[ticker]
                total_value += market_value
                unrealized_pl = market_value - (position['shares'] * position['avg_price'])
                
                position_details.append({
                    'symbol': ticker,
                    'shares': position['shares'],
                    'avg_price': position['avg_price'],
                    'current_price': current_prices[ticker],
                    'market_value': market_value,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_plpc': unrealized_pl / (position['shares'] * position['avg_price']) if position['shares'] > 0 else 0
                })
        
        return {
            'portfolio_value': total_value,
            'cash': self.current_capital,
            'total_unrealized_pl': sum(p['unrealized_pl'] for p in position_details),
            'positions': position_details,
            'total_return': (total_value - self.initial_capital) / self.initial_capital
        }