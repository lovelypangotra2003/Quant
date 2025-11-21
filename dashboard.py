# dashboard.py - Enhanced Portfolio Optimizer with Real-world Validation
import streamlit as st
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import logging
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

from models.markowitz import EnhancedMarkowitzOptimizer
from models.monte_carlo import MonteCarloOptimizer
from utils.data_loader import DataLoader
from utils.calculators import PortfolioCalculators
from validation.real_time_tracker import RealTimePortfolioTracker
# from validation.simulated_forward_tester import SimulatedForwardTestingEngine
from validation.simulated_forward_tester import WalkForwardValidationEngine
from validation.forward_tester import ForwardTestingEngine
from validation.validation_metrics import RealWorldValidation
from utils.comparison_engine import PortfolioComparisonEngine

class EnhancedPortfolioOptimizationDashboard:
    """Enhanced dashboard with real-world validation capabilities."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.forward_tester = None
        self.validation_engine = RealWorldValidation()
        self.set_page_config()
        
    def set_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Portfolio Optimizer Pro",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Enhanced CSS with validation styles
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .validation-success {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .validation-warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .validation-error {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .comparison-table {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }
        .comparison-table table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        .comparison-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: center;
            border: 1px solid #bdc3c7;
            font-weight: bold;
        }
        .comparison-table td {
            padding: 12px;
            border: 1px solid #bdc3c7;
            text-align: center;
            color: #2c3e50;
        }
        .comparison-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .best-metric {
            background-color: #d4edda !important;
            border-left: 4px solid #28a745 !important;
            font-weight: bold;
        }
            border-radius: 0.5rem;
        .metric-card {
            font-size: 25px;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            color: #2c3e50;
            padding: 1rem;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def render_sidebar(self):
        """Render the sidebar with user inputs including Alpaca configuration."""
        st.sidebar.title("🎯 Portfolio Configuration")
        
        # Ticker input
        st.sidebar.subheader("Asset Selection")
        default_tickers = st.sidebar.text_area(
            "Enter tickers (comma-separated):",
            # "AAPL, MSFT, GOOGL, AMZN, TSLA, JNJ, JPM, V, PG, UNH",
            "SPY, AGG, VTI, BND, VXUS, IWM, XLE, XLU, XLP",
            height=100
        )
        tickers = [t.strip().upper() for t in default_tickers.split(',') if t.strip()]
        
        # Date range
        st.sidebar.subheader("Time Period")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime('2015-01-01'))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime('2024-01-01'))
            
        # Optimization parameters
        st.sidebar.subheader("Parameters")
        risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 
                                               value=2.0, step=0.1, min_value=0.0, max_value=10.0) / 100
        
        # Constraints
        st.sidebar.subheader("Constraints")
        no_shorting = st.sidebar.checkbox("No Short Selling", value=True)
        max_weight = st.sidebar.slider("Max Weight per Asset (%)", 
                                     10, 100, 40) / 100
        
        monte_carlo_portfolios = st.sidebar.slider("Monte Carlo Portfolios", 
                                                 5000, 30000, 10000)
        
        # Alpaca Configuration
        # st.sidebar.subheader("🔬 Real-world Validation")
        # enable_validation = st.sidebar.checkbox("Enable Paper Trading Validation", value=False)
        
        # alpaca_config = {}
        # if enable_validation:
        #     st.sidebar.info("Configure Alpaca for paper trading validation")
        #     alpaca_config['api_key'] = st.sidebar.text_input("Alpaca API Key", type="password")
        #     alpaca_config['secret_key'] = st.sidebar.text_input("Alpaca Secret Key", type="password")
        #     alpaca_config['test_days'] = st.sidebar.slider("Validation Period (days)", 7, 90, 30)
        #     alpaca_config['initial_capital'] = st.sidebar.number_input("Paper Trading Capital", 
        #                                                              value=10000, min_value=1000, step=1000)

        # In the render_sidebar method, update the validation section:
        st.sidebar.subheader("🔬 Real-world Validation")
        enable_validation = st.sidebar.checkbox("Enable Forward Testing Validation", value=True)

        # In render_sidebar(), add:
        st.sidebar.subheader("Optimization Parameters")
        regularization = st.sidebar.slider("Regularization Strength", 0.0, 0.5, 0.1, 0.05,help="Prevents overfitting: 0 = no regularization, 0.5 = strong regularization")

        # Then pass this to your optimization calls
        alpaca_config = {}
        if enable_validation:
            st.sidebar.info("Simulated forward testing using historical data")
            alpaca_config['test_days'] = st.sidebar.slider("Validation Period (days)", 7, 90, 30)
            alpaca_config['initial_capital'] = st.sidebar.number_input("Test Capital", 
                                                                    value=10000, min_value=1000, step=1000)
                
        return {
            'tickers': tickers,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'risk_free_rate': risk_free_rate,
            'monte_carlo_portfolios': monte_carlo_portfolios,
            'constraints': {
                'no_shorting': no_shorting,
                'max_weight': max_weight
            },
            'validation_enabled': enable_validation,
            'alpaca_config': alpaca_config,
            'regularization': regularization
        }

    # def setup_forward_testing(self, alpaca_config: Dict):
    #     """Initialize forward testing (simulated instead of real trading)."""
    #     try:
    #         self.forward_tester = SimulatedForwardTestingEngine()
    #         return True
    #     except Exception as e:
    #         st.error(f"Failed to initialize simulated forward testing: {e}")
    #         return False
    
    def setup_alpaca_integration(self, alpaca_config: Dict):
        """Initialize Alpaca integration for paper trading."""
        try:
            if alpaca_config.get('api_key') and alpaca_config.get('secret_key'):
                self.forward_tester = ForwardTestingEngine(
                    alpaca_config['api_key'], 
                    alpaca_config['secret_key']
                )
                return True
            else:
                st.warning("Please provide Alpaca API credentials for validation")
                return False
        except Exception as e:
            st.error(f"Failed to initialize Alpaca: {e}")
            return False
    
    # def render_validation_section(self, params: Dict, returns_data: pd.DataFrame, metrics: Dict):
    #     """Render real-world validation section with proper layout."""
    #     st.header("🔬 Real-world Validation")
        
    #     if not params['validation_enabled']:
    #         st.info("💡 Enable paper trading validation in the sidebar to test strategies in real-time")
    #         return
        
    #     # Initialize simulated testing
    #     if not self.forward_tester:
    #         if not self.setup_forward_testing(params['alpaca_config']):
    #             return
        
    #     # Create main validation container
    #     validation_container = st.container()
        
    #     # Action buttons
    #     run_validation = st.button("🚀 Run Simulated Forward Test", use_container_width=True)
        
    #     # Info box
    #     st.markdown("""
    #     <div class="validation-success">
    #     <h4>✅ Simulated Forward Testing:</h4>
    #     <ul>
    #         <li><strong>Realistic Simulation:</strong> Uses actual historical market data</li>
    #         <li><strong>No Real Trading:</strong> No API keys or paper trading required</li>
    #         <li><strong>Instant Results:</strong> Get validation results immediately</li>
    #         <li><strong>Accurate Metrics:</strong> Realistic returns, volatility, and risk metrics</li>
    #     </ul>
    #     </div>
    #     """, unsafe_allow_html=True)
        
    #     # Handle button action
    #     with validation_container:
    #         if run_validation:
    #             self._run_forward_validation(params, returns_data, metrics)

    # def _run_forward_validation(self, params: Dict, returns_data: pd.DataFrame, metrics: Dict):
    #     """Run forward validation with proper progress tracking."""
    #     # Progress tracking
    #     progress_bar = st.progress(0)
    #     status_text = st.empty()
        
    #     try:
    #         # Step 1: Initialize strategies
    #         status_text.text("📋 Initializing portfolio strategies...")
    #         progress_bar.progress(10)
            
    #         comparison_engine = PortfolioComparisonEngine(
    #             returns_data, metrics['expected_returns'], 
    #             metrics['covariance_matrix'], params['risk_free_rate']
    #         )
    #         portfolios = comparison_engine.generate_all_portfolios(params['constraints'])
            
    #         # Step 2: Prepare strategy weights
    #         status_text.text("⚖️ Calculating target allocations...")
    #         progress_bar.progress(30)
            
    #         strategy_weights = {}
    #         for portfolio_name, portfolio_data in portfolios.items():
    #             weights_dict = {}
    #             for i, ticker in enumerate(returns_data.columns):
    #                 if portfolio_data['weights'][i] > 0.001:
    #                     weights_dict[ticker] = portfolio_data['weights'][i]
    #             strategy_weights[portfolio_name] = weights_dict
            
    #         # Step 3: Execute paper trades
    #         status_text.text("💳 Executing paper trades via Alpaca...")
    #         progress_bar.progress(50)
            
    #         # Run forward test
    #         validation_results = self.forward_tester.run_forward_test(
    #             strategy_weights, 
    #             params['alpaca_config']['test_days'],
    #             params['alpaca_config']['initial_capital']
    #         )
            
    #         # Step 4: Monitor performance
    #         status_text.text("📈 Monitoring daily performance...")
    #         progress_bar.progress(80)
            
    #         # Step 5: Analyze results
    #         status_text.text("📊 Analyzing validation results...")
    #         progress_bar.progress(95)
            
    #         # Display results
    #         self.render_validation_results(validation_results, portfolios)
            
    #         progress_bar.progress(100)
    #         status_text.text("✅ Validation complete!")
            
    #     except Exception as e:
    #         st.error(f"❌ Validation failed: {e}")
    #         import traceback
    #         st.code(traceback.format_exc())


    def render_validation_section(self, params: Dict, returns_data: pd.DataFrame, metrics: Dict):
        """Render walk-forward validation section."""
        st.header("🔬 Walk-Forward Validation")
        
        if not params['validation_enabled']:
            st.info("💡 Enable validation in the sidebar to test strategies across multiple periods")
            return
        
        # Initialize walk-forward validation
        try:
            self.walk_forward_engine = WalkForwardValidationEngine()
        except Exception as e:
            st.error(f"❌ Failed to initialize walk-forward validation: {e}")
            return
        
        # Info box explaining walk-forward validation
        st.markdown("""
        <div class="validation-success">
        <h4>✅ Walk-Forward Validation Benefits:</h4>
        <ul>
            <li><strong>Multiple Testing Periods:</strong> Tests across 10-20+ different market regimes</li>
            <li><strong>Statistical Significance:</strong> Get mean, median, and standard deviation of performance</li>
            <li><strong>Detects Overfitting:</strong> Strategies that work by luck will fail consistently</li>
            <li><strong>Realistic Simulation:</strong> Mimics quarterly/yearly rebalancing</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Validation parameters - UPDATED FOR MORE WINDOWS
        st.subheader("Validation Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            train_days = st.slider("Training Period (days)", 504, 1008, 504, 
                                help="Length of training period for each window")
        with col2:
            test_days = st.slider("Test Period (days)", 63, 252, 126,
                                help="Length of testing period for each window")
        with col3:
            step_days = st.slider("Step Size (days)", 63, 126, 84,
                                help="How much to move the window forward each time")
        
        # Add data range info
        if not returns_data.empty:
            total_days = len(returns_data)
            estimated_windows = max(1, (total_days - train_days - test_days) // step_days + 1)
            st.info(f"📅 Data range: {total_days} days | Estimated windows: {estimated_windows}")
        
        run_validation = st.button("🚀 Run Walk-Forward Validation", use_container_width=True)
        
        if run_validation:
            self._run_walk_forward_validation(params, returns_data, metrics, train_days, test_days, step_days)

    def _run_walk_forward_validation(self, params: Dict, returns_data: pd.DataFrame, 
                                metrics: Dict, train_days: int, test_days: int, step_days: int):
        """Run walk-forward validation."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Generate strategies
            status_text.text("📋 Generating portfolio strategies...")
            progress_bar.progress(20)
            
            comparison_engine = PortfolioComparisonEngine(
                returns_data, metrics['expected_returns'], 
                metrics['covariance_matrix'], params['risk_free_rate']
            )
            portfolios = comparison_engine.generate_all_portfolios(
                params['constraints'],
                regularization=params.get('regularization', 0.1)
            )
            
            # Prepare strategy weights
            status_text.text("⚖️ Preparing strategy allocations...")
            progress_bar.progress(40)
            
            # Build strategy_weights robustly: accept dict or array-like, align to returns_data.columns,
            # threshold tiny weights, normalize to sum=1, skip empty allocations.
            strategy_weights = {}
            for portfolio_name, portfolio_data in portfolios.items():
               weights_map = {}
               w_field = portfolio_data.get('weights', None)

               # If ticker->weight dict already provided
               if isinstance(w_field, dict):
                   for t, w in w_field.items():
                       if w and float(w) > 1e-3:
                           weights_map[t] = float(w)
               # If array-like, align by index order
               elif isinstance(w_field, (list, tuple, np.ndarray)):
                   w_arr = np.asarray(w_field, dtype=float)
                   if len(w_arr) == len(returns_data.columns):
                       for i, ticker in enumerate(returns_data.columns):
                           w = float(w_arr[i])
                           if w > 1e-3:
                               weights_map[ticker] = w
                   else:
                       # fallback mapping field name patterns
                       alt_map = portfolio_data.get('weights_map') or portfolio_data.get('weights_dict')
                       if isinstance(alt_map, dict):
                           for t, w in alt_map.items():
                               if w and float(w) > 1e-3:
                                   weights_map[t] = float(w)
               # If nothing recognized, try to extract per-ticker weights from other fields
               else:
                   alt_map = portfolio_data.get('weights_map') or portfolio_data.get('weights_dict')
                   if isinstance(alt_map, dict):
                       for t, w in alt_map.items():
                           if w and float(w) > 1e-3:
                               weights_map[t] = float(w)

               # Normalize weights to sum to 1 (if non-empty); skip wholly empty strategies
               total_w = sum(weights_map.values())
               if total_w <= 0:
                   # skip adding empty allocation (prevents meaningless sims)
                   logging.getLogger(__name__).debug(f"Skipping empty strategy weights for {portfolio_name}")
                   continue
               # normalize
               for t in list(weights_map.keys()):
                   weights_map[t] = float(weights_map[t] / total_w)

               strategy_weights[portfolio_name] = weights_map
            
            # Run walk-forward validation
            status_text.text("📊 Running walk-forward validation...")
            progress_bar.progress(60)
            
            validation_results = self.walk_forward_engine.run_walk_forward_validation(
                strategy_weights, 
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                initial_capital=params['alpaca_config']['initial_capital'],
                tickers=params['tickers']
            )
            realistic_results = {}
            for strategy, results in validation_results.items():
                if 'mean_annualized_return' in results and results['mean_annualized_return'] < 0.30:  # Max 30%
                    realistic_results[strategy] = results
            
            # Display results
            status_text.text("📈 Analyzing results...")
            progress_bar.progress(90)
            
            self.render_walk_forward_results(validation_results, portfolios)
            self.render_performance_diagnostics(validation_results, portfolios)
            
            progress_bar.progress(100)
            status_text.text("✅ Validation complete!")
            
        except Exception as e:
            st.error(f"❌ Walk-forward validation failed: {e}")

    def render_walk_forward_results(self, validation_results: Dict, backtest_portfolios: Dict):
        """Render walk-forward validation results."""
        st.subheader("📊 Walk-Forward Validation Results")
        
        # Create performance comparison
        comparison_data = []
        for strategy_name, results in validation_results.items():
            if 'error' in results:
                continue
                
            backtest_metrics = backtest_portfolios.get(strategy_name, {})
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Windows': results['window_count'],
                'Mean Annual Return': f"{results['mean_annualized_return']:.1%}",
                'Return Std Dev': f"{results['std_annualized_return']:.1%}",
                'Median Sharpe': f"{results['median_sharpe_ratio']:.2f}",
                'Win Rate': f"{results['win_rate']:.1%}",
                'Consistency': f"{results['consistency_score']:.2f}",
                'Backtest Return': f"{backtest_metrics.get('expected_return', 0):.1%}",
                'Return Ratio': f"{results['mean_annualized_return']/max(backtest_metrics.get('expected_return', 1e-8), 0.01):.2f}"
            })
        
        if comparison_data:
            st.table(pd.DataFrame(comparison_data))
            
            # Visualizations
            self.render_walk_forward_charts(validation_results)
        else:
            st.warning("No valid validation results available")

    def render_performance_diagnostics(self, validation_results: Dict, backtest_portfolios: Dict):
        """Render performance diagnostics to identify issues."""
        st.subheader("🔍 Performance Diagnostics")
        
        if not validation_results:
            st.warning("No validation results available for diagnostics")
            return
        
        # Check for common issues
        issues = []
        
        for strategy, results in validation_results.items():
            if 'error' in results:
                continue
                
            # Issue 1: Too few validation windows
            if results.get('window_count', 0) < 10:
                issues.append(f"⚠️ **{strategy}**: Only {results['window_count']} validation windows - insufficient statistical power")
            
            # Issue 2: High return standard deviation
            return_std = results.get('std_annualized_return', 0)
            if return_std > 0.20:  # >20% std dev
                issues.append(f"📊 **{strategy}**: High return variability (std: {return_std:.1%}) - strategy is unstable")
            
            # Issue 3: Negative median Sharpe
            median_sharpe = results.get('median_sharpe_ratio', 0)
            if median_sharpe < 0:
                issues.append(f"📉 **{strategy}**: Negative Sharpe ratio ({median_sharpe:.2f}) - strategy underperforms risk-free rate")
            
            # Issue 4: Low win rate
            win_rate = results.get('win_rate', 0)
            if win_rate < 0.5:
                issues.append(f"🎯 **{strategy}**: Low win rate ({win_rate:.0%}) - strategy fails more often than it succeeds")
        
        # Display issues
        if issues:
            st.markdown("""
            <div class="validation-error">
            <h4>🚨 Identified Issues:</h4>
            """, unsafe_allow_html=True)
            
            for issue in issues:
                st.write(f"- {issue}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("""
            <div class="validation-success">
            <h4>💡 Recommended Fixes:</h4>
            <ul>
                <li><strong>Increase training data:</strong> Use 3+ years of historical data</li>
                <li><strong>Stronger regularization:</strong> Increase to 0.2-0.3 to reduce overfitting</li>
                <li><strong>Tighter constraints:</strong> Limit maximum weight to 20-25% per asset</li>
                <li><strong>Focus on robust methods:</strong> Minimum volatility and maximum diversification often work better</li>
                <li><strong>Realistic expectations:</strong> Target 5-10% annual returns for diversified portfolios</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ No major issues detected - strategies appear robust")

    # Call this after rendering walk-forward results

    def render_walk_forward_charts(self, validation_results: Dict):
        """Render walk-forward validation charts."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns distribution
            returns_data = []
            for strategy, results in validation_results.items():
                if 'all_window_results' in results:
                    for window in results['all_window_results']:
                        returns_data.append({
                            'Strategy': strategy,
                            'Annualized Return': window['annualized_return']
                        })
            
            if returns_data:
                df = pd.DataFrame(returns_data)
                fig = px.box(df, x='Strategy', y='Annualized Return', 
                            title="Distribution of Annualized Returns Across Windows")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Consistency comparison
            strategies = []
            consistency_scores = []
            mean_returns = []
            
            for strategy, results in validation_results.items():
                if 'consistency_score' in results:
                    strategies.append(strategy)
                    consistency_scores.append(results['consistency_score'])
                    mean_returns.append(results['mean_annualized_return'])
            
            if strategies:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=consistency_scores, y=mean_returns,
                    mode='markers+text',
                    text=strategies,
                    marker=dict(size=20, color=mean_returns, colorscale='Viridis'),
                    textposition="top center"
                ))
                fig.update_layout(
                    title="Return vs Consistency",
                    xaxis_title="Consistency Score",
                    yaxis_title="Mean Annual Return"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _run_forward_validation(self, params: Dict, returns_data: pd.DataFrame, metrics: Dict):
        """Run forward validation with simulated testing."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize strategies
            status_text.text("📋 Initializing portfolio strategies...")
            progress_bar.progress(10)
            
            comparison_engine = PortfolioComparisonEngine(
                returns_data, metrics['expected_returns'], 
                metrics['covariance_matrix'], params['risk_free_rate']
            )
            portfolios = comparison_engine.generate_all_portfolios(params['constraints'])
            
            # Step 2: Prepare strategy weights
            status_text.text("⚖️ Calculating target allocations...")
            progress_bar.progress(30)
            
            strategy_weights = {}
            for portfolio_name, portfolio_data in portfolios.items():
               weights_map = {}
               w_field = portfolio_data.get('weights', None)

               # If ticker->weight dict already provided
               if isinstance(w_field, dict):
                   for t, w in w_field.items():
                       if w and float(w) > 1e-3:
                           weights_map[t] = float(w)
               # If array-like, align by index order
               elif isinstance(w_field, (list, tuple, np.ndarray)):
                   w_arr = np.asarray(w_field, dtype=float)
                   if len(w_arr) == len(returns_data.columns):
                       for i, ticker in enumerate(returns_data.columns):
                           w = float(w_arr[i])
                           if w > 1e-3:
                               weights_map[ticker] = w
                   else:
                       # fallback mapping field name patterns
                       alt_map = portfolio_data.get('weights_map') or portfolio_data.get('weights_dict')
                       if isinstance(alt_map, dict):
                           for t, w in alt_map.items():
                               if w and float(w) > 1e-3:
                                   weights_map[t] = float(w)
               # If nothing recognized, try to extract per-ticker weights from other fields
               else:
                   alt_map = portfolio_data.get('weights_map') or portfolio_data.get('weights_dict')
                   if isinstance(alt_map, dict):
                       for t, w in alt_map.items():
                           if w and float(w) > 1e-3:
                               weights_map[t] = float(w)

               # Normalize weights to sum to 1 (if non-empty); skip wholly empty strategies
               total_w = sum(weights_map.values())
               if total_w <= 0:
                   # skip adding empty allocation (prevents meaningless sims)
                   logging.getLogger(__name__).debug(f"Skipping empty strategy weights for {portfolio_name}")
                   continue
               # normalize
               for t in list(weights_map.keys()):
                   weights_map[t] = float(weights_map[t] / total_w)

               strategy_weights[portfolio_name] = weights_map
            
            # Step 3: Run simulated forward test
            status_text.text("📊 Running simulated forward test...")
            progress_bar.progress(60)
            
            validation_results = self.forward_tester.run_forward_test(
                strategy_weights, 
                params['alpaca_config']['test_days'],
                params['alpaca_config']['initial_capital'],
                tickers=params['tickers']  # Pass the tickers for data download
            )
            realistic_results = {}
            for strategy, results in validation_results.items():
                if 'mean_annualized_return' in results and results['mean_annualized_return'] < 0.30:  # Max 30%
                    realistic_results[strategy] = results
            # Step 4: Analyze results
            status_text.text("📈 Analyzing validation results...")
            progress_bar.progress(90)
            
            # Display results
            self.render_validation_results(validation_results, portfolios)
            
            progress_bar.progress(100)
            status_text.text("✅ Validation complete!")
            
        except Exception as e:
            st.error(f"❌ Validation failed: {e}")
            import traceback
            st.code(traceback.format_exc())


    def render_validation_results(self, validation_results: Dict, backtest_portfolios: Dict):
        """Render validation results with better error handling and debugging."""

        st.subheader("📊 Validation Results")
        
        if not validation_results:
            st.error("❌ No validation results available from forward testing.")
            st.info("This could mean:")
            st.write("- Paper trading orders failed to execute")
            st.write("- Alpaca API connection issues")
            st.write("- No performance data collected during the test period")
            return
        
        # Debug info (can be removed later)
        with st.expander("🔍 Debug Info (Raw Results)"):
            st.write("Validation results keys:", list(validation_results.keys()))
            for strategy, results in validation_results.items():
                st.write(f"**{strategy}:**", results)
        
        try:
            # Calculate validation metrics
            comparison_results = self.validation_engine.compare_backtest_vs_realtime(
                backtest_portfolios, validation_results
            )
            
            if not comparison_results:
                st.warning("⚠️ Could not generate comparison results. Common issues:")
                st.write("1. **Strategy name mismatch** between backtest and forward test")
                st.write("2. **Missing required metrics** in forward test results")
                st.write("3. **Data format inconsistencies**")
                
                # Show what we have to help debug
                st.write("**Backtest portfolios:**", list(backtest_portfolios.keys()))
                st.write("**Validation results:**", list(validation_results.keys()))
                return
            
            # Display comparison table
            comparison_data = []
            for strategy, metrics in comparison_results.items():
                if 'error' in metrics:
                    comparison_data.append({
                        'Strategy': strategy,
                        'Status': f"❌ {metrics['error']}",
                        'Predicted Return': 'N/A',
                        'Actual Return': 'N/A',
                        'Return Deviation': 'N/A',
                        'Consistency Score': 'N/A',
                        'Effectiveness': 'N/A'
                    })
                else:
                    comparison_data.append({
                        'Strategy': strategy,
                        'Status': '✅ Success',
                        'Predicted Return': f"{metrics.get('predicted_return', 0):.1%}",
                        'Actual Return': f"{metrics.get('actual_return', 0):.1%}",
                        'Return Deviation': f"{metrics.get('return_deviation', 0):.1%}",
                        'Consistency Score': f"{metrics.get('consistency_score', 0):.2f}",
                        'Effectiveness': "✅ High" if metrics.get('consistency_score', 0) > 0.7 
                                    else "⚠️ Medium" if metrics.get('consistency_score', 0) > 0.5 
                                    else "❌ Low"
                    })
            
            if comparison_data:
                st.table(pd.DataFrame(comparison_data))
            else:
                st.warning("No comparison data available after processing.")
                return
            
            # Strategy ranking
            try:
                rankings = self.validation_engine.rank_strategies_effectiveness(comparison_results)
                
                if rankings:
                    st.subheader("🏆 Strategy Effectiveness Ranking")
                    for i, (strategy, score) in enumerate(rankings.items(), 1):
                        # Create a visual progress bar for the score
                        progress = min(score, 1.0)
                        col1, col2 = st.columns([3, 7])
                        
                        with col1:
                            st.write(f"{i}. **{strategy}**")
                            st.write(f"Score: {score:.3f}")
                        
                        with col2:
                            st.progress(progress)
                        
                        st.write("---")
                    
                    # Key insights
                    self.render_validation_insights(comparison_results, rankings)
                else:
                    st.warning("Could not generate strategy rankings.")
            except Exception as e:
                st.error(f"Error generating rankings: {e}")
                import traceback
                st.code(traceback.format_exc())
                
        except Exception as e:
            st.error(f"❌ Error processing validation results: {e}")
            import traceback
            st.code(traceback.format_exc())


    def render_validation_insights(self, comparison_results: Dict, rankings: Dict):
        """Render enhanced key insights focusing on robust methods."""
        st.subheader("💡 Enhanced Insights & Recommendations")
        
        if not rankings:
            st.warning("No rankings available for insights.")
            return
        
        best_strategy = list(rankings.keys())[0]
        best_score = rankings[best_strategy]
        
        # Calculate optimization effectiveness
        equal_weight_score = rankings.get('equal_weight', 0)
        optimized_scores = {k: v for k, v in rankings.items() if k != 'equal_weight'}
        best_optimized_score = max(optimized_scores.values()) if optimized_scores else 0
        
        # Enhanced analysis focusing on robust methods
        if any(robust in best_strategy.lower() for robust in ['regularized', 'bootstrap', 'diversification']):
            st.markdown(f"""
            <div class="validation-success">
            <h4>🎉 Robust Optimization Working!</h4>
            <p><strong>{best_strategy}</strong> is effective with score <strong>{best_score:.3f}</strong></p>
            <p>**Why this works:** Advanced methods reduce overfitting and improve stability</p>
            <p>**Recommendation:** Focus on robust methods for real implementation</p>
            <ul>
                <li><strong>Regularized Markowitz:</strong> Prevents extreme allocations</li>
                <li><strong>Bootstrap Robust:</strong> Most stable across market conditions</li>
                <li><strong>Maximum Diversification:</strong> Better risk-adjusted returns</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif best_optimized_score > equal_weight_score * 1.1:
            improvement = (best_optimized_score / equal_weight_score - 1) * 100
            st.markdown(f"""
            <div class="validation-warning">
            <h4>⚠️ Optimization Shows Promise</h4>
            <p><strong>{best_strategy}</strong> beats equal-weight by <strong>{improvement:.1f}%</strong></p>
            <p>**Next steps to improve further:**</p>
            <ul>
                <li>Try increasing regularization strength</li>
                <li>Use bootstrap method for more stability</li>
                <li>Test maximum diversification approach</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="validation-error">
            <h4>🔍 Traditional Optimization Underperforms - Use Robust Methods</h4>
            <p>Equal-weight strategy currently outperforms traditional optimization</p>
            
            <h5>🛠️ Recommended Solutions:</h5>
            <ul>
                <li><strong>Use Regularized Markowitz:</strong> Add L2 regularization (0.1-0.3)</li>
                <li><strong>Try Bootstrap Robust:</strong> Most reliable method for unstable markets</li>
                <li><strong>Maximum Diversification:</strong> Often beats traditional Sharpe ratio</li>
                <li><strong>Reduce Maximum Weight:</strong> From 40% to 20-25% for better diversification</li>
            </ul>
            
            <h5>📊 Why This Happens:</h5>
            <ul>
                <li>Traditional Markowitz overfits to historical data</li>
                <li>Extreme weights perform poorly out-of-sample</li>
                <li>Robust methods handle estimation errors better</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced diagnostic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            improvement = (best_optimized_score / equal_weight_score - 1) * 100
            st.metric("Optimization Improvement", f"{improvement:+.1f}%")
        
        with col2:
            # Show which robust methods are working
            robust_methods = ['regularized', 'bootstrap', 'diversification']
            robust_scores = [rankings.get(k, 0) for k in rankings if any(r in k for r in robust_methods)]
            best_robust_score = max(robust_scores) if robust_scores else 0
            st.metric("Best Robust Score", f"{best_robust_score:.3f}")
        
        with col3:
            traditional_score = rankings.get('markowitz_regularized', 0)
            robust_improvement = (best_robust_score / traditional_score - 1) * 100 if traditional_score > 0 else 0
            st.metric("Robust vs Traditional", f"{robust_improvement:+.1f}%")
        
        with col4:
            recommendation = "Use Robust Methods" if best_robust_score > 0.6 else "Stick to Equal-Weight"
            st.metric("Recommendation", recommendation)

    def _render_historical_validation(self):
        """Render historical validation results."""
        st.subheader("📈 Historical Validation Analysis")
        
        if not hasattr(self, 'forward_tester') or not self.forward_tester.test_results:
            st.info("No historical validation data available. Run a forward test first.")
            return
        
        test_results = self.forward_tester.test_results
        
        # Display test history
        for test_id, test_data in test_results.items():
            with st.expander(f"Test: {test_id}"):
                st.write(f"**Strategies:** {', '.join(test_data['strategies'])}")
                st.write(f"**Period:** {test_data['period_days']} days")
                st.write(f"**Initial Capital:** ${test_data['initial_capital']:,.2f}")
                
                # Show performance summary
                performance = test_data['performance']
                for strategy, metrics in performance.items():
                    if 'error' not in metrics:
                        st.metric(
                            f"{strategy} - Final Return",
                            f"{metrics.get('actual_return', 0):.2%}",
                            f"Sharpe: {metrics.get('actual_sharpe', 0):.2f}"
                        )
    # ORIGINAL METHODS FROM YOUR DASHBOARD - ADDED BACK
    def render_data_overview(self, returns_data: pd.DataFrame, metrics: Dict):
        """Render data overview section."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Assets", len(returns_data.columns))
            st.metric("Days", f"{len(returns_data)}")
            
        with col2:
            avg_return_arithmetic = metrics['expected_returns'].mean()
            avg_return_geometric = metrics.get('geometric_annual_return', avg_return_arithmetic)
            st.metric("Avg Return (Arithmetic)", f"{avg_return_arithmetic:.2%}")
            st.metric("Avg Return (Geometric)", f"{avg_return_geometric:.2%}")
            
        with col3:
            avg_vol = metrics['volatilities'].mean()
            st.metric("Avg Volatility", f"{avg_vol:.2%}")
            
        with col4:
            best_asset = metrics['sharpe_ratios'].idxmax()
            st.metric("Best Sharpe", best_asset)
        
        # Debug info in expander
        with st.expander("🔍 Data Consistency Check"):
            st.write("**Annualization Method:**", metrics.get('annualization_method', 'Unknown'))
            st.write("**Return Statistics:**")
            st.write(f"- Max: {metrics['expected_returns'].max():.2%}")
            st.write(f"- Min: {metrics['expected_returns'].min():.2%}")
            st.write(f"- Std: {metrics['expected_returns'].std():.2%}")
            
            # Check covariance matrix type
            cov_type = type(metrics['covariance_matrix']).__name__
            st.write(f"**Covariance Matrix Type:** {cov_type}")

        # In your dashboard, add this expander
        # with st.expander("🔍 Validation Diagnostics"):
        #     st.write("**Data Quality Check:**")
        #     st.write(f"- Total days of data: {len(returns_data)}")
        #     st.write(f"- Expected windows: {estimated_windows}")
        #     st.write(f"- Actual windows generated: {len(validation_results.get('equal_weight', {}).get('all_window_results', []))}")
            
        #     if validation_results:
        #         st.write("**Strategy Performance Summary:**")
        #         for strategy, results in validation_results.items():
        #             if 'error' not in results:
        #                 st.write(f"- {strategy}: {results['window_count']} windows, {results['mean_annualized_return']:.1%} mean return")

        # Correlation heatmap
        st.header("Asset Correlation Matrix")
        corr_matrix = returns_data.corr()
        fig = px.imshow(corr_matrix, 
                       color_continuous_scale="RdBu_r",
                       aspect="auto",
                       zmin=-1, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_comparison_section(self, returns_data: pd.DataFrame, metrics: Dict, params
    : Dict):
        """Render comprehensive portfolio comparison using the proper engine."""
        # Get regularization strength from params
        regularization = params.get('regularization', 0.1)
        # Initialize the SINGLE comparison engine
        comparison_engine = PortfolioComparisonEngine(
            returns_data, metrics['expected_returns'], 
            metrics['covariance_matrix'], params['risk_free_rate']
        )
        
        # Generate all portfolios
        with st.spinner("Generating portfolio strategies..."):
            portfolios = comparison_engine.generate_all_portfolios(
                params['constraints'],
                regularization=regularization
            )
            
        # Initialize comparison engine
        # comparison_engine = PortfolioComparisonEngine(
        #     returns_data, metrics['expected_returns'], metrics['covariance_matrix'], params['risk_free_rate']
        # )
        
        # # Generate all portfolios
        # with st.spinner("Generating portfolio strategies..."):
        #     portfolios = comparison_engine.generate_all_portfolios(params['constraints'])
        
        # Create comparison table
        self._render_comparison_table(portfolios)
        
        # Key insights
        st.subheader("💡 Key Insights")
        self._render_key_insights(portfolios)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_risk_return_chart(portfolios)
        
        with col2:
            self._render_metrics_comparison(portfolios)
    
    def _render_comparison_table(self, portfolios: Dict):
        """Render comparison table with proper styling."""
        # Create table data
        table_rows = []
        for portfolio in portfolios.values():
            # Use geometric return if available, otherwise fallback to expected_return
            display_return = portfolio.get('geometric_annual_return', portfolio['expected_return'])
            table_rows.append({
                'Portfolio': portfolio['portfolio_name'],
                'Sharpe': portfolio['sharpe_ratio'],
                'Volatility': portfolio['volatility'],
                'Return': display_return, #Sample based
                # 'Return (Geometric)': portfolio.get('geometric_annual_return', portfolio['expected_return']),  # Geometric
                'Max DD': portfolio['max_drawdown'],
                'CI Low': portfolio['mc_90_low'],
                'CI High': portfolio['mc_90_high'],
                'Loss Prob': portfolio['loss_prob_5_percent']
            })
        
        df = pd.DataFrame(table_rows)
        
        # Find best values for highlighting
        best_sharpe = df['Sharpe'].max()
        best_vol = df['Volatility'].min()
        best_return = df['Return'].max()
        best_dd = df['Max DD'].max()  # Closest to 0
        best_loss = df['Loss Prob'].min()
        
        # Build HTML table
        html = '<div class="comparison-table"><table>'
        html += '<thead><tr>'
        html += '<th>Portfolio</th><th>Sharpe Ratio</th><th>Volatility</th>'
        html += '<th>Expected Return</th><th>Max Drawdown</th>'
        html += '<th>90% CI Range</th><th>Loss Probability</th>'
        html += '</tr></thead><tbody>'
        
        for _, row in df.iterrows():
            html += '<tr>'
            html += f'<td style="font-weight: bold;">{row["Portfolio"]}</td>'
            
            # Sharpe
            css = ' class="best-metric"' if row['Sharpe'] == best_sharpe else ''
            html += f'<td{css}>{row["Sharpe"]:.2f}</td>'
            
            # Volatility
            css = ' class="best-metric"' if row['Volatility'] == best_vol else ''
            html += f'<td{css}>{row["Volatility"]:.1%}</td>'
            
            # Return
            css = ' class="best-metric"' if row['Return'] == best_return else ''
            html += f'<td{css}>{row["Return"]:.1%}</td>'
            
            # Max DD
            css = ' class="best-metric"' if row['Max DD'] == best_dd else ''
            html += f'<td{css}>{row["Max DD"]:.1%}</td>'
            
            # CI Range
            html += f'<td>{row["CI Low"]:.1%} to {row["CI High"]:.1%}</td>'
            
            # Loss Prob
            css = ' class="best-metric"' if row['Loss Prob'] == best_loss else ''
            html += f'<td{css}>{row["Loss Prob"]:.0%}</td>'
            
            html += '</tr>'
        
        html += '</tbody></table></div>'
        st.markdown(html, unsafe_allow_html=True)
    
    def _render_key_insights(self, portfolios: Dict):
        """Render key insights."""
        portfolio_list = list(portfolios.values())
        
        best_sharpe = max(portfolio_list, key=lambda x: x['sharpe_ratio'])
        best_vol = min(portfolio_list, key=lambda x: x['volatility'])
        best_dd = min(portfolio_list, key=lambda x: x['max_drawdown'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Best Risk-Adjusted", best_sharpe['portfolio_name'], 
                     f"Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
        
        with col2:
            st.metric("🛡️ Lowest Risk", best_vol['portfolio_name'],
                     f"Vol: {best_vol['volatility']:.1%}")
        
        with col3:
            st.metric("📉 Best Downside", best_dd['portfolio_name'],
                     f"Max DD: {best_dd['max_drawdown']:.1%}")
    
    def _render_risk_return_chart(self, portfolios: Dict):
        """Render risk-return comparison chart."""
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, portfolio in enumerate(portfolios.values()):
            fig.add_trace(go.Scatter(
                x=[portfolio['volatility']],
                y=[portfolio['expected_return']],
                mode='markers+text',
                marker=dict(size=20, color=colors[i]),
                name=portfolio['portfolio_name'],
                text=[portfolio['portfolio_name']],
                textposition="top center"
            ))
        
        fig.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Volatility",
            yaxis_title="Expected Return",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_metrics_comparison(self, portfolios: Dict):
        """Render bar chart comparing key metrics."""
        metrics_data = []
        for portfolio in portfolios.values():
            metrics_data.append({
                'Portfolio': portfolio['portfolio_name'],
                'Sharpe': portfolio['sharpe_ratio'],
                'Sortino': portfolio['sortino_ratio']
            })
        
        df = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Sharpe', x=df['Portfolio'], y=df['Sharpe'], marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name='Sortino', x=df['Portfolio'], y=df['Sortino'], marker_color='#4ECDC4'))
        
        fig.update_layout(
            title="Risk-Adjusted Returns",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_markowitz_results(self, returns_data: pd.DataFrame, metrics: Dict, params: Dict):
        """Render Markowitz optimization results."""
        optimizer = EnhancedMarkowitzOptimizer(
            metrics['expected_returns'],
            metrics['covariance_matrix'],
            params['risk_free_rate']
        )
        
        # Calculate optimal portfolios
        sharpe_result = optimizer.optimize('max_sharpe', params['constraints'], regularization = 0.1)
        
        # Display allocation
        self.render_portfolio_allocation(sharpe_result, metrics['expected_returns'].index)
        
        # Efficient Frontier
        st.subheader("Efficient Frontier")
        frontier_data = optimizer.efficient_frontier(n_points=50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'],
            y=frontier_data['return'],
            mode='lines',
            line=dict(color='blue', width=3),
            name='Efficient Frontier'
        ))
        
        fig.add_trace(go.Scatter(
            x=[sharpe_result['volatility']], 
            y=[sharpe_result['return']],
            mode='markers',
            marker=dict(size=15, color='gold', symbol='star'),
            name='Max Sharpe'
        ))
        
        fig.update_layout(
            xaxis_title='Volatility',
            yaxis_title='Return',
            height=400,
            legend=dict(
                orientation = 'h',
                x= 0.27,
                y=-0.3,
            ),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_monte_carlo_results(self, returns_data: pd.DataFrame, metrics: Dict, params: Dict):
        """Render Monte Carlo simulation results."""
        optimizer = MonteCarloOptimizer(
            metrics['expected_returns'],
            metrics['covariance_matrix'],
            params['risk_free_rate']
        )
        
        # Run simulation
        with st.spinner("Running Monte Carlo..."):
            simulation_results = optimizer.run_simulation(
                params['monte_carlo_portfolios'], 
                params['constraints']
            )
        
        optimal_portfolios = optimizer.find_optimal_portfolios(simulation_results)
        
        # Display allocation
        self.render_portfolio_allocation(
            optimal_portfolios['max_sharpe'], 
            metrics['expected_returns'].index
        )
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=simulation_results['volatility'],
            y=simulation_results['return'],
            mode='markers',
            marker=dict(
                size=3,
                color=simulation_results['sharpe_ratio'],
                colorscale='Viridis',
                showscale=True,
                opacity=0.6,
                colorbar=dict(
                    title="Sharpe Ratio",
                    x=1.05,
                    thickness=20,
                    tickfont=dict(size=10)
                )
            ),
            name='Random Portfolios'
        ))
        
        fig.add_trace(go.Scatter(
            x=[optimal_portfolios['max_sharpe']['volatility']],
            y=[optimal_portfolios['max_sharpe']['return']],
            mode='markers',
            marker=dict(size=15, color='gold', symbol='star'),
            name='Max Sharpe'
        ))
        
        fig.update_layout(
            xaxis_title='Volatility',
            yaxis_title='Return',
            legend=dict(
                orientation = 'h',
                x= 0.24,
                y=-0.2,
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_portfolio_allocation(self, result: Dict, tickers: List[str]):
        """Render portfolio allocation."""
        weights_series = pd.Series(result['weights'], index=tickers)
        positive_weights = weights_series[weights_series > 0.001]
        
        # Metrics
        st.markdown(f"""
        <div class="metric-card">
            <p><strong>Return:</strong> {result['return']:.2%} | 
            <strong>Volatility:</strong> {result['volatility']:.2%} | 
            <strong>Sharpe:</strong> {result['sharpe_ratio']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pie chart
        if len(positive_weights) > 0:
            fig = px.pie(positive_weights, 
                        values=positive_weights.values * 100,
                        names=positive_weights.index,
                        title="Allocation")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_analysis(self, returns_data: pd.DataFrame, metrics: Dict):
        """Render risk analysis section."""
        equal_weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            var_95 = PortfolioCalculators.calculate_value_at_risk(returns_data, equal_weights, 0.05)
            st.metric("95% VaR (1-day)", f"{var_95:.2%}")
            
        with col2:
            cvar_95 = PortfolioCalculators.calculate_cvar(returns_data, equal_weights, 0.05)
            st.metric("95% CVaR (1-day)", f"{cvar_95:.2%}")
            
        with col3:
            max_dd = PortfolioCalculators.calculate_max_drawdown(returns_data, equal_weights)
            st.metric("Maximum Drawdown", f"{max_dd:.2%}")
            
        with col4:
            sortino = PortfolioCalculators.calculate_sortino_ratio(returns_data, equal_weights, 0.02)
            st.metric("Sortino Ratio", f"{sortino:.2f}")

    def render_performance_analysis(self, returns_data: pd.DataFrame, metrics: Dict, params: Dict):
        """Render performance analysis with benchmark comparison."""
        
        # Load benchmark data
        benchmark_ticker = params.get('benchmark_ticker', 'SPY')
        benchmark_data = self.data_loader.load_ticker_data(
            [benchmark_ticker], params['start_date'], params['end_date']
        )
        
        if benchmark_data.empty:
            st.warning(f"Could not load benchmark data for {benchmark_ticker}")
            return
        
        benchmark_returns = benchmark_data.iloc[:, 0]
        
        # Calculate portfolios to compare
        portfolios_to_compare = {}
        
        # Equal-weighted
        equal_weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
        portfolios_to_compare['Equal-Weighted'] = equal_weights
        
        # Markowitz optimal
        markowitz_opt = EnhancedMarkowitzOptimizer(
            metrics['expected_returns'],
            metrics['covariance_matrix'],
            params['risk_free_rate']
        )
        sharpe_result = markowitz_opt.optimize('max_sharpe', params['constraints'])
        portfolios_to_compare['Markowitz Optimal'] = sharpe_result['weights']
        
        # Min volatility
        min_vol_result = markowitz_opt.optimize('min_volatility', params['constraints'])
        portfolios_to_compare['Min Volatility'] = min_vol_result['weights']
        
        # Create cumulative returns chart
        fig = go.Figure()
        
        # Benchmark
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name=f'{benchmark_ticker} Benchmark',
            line=dict(color='#95a5a6', width=3, dash='dash'),
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
        
        # Portfolio strategies
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for i, (name, weights) in enumerate(portfolios_to_compare.items()):
            portfolio_returns = returns_data.dot(weights)
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            
            fig.add_trace(go.Scatter(
                x=portfolio_cumulative.index,
                y=portfolio_cumulative.values,
                mode='lines',
                name=name,
                line=dict(color=colors[i], width=2),
                hovertemplate='%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Cumulative Returns: Portfolio Strategies vs {benchmark_ticker}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (Normalized to 1)",
            height=500,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_main_content(self, params: Dict, returns_data: pd.DataFrame, metrics: Dict):
        """Render the main dashboard content."""
        st.markdown('<h1 class="main-header">📊 Modern Portfolio Theory Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.info("**Professional Portfolio Optimization** with Real-world Validation")
        
        # Data overview
        self.render_data_overview(returns_data, metrics)
        
        # Validation section (new)
        self.render_validation_section(params, returns_data, metrics)
        
        # Main comparison section
        st.header("🎯 Portfolio Strategy Comparison")
        self.render_comparison_section(returns_data, metrics, params)
        
        # Combined optimization results
        st.header("📈 Optimization Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Markowitz Optimization")
            self.render_markowitz_results(returns_data, metrics, params)
        
        with col2:
            st.subheader("Monte Carlo Simulation")
            self.render_monte_carlo_results(returns_data, metrics, params)
        
        # Risk analysis
        st.header("⚠️ Risk Metrics")
        self.render_risk_analysis(returns_data, metrics)

        # Performance Analysis Section
        st.header("📈 Performance Analysis vs Benchmark")
        self.render_performance_analysis(returns_data, metrics, params)
    
    def run(self):
        """Run the enhanced dashboard application."""
        try:

            # Set up Streamlit-specific shutdown handling
            # if not self._setup_shutdown_handling():
            #     return
                

            # Render sidebar and get parameters
            params = self.render_sidebar()
            
            # Validate and load data
            valid_tickers = self.data_loader.validate_tickers(params['tickers'])
            if not valid_tickers:
                st.error("No valid tickers found. Please check your input.")
                return
                
            returns_data = self.data_loader.load_ticker_data(
                valid_tickers, 
                params['start_date'], 
                params['end_date']
            )
            
            if returns_data.empty:
                st.error("No data loaded. Please check your ticker symbols and date range.")
                return
                
            # Calculate metrics
            metrics = self.data_loader.calculate_metrics(returns_data)
            
            # Render main content
            self.render_main_content(params, returns_data, metrics)                            

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")
    
    # def _setup_shutdown_handling(self):
    #     """Set up proper shutdown handling for Streamlit."""
    #     try:
    #         # Streamlit already handles SIGINT, so we just need to ensure
    #         # our components clean up properly
    #         return True
    #     except Exception as e:
    #         st.warning(f"Shutdown handling setup failed: {e}")
    #         return True  # Continue anyway

# Run the enhanced dashboard
if __name__ == "__main__":
    dashboard = EnhancedPortfolioOptimizationDashboard()
    dashboard.run()