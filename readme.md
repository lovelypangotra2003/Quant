# Quantitative Portfolio Optimizer and Backtesting Framework

## Overview

This project implements a modular framework for systematic portfolio construction and evaluation.

The goal is to study how portfolio optimization methods behave under realistic trading assumptions. The framework integrates portfolio construction, walk-forward validation, and backtesting with transaction cost modeling to avoid common pitfalls such as look-ahead bias and overly optimistic performance estimates. The framework also supports walk-forward re-estimation of covariance matrices and portfolio weights to test stability across changing market regimes.

## Features

- **Modern Portfolio Theory**: Portfolio optimization using Mean-variance optimization (Markowitz Model) with covariance shrinkage
- **Multiple Objectives**: Maximize Sharpe ratio, minimize volatility, maximize return
- **Research and Validation**: Walk-forward validation framework to avoid look-ahead bias
- **Backtesting engine**: Backtesting engine with transaction cost and slippage modelling
- **Visualization**: Interactive efficient frontier and allocation charts
- **Risk Analysis**: Sharpe, VaR, CVaR, turnover, maximum drawdown, etc.
- **Flexible Constraints**: No shorting, maximum weight limits

## Architecture

project/
│
├── dashboard.py
│
├── models/
│   ├── markowitz.py
│   ├── monte_carlo.py
│
├── utils/
│   ├── data_loader.py
│   ├── calculators.py
│   ├── comparison_engine.py
│
├── validation/
│   ├── real_time_tracker.py
│   ├── simulated_forward_tester.py
│   ├── validation_metrics.py
  

## Data

The framework currently uses historical equity price data sourced via Yahoo Finance. Data preprocessing includes return computation, missing value handling, and rolling window estimation for covariance matrices used in optimization.


## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/quant-portfolio-optimizer.git
cd quant-portfolio-optimizer
```


## Usage

### Web Dashboard
```
streamlit run dashboard.py
```
# or
```
python main.py dashboard
```
