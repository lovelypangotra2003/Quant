# validation/validation_metrics.py
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from models.markowitz import EnhancedMarkowitzOptimizer

class RealWorldValidation:
    """Handles validation metrics and strategy effectiveness scoring."""
    
    def __init__(self):
        self.validation_history = []
        self.logger = logging.getLogger(__name__)

    # def compare_backtest_vs_realtime(self, backtest_metrics: Dict, realtime_metrics: Dict, 
    #                            test_period_days: int = 30) -> Dict:
    #     """Compare backtest predictions vs real-world performance with consistent units."""
    #     comparison_results = {}
        
    #     for strategy_name, backtest_data in backtest_metrics.items():
    #         if strategy_name in realtime_metrics:
    #             realtime_data = realtime_metrics[strategy_name]
                
    #             if 'error' in realtime_data:
    #                 comparison_results[strategy_name] = {
    #                     'error': realtime_data['error'],
    #                     'consistency_score': 0.0,
    #                     'predicted_return': backtest_data.get('expected_return', 0),
    #                     'actual_return': 0.0
    #                 }
    #                 continue
                
    #             # USE ANNUALIZED RETURNS FOR COMPARISON
    #             predicted_annual = backtest_data.get('expected_return', 0)
    #             actual_annual = realtime_data.get('annualized_return', 0)
    #             actual_annual = realtime_data.get('annualized_return', None)
    #             # Fallback: if forward test didn't provide annualized_return, compute from period return + days
    #             if actual_annual is None:
    #                 if realtime_data.get('total_days') and 'actual_return' in realtime_data:
    #                     days = max(1, int(realtime_data.get('total_days', test_period_days)))
    #                     period_return = realtime_data.get('actual_return', 0)
    #                     try:
    #                         actual_annual = (1 + period_return) ** (252.0 / days) - 1
    #                     except Exception:
    #                         actual_annual = float(period_return)
    #                 else:
    #                     actual_annual = float(realtime_data.get('actual_return', 0))
    #             comparison = {
    #                 'strategy': strategy_name,
    #                 'predicted_return': predicted_annual,
    #                 'actual_return': actual_annual,
    #                 'predicted_volatility': backtest_data.get('volatility', 0),
    #                 'actual_volatility': realtime_data.get('volatility', 0),
    #                 'predicted_sharpe': backtest_data.get('sharpe_ratio', 0),
    #                 'actual_sharpe': realtime_data.get('actual_sharpe', 0),
    #                 'return_deviation': self._calculate_deviation(predicted_annual, actual_annual),
    #                 'volatility_deviation': self._calculate_deviation(
    #                     backtest_data.get('volatility', 0),
    #                     realtime_data.get('volatility', 0)
    #                 ),
    #                 # PASS TEST PERIOD TO CONSISTENCY SCORE
    #                 'consistency_score': self._calculate_consistency_score(
    #                     backtest_data, realtime_data, test_period_days
    #                 )
    #             }
    #             comparison_results[strategy_name] = comparison
        
    #     return comparison_results
    def compare_backtest_vs_walk_forward(self, backtest_metrics: Dict, walk_forward_metrics: Dict) -> Dict:
        """Compare backtest predictions vs walk-forward performance."""
        comparison_results = {}
        
        for strategy_name, backtest_data in backtest_metrics.items():
            if strategy_name in walk_forward_metrics:
                wf_data = walk_forward_metrics[strategy_name]
                
                if 'error' in wf_data:
                    comparison_results[strategy_name] = {'error': wf_data['error']}
                    continue
                
                # Use median performance for more robust comparison
                predicted_annual = backtest_data.get('expected_return', 0)
                actual_annual_median = wf_data.get('median_annualized_return', 0)
                actual_annual_mean = wf_data.get('mean_annualized_return', 0)
                
                comparison = {
                    'strategy': strategy_name,
                    'predicted_return': predicted_annual,
                    'actual_return_median': actual_annual_median,
                    'actual_return_mean': actual_annual_mean,
                    'return_std_dev': wf_data.get('std_annualized_return', 0),
                    'consistency_score': wf_data.get('consistency_score', 0),
                    'win_rate': wf_data.get('win_rate', 0),
                    'window_count': wf_data.get('window_count', 0),
                    'return_ratio': actual_annual_median / max(predicted_annual, 0.01),
                    'prediction_accuracy': 1 - min(1.0, abs(actual_annual_median - predicted_annual) / 
                                                (abs(predicted_annual) + 0.05))
                }
                comparison_results[strategy_name] = comparison
        
        return comparison_results
    
    def _calculate_deviation(self, predicted: float, actual: float) -> float:
        """Calculate normalized deviation between predicted and actual values."""
        if predicted == 0:
            return abs(actual) if actual != 0 else 0.0
        
        return abs((actual - predicted) / predicted)
    
    
    def _calculate_consistency_score(self, backtest_data: Dict, realtime_data: Dict, 
                               test_period_days: int = 30) -> float:
        """Calculate consistency score adjusted for test period length."""
        try:
            scores = []
            
            # Adjust expectations for short test periods
            period_adjustment = min(1.0, test_period_days / 90)  # Normalize to 90-day "full" period
            
            predicted_annual = backtest_data.get('expected_return', 0)
            actual_annual = realtime_data.get('annualized_return', 0)
            
            # For short periods, be more tolerant of deviations
            tolerance = 0.5 + (0.5 * period_adjustment)  # 0.5 to 1.0 tolerance
            
            return_deviation = self._calculate_deviation(predicted_annual, actual_annual)
            adjusted_return_score = 1 - min(1.0, return_deviation / tolerance)
            scores.append(adjusted_return_score * 0.4)
            
            # Similar adjustments for other metrics...
            volatility_deviation = self._calculate_deviation(
                backtest_data.get('volatility', 0),
                realtime_data.get('volatility', 0)
            )
            adjusted_vol_score = 1 - min(1.0, volatility_deviation / tolerance)
            scores.append(adjusted_vol_score * 0.3)
            
            # Sharpe ratio comparison
            sharpe_deviation = self._calculate_deviation(
                backtest_data.get('sharpe_ratio', 0),
                realtime_data.get('actual_sharpe', 0)
            )
            adjusted_sharpe_score = 1 - min(1.0, sharpe_deviation / tolerance)
            scores.append(adjusted_sharpe_score * 0.3)
            
            return max(0.0, min(1.0, sum(scores)))
            
        except Exception as e:
            self.logger.warning(f"Error calculating period-adjusted consistency score: {e}")
            return 0.0
    def _calculate_prediction_accuracy(self, backtest_data: Dict, realtime_data: Dict) -> float:
        """Calculate prediction accuracy score."""
        try:
            # Compare multiple metrics
            metrics_to_compare = [
                ('expected_return', 'actual_return'),
                ('volatility', 'volatility'),
                ('max_drawdown', 'max_drawdown')
            ]
            
            accuracies = []
            for backtest_metric, realtime_metric in metrics_to_compare:
                pred = backtest_data.get(backtest_metric, 0)
                actual = realtime_data.get(realtime_metric, 0)
                
                if pred != 0 or actual != 0:
                    accuracy = 1 - min(1.0, abs(actual - pred) / (abs(pred) + 1e-8))
                    accuracies.append(accuracy)
            
            return np.mean(accuracies) if accuracies else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating prediction accuracy: {e}")
            return 0.0
    
    def _calculate_risk_consistency(self, backtest_data: Dict, realtime_data: Dict) -> float:
        """Calculate risk adjustment consistency."""
        try:
            # Compare risk-adjusted returns
            backtest_sharpe = backtest_data.get('sharpe_ratio', 0)
            actual_sharpe = realtime_data.get('actual_sharpe', 0)
            
            if backtest_sharpe != 0 or actual_sharpe != 0:
                sharpe_consistency = 1 - min(1.0, abs(actual_sharpe - backtest_sharpe) / 
                                           (abs(backtest_sharpe) + 1e-8))
            else:
                sharpe_consistency = 0.0
            
            # Compare drawdowns
            backtest_dd = backtest_data.get('max_drawdown', 0)
            actual_dd = realtime_data.get('max_drawdown', 0)
            
            if backtest_dd != 0 or actual_dd != 0:
                dd_consistency = 1 - min(1.0, abs(actual_dd - backtest_dd) / 
                                       (abs(backtest_dd) + 1e-8))
            else:
                dd_consistency = 0.0
            
            return (sharpe_consistency + dd_consistency) / 2
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk consistency: {e}")
            return 0.0
    
    # Add this to your validation_metrics.py
    def calculate_portfolio_stability(self, weights: np.ndarray, returns_data: pd.DataFrame) -> float:
        """Calculate portfolio stability score based on weight concentration and turnover."""
        
        # 1. Concentration penalty (Herfindahl index)
        concentration = np.sum(weights ** 2)
        concentration_penalty = min(1.0, concentration * 5)  # Normalize
        
        # 2. Check for extreme weights
        extreme_weights_penalty = np.sum(weights > 0.3) * 0.2  # Penalty for weights > 30%
        
        # 3. Bootstrap stability test
        stability_scores = []
        n_bootstraps = 20
        
        for _ in range(n_bootstraps):
            # Sample different time periods
            sample_size = len(returns_data) // 2
            sample_returns = returns_data.sample(n=sample_size, replace=True)
            
            # Calculate metrics on sample
            sample_er = sample_returns.mean() * 252
            sample_cov = sample_returns.cov() * 252
            
            # Simple stability check
            try:
                sample_optimizer = EnhancedMarkowitzOptimizer(sample_er, sample_cov, self.risk_free_rate)
                sample_result = sample_optimizer.optimize('max_sharpe', {'no_shorting': True}, regularization=0.1)
                sample_weights = sample_result['weights']
                
                # Calculate weight similarity
                weight_similarity = 1 - np.abs(weights - sample_weights).mean()
                stability_scores.append(weight_similarity)
            except:
                stability_scores.append(0.0)
        
        stability_score = np.mean(stability_scores) if stability_scores else 0.0
        
        # Combine scores
        final_stability = max(0.0, stability_score - concentration_penalty - extreme_weights_penalty)
        return final_stability

    def rank_strategies_effectiveness(self, comparison_results: Dict) -> Dict:
        """Rank strategies based on real-world effectiveness."""
        effectiveness_scores = {}
        
        for strategy, metrics in comparison_results.items():
            if 'error' in metrics:
                effectiveness_scores[strategy] = 0.0
                continue
                
            try:
                # Composite effectiveness score
                consistency = metrics.get('consistency_score', 0)
                prediction_accuracy = metrics.get('prediction_accuracy', 0)
                risk_consistency = metrics.get('risk_adjustment_consistency', 0)
                actual_performance = min(2.0, metrics.get('actual_sharpe', 0)) / 2.0  # Normalize Sharpe
                
                score = (
                    0.3 * consistency +
                    0.25 * prediction_accuracy +
                    0.25 * risk_consistency +
                    0.2 * actual_performance
                )
                
                effectiveness_scores[strategy] = max(0.0, min(1.0, score))
                
            except Exception as e:
                self.logger.warning(f"Error calculating effectiveness for {strategy}: {e}")
                effectiveness_scores[strategy] = 0.0
        
        return dict(sorted(effectiveness_scores.items(), key=lambda x: x[1], reverse=True))
    
    def assess_optimization_effectiveness(self, validation_results: Dict) -> Dict:
        """Comprehensive assessment of optimization effectiveness."""
        assessment = {
            'is_optimization_working': False,
            'best_performing_strategy': None,
            'overall_effectiveness': 0.0,
            'recommendations': [],
            'key_metrics': {}
        }
        
        if not validation_results:
            return assessment
        
        # Calculate overall effectiveness
        effectiveness_scores = self.rank_strategies_effectiveness(validation_results)
        
        if effectiveness_scores:
            assessment['best_performing_strategy'] = list(effectiveness_scores.keys())[0]
            assessment['overall_effectiveness'] = effectiveness_scores[assessment['best_performing_strategy']]
            
            # Determine if optimization is working
            equal_weight_score = effectiveness_scores.get('Equal-Weighted', 0)
            best_optimized_score = max([score for strategy, score in effectiveness_scores.items() 
            if strategy != 'Equal-Weighted'], default=0)
            
            assessment['is_optimization_working'] = best_optimized_score > equal_weight_score * 1.1
            
            # Determine if optimization is working
            # Be tolerant of naming variants for the equal-weight baseline
            equal_key_candidates = {'equal_weight', 'Equal-Weighted', 'Equal-Weighted ', 'equal-weight', 'equal weight'}
            equal_weight_score = 0.0
            for k in equal_key_candidates:
                if k in effectiveness_scores:
                    equal_weight_score = effectiveness_scores[k]
                    break
            # fallback to any strategy with "equal" in name
            if equal_weight_score == 0.0:
                for k, v in effectiveness_scores.items():
                    if 'equal' in k.lower():
                        equal_weight_score = v
                        break

            best_optimized_score = max([score for strategy, score in effectiveness_scores.items() 
                                      if strategy not in equal_key_candidates and 'equal' not in strategy.lower()],
                                     default=0)

            assessment['is_optimization_working'] = best_optimized_score > equal_weight_score * 1.1 if equal_weight_score >= 0 else False

            # Generate recommendations
            assessment['recommendations'] = self._generate_recommendations(
                assessment, effectiveness_scores, validation_results
            )
            
            # Key metrics
            assessment['key_metrics'] = {
                'average_consistency': np.mean([m.get('consistency_score', 0) 
                                              for m in validation_results.values()]),
                'best_consistency': max([m.get('consistency_score', 0) 
                                       for m in validation_results.values()]),
                'prediction_accuracy_range': self._calculate_accuracy_range(validation_results)
            }
        
        return assessment
    
    def _generate_recommendations(self, assessment: Dict, scores: Dict, results: Dict) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        overall_effectiveness = assessment['overall_effectiveness']
        best_strategy = assessment['best_performing_strategy']
        
        if overall_effectiveness > 0.8:
            recommendations.append(f"✅ **Excellent performance** - {best_strategy} is highly effective")
            recommendations.append("Consider scaling the strategy with real capital")
        elif overall_effectiveness > 0.6:
            recommendations.append(f"⚠️ **Good performance** - {best_strategy} shows promise")
            recommendations.append("Monitor transaction costs and market regime sensitivity")
        else:
            recommendations.append("🔍 **Optimization needs improvement**")
            recommendations.append("Test across different time periods and market conditions")
            recommendations.append("Consider adding more robust constraints to reduce overfitting")
        
        # Check for specific issues
        for strategy, metrics in results.items():
            if metrics.get('return_deviation', 0) > 0.5:
                recommendations.append(f"High return prediction deviation for {strategy} - review assumptions")
            if metrics.get('consistency_score', 0) < 0.4:
                recommendations.append(f"Low consistency for {strategy} - potential overfitting")
        
        return recommendations
    
    def _calculate_accuracy_range(self, validation_results: Dict) -> Dict:
        """Calculate range of prediction accuracies."""
        accuracies = [m.get('prediction_accuracy', 0) for m in validation_results.values()]
        
        if not accuracies:
            return {'min': 0, 'max': 0, 'mean': 0}
        
        return {
            'min': min(accuracies),
            'max': max(accuracies),
            'mean': np.mean(accuracies)
        }
    
    def save_validation_results(self, results: Dict, filename: str = None):
        """Save validation results for future analysis."""
        if filename is None:
            filename = f"validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Validation results saved to {filename}")
    
    def load_validation_results(self, filename: str) -> Dict:
        """Load previously saved validation results."""
        import pickle
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading validation results: {e}")
            return {}