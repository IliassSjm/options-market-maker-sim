"""
Market Making Strategy Backtester
================================

Comprehensive backtesting framework for market making strategies.
Runs multiple independent simulations and provides statistical analysis
of strategy performance, risk metrics, and P&L distributions.

This backtester:
1. Runs N independent simulations with different random seeds
2. Collects comprehensive performance metrics from each run
3. Calculates aggregate statistics across all runs
4. Provides detailed analysis of P&L distributions
5. Generates visualizations for strategy assessment

Author: Backtesting Module
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our simulation modules
try:
    from main_simulator import MarketMakingSimulation
    from market_maker import MarketMaker
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure main_simulator.py and market_maker.py are in the same directory.")
    exit(1)


class MarketMakingBacktester:
    """
    Comprehensive backtesting framework for market making strategies.
    
    This class orchestrates multiple simulation runs and provides
    statistical analysis of strategy performance across different
    market conditions and random scenarios.
    """
    
    def __init__(self, base_simulation_params: Dict, base_market_maker_params: Dict):
        """
        Initialize the backtester.
        
        Args:
            base_simulation_params: Base parameters for simulations
            base_market_maker_params: Base parameters for market maker
        """
        self.base_sim_params = base_simulation_params
        self.base_mm_params = base_market_maker_params
        
        # Results storage
        self.simulation_results: List[Dict] = []
        self.performance_reports: List[Dict] = []
        self.simulation_objects: List[MarketMakingSimulation] = []
        
        # Aggregate statistics
        self.aggregate_stats: Dict = {}
        
    def run_backtest(self, n_runs: int = 500, vary_parameters: bool = True,
                    parameter_noise: float = 0.1, progress_interval: int = 50) -> Dict:
        """
        Run comprehensive backtest with multiple simulation runs.
        
        Args:
            n_runs: Number of simulation runs
            vary_parameters: Whether to add noise to parameters across runs
            parameter_noise: Standard deviation of parameter noise (as fraction)
            progress_interval: How often to print progress updates
            
        Returns:
            Dictionary containing aggregate backtest results
        """
        print(f"Starting Market Making Backtest")
        print(f"{'='*50}")
        print(f"Number of runs: {n_runs}")
        print(f"Parameter variation: {vary_parameters}")
        if vary_parameters:
            print(f"Parameter noise level: {parameter_noise*100:.1f}%")
        print(f"Base simulation parameters:")
        
        # Print key base parameters
        key_params = ['initial_price', 'strike_price', 'volatility', 'time_horizon', 
                     'order_probability', 'base_vol', 'smile_curvature']
        for param in key_params:
            if param in self.base_sim_params:
                value = self.base_sim_params[param]
                if isinstance(value, float):
                    if 'price' in param:
                        print(f"  {param}: ${value:.2f}")
                    elif 'probability' in param or 'vol' in param:
                        print(f"  {param}: {value*100:.1f}%")
                    else:
                        print(f"  {param}: {value}")
                else:
                    print(f"  {param}: {value}")
        
        print(f"\nStarting {n_runs} simulation runs...")
        print()
        
        start_time = time.time()
        successful_runs = 0
        failed_runs = 0
        
        # Run simulations
        for run_idx in range(n_runs):
            try:
                # Create varied parameters for this run
                sim_params = self._create_varied_parameters(
                    self.base_sim_params, run_idx, vary_parameters, parameter_noise
                )
                mm_params = self.base_mm_params.copy()
                
                # Create and run simulation
                simulation = MarketMakingSimulation(sim_params, mm_params)
                results = simulation.run_simulation()
                
                # Generate performance report
                performance_report = simulation.generate_performance_report()
                
                # Store results
                results['run_id'] = run_idx
                results['seed'] = sim_params['seed']
                self.simulation_results.append(results)
                self.performance_reports.append(performance_report)
                # Note: Not storing simulation objects to save memory for large backtests
                
                successful_runs += 1
                
                # Progress reporting
                if (run_idx + 1) % progress_interval == 0 or run_idx == n_runs - 1:
                    elapsed = time.time() - start_time
                    remaining_runs = n_runs - (run_idx + 1)
                    eta = (elapsed / (run_idx + 1)) * remaining_runs if remaining_runs > 0 else 0
                    
                    print(f"Progress: {run_idx + 1}/{n_runs} ({((run_idx + 1)/n_runs)*100:.1f}%) "
                          f"- Successful: {successful_runs}, Failed: {failed_runs}")
                    print(f"  Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                    
                    if len(self.simulation_results) > 0:
                        recent_pnls = [r['total_pnl'] for r in self.simulation_results[-progress_interval:]]
                        avg_recent_pnl = np.mean(recent_pnls)
                        print(f"  Recent avg P&L: ${avg_recent_pnl:.2f}")
                    print()
                
            except Exception as e:
                failed_runs += 1
                print(f"Run {run_idx} failed: {str(e)[:100]}")
                continue
        
        total_time = time.time() - start_time
        
        print(f"Backtest completed in {total_time:.1f} seconds")
        print(f"Successful runs: {successful_runs}/{n_runs} ({successful_runs/n_runs*100:.1f}%)")
        print(f"Failed runs: {failed_runs}")
        print()
        
        # Calculate aggregate statistics
        self.aggregate_stats = self._calculate_aggregate_statistics()
        
        return self.aggregate_stats
    
    def _create_varied_parameters(self, base_params: Dict, run_idx: int, 
                                 vary_parameters: bool, noise_level: float) -> Dict:
        """
        Create varied parameters for a simulation run.
        
        Args:
            base_params: Base simulation parameters
            run_idx: Run index for seed generation
            vary_parameters: Whether to add parameter variation
            noise_level: Level of noise to add
            
        Returns:
            Dictionary of parameters for this run
        """
        params = base_params.copy()
        
        # Always vary the seed
        params['seed'] = run_idx + 1000  # Offset to avoid low seeds
        
        if not vary_parameters:
            return params
        
        # Add controlled noise to key parameters
        np.random.seed(run_idx + 2000)  # Separate seed for parameter generation
        
        # Parameters to vary and their constraints
        param_configs = {
            'volatility': {'min': 0.1, 'max': 0.6, 'type': 'multiplicative'},
            'base_vol': {'min': 0.1, 'max': 0.6, 'type': 'multiplicative'},
            'smile_curvature': {'min': 0.0, 'max': 0.3, 'type': 'additive'},
            'order_probability': {'min': 0.02, 'max': 0.15, 'type': 'multiplicative'},
            'drift': {'min': -0.1, 'max': 0.2, 'type': 'additive'},
            'gamma': {'min': 0.05, 'max': 0.3, 'type': 'multiplicative'},  # Market maker param
        }
        
        for param_name, config in param_configs.items():
            if param_name in params:
                base_value = params[param_name]
                
                if config['type'] == 'multiplicative':
                    # Multiplicative noise (log-normal)
                    noise = np.random.normal(1.0, noise_level)
                    new_value = base_value * noise
                else:
                    # Additive noise
                    noise = np.random.normal(0.0, noise_level * base_value)
                    new_value = base_value + noise
                
                # Apply constraints
                new_value = np.clip(new_value, config['min'], config['max'])
                params[param_name] = new_value
        
        return params
    
    def _calculate_aggregate_statistics(self) -> Dict:
        """
        Calculate comprehensive aggregate statistics across all runs.
        
        Returns:
            Dictionary containing aggregate statistics
        """
        if not self.simulation_results:
            return {}
        
        # Extract key metrics from all runs
        metrics = {
            'final_pnl': [r['total_pnl'] for r in self.simulation_results],
            'total_trades': [r['total_trades'] for r in self.simulation_results],
            'total_hedges': [r['total_hedges'] for r in self.simulation_results],
            'final_inventory': [r['final_inventory'] for r in self.simulation_results],
            'final_cash': [r['final_cash'] for r in self.simulation_results],
            'final_underlying_shares': [r['final_underlying_shares'] for r in self.simulation_results],
            'price_return': [r['price_return'] for r in self.simulation_results]
        }
        
        # Add metrics from performance reports if available
        if self.performance_reports:
            try:
                sharpe_ratios = [r['pnl_metrics']['sharpe_ratio'] for r in self.performance_reports 
                               if not np.isnan(r['pnl_metrics']['sharpe_ratio']) and 
                               not np.isinf(r['pnl_metrics']['sharpe_ratio'])]
                max_drawdowns = [r['pnl_metrics']['max_drawdown'] for r in self.performance_reports]
                pnl_volatilities = [r['pnl_metrics']['pnl_volatility'] for r in self.performance_reports]
                
                if sharpe_ratios:
                    metrics['sharpe_ratio'] = sharpe_ratios
                if max_drawdowns:
                    metrics['max_drawdown'] = max_drawdowns
                if pnl_volatilities:
                    metrics['pnl_volatility'] = pnl_volatilities
            except (KeyError, TypeError) as e:
                print(f"Warning: Could not extract some performance metrics: {e}")
        
        # Calculate statistics for each metric
        aggregate_stats = {
            'backtest_summary': {
                'total_runs': len(self.simulation_results),
                'successful_runs': len(self.simulation_results),
                'success_rate': 1.0
            }
        }
        
        for metric_name, values in metrics.items():
            if not values:
                continue
                
            values = np.array(values)
            
            # Remove any infinite or NaN values
            valid_values = values[np.isfinite(values)]
            
            if len(valid_values) == 0:
                continue
            
            metric_stats = {
                'count': len(valid_values),
                'mean': np.mean(valid_values),
                'median': np.median(valid_values),
                'std': np.std(valid_values),
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'q25': np.percentile(valid_values, 25),
                'q75': np.percentile(valid_values, 75),
                'skewness': self._calculate_skewness(valid_values),
                'kurtosis': self._calculate_kurtosis(valid_values)
            }
            
            # Special calculations for P&L
            if metric_name == 'final_pnl':
                profitable_runs = np.sum(valid_values > 0)
                metric_stats.update({
                    'profitable_runs': profitable_runs,
                    'profit_percentage': (profitable_runs / len(valid_values)) * 100,
                    'loss_runs': len(valid_values) - profitable_runs,
                    'avg_profit': np.mean(valid_values[valid_values > 0]) if profitable_runs > 0 else 0,
                    'avg_loss': np.mean(valid_values[valid_values <= 0]) if profitable_runs < len(valid_values) else 0,
                    'profit_factor': abs(np.sum(valid_values[valid_values > 0]) / 
                                       np.sum(valid_values[valid_values <= 0])) if np.sum(valid_values[valid_values <= 0]) != 0 else np.inf,
                    'var_95': np.percentile(valid_values, 5),  # Value at Risk (95% confidence)
                    'cvar_95': np.mean(valid_values[valid_values <= np.percentile(valid_values, 5)])  # Conditional VaR
                })
            
            aggregate_stats[metric_name] = metric_stats
        
        return aggregate_stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
    
    def print_backtest_results(self) -> None:
        """
        Print comprehensive backtest results.
        """
        if not self.aggregate_stats:
            print("No backtest results available. Run backtest first.")
            return
        
        print(f"\n{'='*70}")
        print(f"MARKET MAKING STRATEGY BACKTEST RESULTS")
        print(f"{'='*70}")
        
        # Summary
        summary = self.aggregate_stats['backtest_summary']
        print(f"\nBACKTEST SUMMARY:")
        print(f"  Total Runs: {summary['total_runs']}")
        print(f"  Successful Runs: {summary['successful_runs']}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
        
        # P&L Statistics
        if 'final_pnl' in self.aggregate_stats:
            pnl_stats = self.aggregate_stats['final_pnl']
            print(f"\nP&L STATISTICS:")
            print(f"  Mean P&L: ${pnl_stats['mean']:,.2f}")
            print(f"  Median P&L: ${pnl_stats['median']:,.2f}")
            print(f"  Std Dev P&L: ${pnl_stats['std']:,.2f}")
            print(f"  Min P&L: ${pnl_stats['min']:,.2f}")
            print(f"  Max P&L: ${pnl_stats['max']:,.2f}")
            print(f"  25th Percentile: ${pnl_stats['q25']:,.2f}")
            print(f"  75th Percentile: ${pnl_stats['q75']:,.2f}")
            print(f"  Skewness: {pnl_stats['skewness']:.3f}")
            print(f"  Kurtosis: {pnl_stats['kurtosis']:.3f}")
            
            print(f"\nPROFITABILITY ANALYSIS:")
            print(f"  Profitable Runs: {pnl_stats['profitable_runs']} ({pnl_stats['profit_percentage']:.1f}%)")
            print(f"  Loss Runs: {pnl_stats['loss_runs']}")
            print(f"  Average Profit: ${pnl_stats['avg_profit']:,.2f}")
            print(f"  Average Loss: ${pnl_stats['avg_loss']:,.2f}")
            print(f"  Profit Factor: {pnl_stats['profit_factor']:.2f}")
            
            print(f"\nRISK METRICS:")
            print(f"  VaR (95%): ${pnl_stats['var_95']:,.2f}")
            print(f"  CVaR (95%): ${pnl_stats['cvar_95']:,.2f}")
        
        # Trading Statistics
        if 'total_trades' in self.aggregate_stats:
            trade_stats = self.aggregate_stats['total_trades']
            print(f"\nTRADING STATISTICS:")
            print(f"  Average Trades per Run: {trade_stats['mean']:.1f}")
            print(f"  Median Trades per Run: {trade_stats['median']:.1f}")
            print(f"  Min Trades: {trade_stats['min']:.0f}")
            print(f"  Max Trades: {trade_stats['max']:.0f}")
        
        # Hedging Statistics
        if 'total_hedges' in self.aggregate_stats:
            hedge_stats = self.aggregate_stats['total_hedges']
            print(f"\nHEDGING STATISTICS:")
            print(f"  Average Hedges per Run: {hedge_stats['mean']:.1f}")
            print(f"  Median Hedges per Run: {hedge_stats['median']:.1f}")
            print(f"  Min Hedges: {hedge_stats['min']:.0f}")
            print(f"  Max Hedges: {hedge_stats['max']:.0f}")
        
        # Risk-Adjusted Performance
        if 'sharpe_ratio' in self.aggregate_stats:
            sharpe_stats = self.aggregate_stats['sharpe_ratio']
            print(f"\nRISK-ADJUSTED PERFORMANCE:")
            print(f"  Average Sharpe Ratio: {sharpe_stats['mean']:.3f}")
            print(f"  Median Sharpe Ratio: {sharpe_stats['median']:.3f}")
            print(f"  Sharpe Ratio Std Dev: {sharpe_stats['std']:.3f}")
        
        print(f"\n{'='*70}")
    
    def plot_backtest_results(self, save_plots: bool = False, plot_dir: str = "backtest_plots") -> None:
        """
        Create comprehensive plots of backtest results.
        
        Args:
            save_plots: Whether to save plots to files
            plot_dir: Directory to save plots (if save_plots=True)
        """
        if not self.simulation_results:
            print("No backtest results to plot. Run backtest first.")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # Extract data for plotting
        pnls = [r['total_pnl'] for r in self.simulation_results]
        trades = [r['total_trades'] for r in self.simulation_results]
        hedges = [r['total_hedges'] for r in self.simulation_results]
        inventories = [r['final_inventory'] for r in self.simulation_results]
        price_returns = [r['price_return'] for r in self.simulation_results]
        
        # 1. P&L Distribution (Main plot)
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(pnls, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(pnls), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')
        ax1.axvline(np.median(pnls), color='green', linestyle='--', linewidth=2, label=f'Median: ${np.median(pnls):.2f}')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.5, label='Break-even')
        ax1.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Final P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. P&L vs Trade Count
        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(trades, pnls, alpha=0.6, s=20)
        ax2.set_title('P&L vs Number of Trades')
        ax2.set_xlabel('Total Trades')
        ax2.set_ylabel('Final P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(trades, pnls)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 3. P&L vs Hedge Count
        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(hedges, pnls, alpha=0.6, s=20, color='orange')
        ax3.set_title('P&L vs Number of Hedges')
        ax3.set_xlabel('Total Hedges')
        ax3.set_ylabel('Final P&L ($)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trade Count Distribution
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(trades, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax4.set_title('Trade Count Distribution')
        ax4.set_xlabel('Number of Trades')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Final Inventory Distribution
        ax5 = plt.subplot(3, 3, 5)
        ax5.hist(inventories, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', alpha=0.7, label='Flat')
        ax5.set_title('Final Inventory Distribution')
        ax5.set_xlabel('Final Inventory (Contracts)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. P&L vs Market Return
        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(price_returns, pnls, alpha=0.6, s=20, color='red')
        ax6.set_title('P&L vs Market Return')
        ax6.set_xlabel('Underlying Return (%)')
        ax6.set_ylabel('Final P&L ($)')
        ax6.grid(True, alpha=0.3)
        
        # 7. P&L Cumulative Distribution
        ax7 = plt.subplot(3, 3, 7)
        sorted_pnls = np.sort(pnls)
        percentiles = np.arange(1, len(sorted_pnls) + 1) / len(sorted_pnls) * 100
        ax7.plot(sorted_pnls, percentiles, linewidth=2, color='navy')
        ax7.axvline(0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax7.set_title('P&L Cumulative Distribution')
        ax7.set_xlabel('P&L ($)')
        ax7.set_ylabel('Percentile')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Sharpe Ratio Distribution (if available)
        ax8 = plt.subplot(3, 3, 8)
        if 'sharpe_ratio' in self.aggregate_stats:
            sharpe_ratios = [r['pnl_metrics']['sharpe_ratio'] for r in self.performance_reports 
                           if not np.isnan(r['pnl_metrics']['sharpe_ratio']) and 
                           not np.isinf(r['pnl_metrics']['sharpe_ratio'])]
            if sharpe_ratios:
                ax8.hist(sharpe_ratios, bins=30, alpha=0.7, color='brown', edgecolor='black')
                ax8.axvline(np.mean(sharpe_ratios), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(sharpe_ratios):.3f}')
                ax8.set_title('Sharpe Ratio Distribution')
                ax8.set_xlabel('Sharpe Ratio')
                ax8.set_ylabel('Frequency')
                ax8.legend()
        else:
            ax8.text(0.5, 0.5, 'Sharpe Ratio\nData Not Available', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Sharpe Ratio Distribution')
        ax8.grid(True, alpha=0.3)
        
        # 9. Box plot of key metrics
        ax9 = plt.subplot(3, 3, 9)
        metrics_data = [pnls, trades, hedges]
        metrics_labels = ['P&L\n($)', 'Trades\n(count)', 'Hedges\n(count)']
        
        # Normalize data for visualization
        normalized_data = []
        for data in metrics_data:
            if np.std(data) > 0:
                normalized_data.append((np.array(data) - np.mean(data)) / np.std(data))
            else:
                normalized_data.append(np.array(data))
        
        box_plot = ax9.boxplot(normalized_data, labels=metrics_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax9.set_title('Normalized Metrics Box Plot')
        ax9.set_ylabel('Standardized Values')
        ax9.grid(True, alpha=0.3)
        ax9.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(f"{plot_dir}/backtest_results.png", dpi=300, bbox_inches='tight')
            print(f"Plots saved to {plot_dir}/backtest_results.png")
        
        plt.show()
    
    def export_results(self, filename: str = "backtest_results.csv") -> None:
        """
        Export detailed results to CSV file.
        
        Args:
            filename: Name of CSV file to create
        """
        if not self.simulation_results:
            print("No results to export. Run backtest first.")
            return
        
        # Create comprehensive DataFrame
        df = pd.DataFrame(self.simulation_results)
        
        # Add performance metrics if available
        if self.performance_reports:
            for i, report in enumerate(self.performance_reports):
                try:
                    df.loc[i, 'sharpe_ratio'] = report['pnl_metrics']['sharpe_ratio']
                    df.loc[i, 'max_drawdown'] = report['pnl_metrics']['max_drawdown']
                    df.loc[i, 'pnl_volatility'] = report['pnl_metrics']['pnl_volatility']
                except (KeyError, IndexError):
                    continue
        
        # Export to CSV
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
        print(f"Exported {len(df)} simulation results with {len(df.columns)} metrics")


def run_comprehensive_backtest(n_runs: int = 500) -> MarketMakingBacktester:
    """
    Run a comprehensive backtest with default parameters.
    
    Args:
        n_runs: Number of simulation runs
        
    Returns:
        Configured and executed backtester object
    """
    print("Comprehensive Market Making Strategy Backtest")
    print("=" * 50)
    
    # Base simulation parameters
    simulation_params = {
        'initial_price': 100.0,
        'strike_price': 105.0,
        'drift': 0.05,
        'volatility': 0.25,
        'time_horizon': 0.25,
        'time_step': 1/252,
        'option_maturity': 0.25,
        'option_type': 'call',
        'order_probability': 0.08,
        'use_vol_surface': True,
        'base_vol': 0.25,
        'smile_curvature': 0.1,
        'seed': 42  # This will be varied across runs
    }
    
    # Base market maker parameters
    market_maker_params = {
        'starting_cash': 50000.0,
        'gamma': 0.1,
        'risk_free_rate': 0.05,
        'base_spread': 0.30,
        'min_spread': 0.05,
        'max_spread': 1.50
    }
    
    # Create and run backtester
    backtester = MarketMakingBacktester(simulation_params, market_maker_params)
    
    # Run backtest with parameter variation
    results = backtester.run_backtest(
        n_runs=n_runs, 
        vary_parameters=True, 
        parameter_noise=0.15,  # 15% parameter noise
        progress_interval=max(10, n_runs // 20)
    )
    
    # Print results
    backtester.print_backtest_results()
    
    # Create plots
    print("\nGenerating backtest visualization plots...")
    backtester.plot_backtest_results()
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_results_{n_runs}runs_{timestamp}.csv"
    backtester.export_results(filename)
    
    return backtester


if __name__ == "__main__":
    # Configuration
    N_RUNS = 500  # Number of simulation runs
    
    print(f"Starting comprehensive backtest with {N_RUNS} runs...")
    print("This may take several minutes depending on your system.")
    print()
    
    # Run the backtest
    backtester = run_comprehensive_backtest(N_RUNS)
    
    print(f"\nBacktest completed!")
    print(f"Access backtester object for detailed analysis: backtester")
    print(f"Access aggregate statistics: backtester.aggregate_stats")
    print(f"Access individual run results: backtester.simulation_results")
    
    # Example of additional analysis
    if backtester.simulation_results:
        pnls = [r['total_pnl'] for r in backtester.simulation_results]
        print(f"\nQuick Summary:")
        print(f"  Best run P&L: ${max(pnls):,.2f}")
        print(f"  Worst run P&L: ${min(pnls):,.2f}")
        print(f"  Profit percentage: {(sum(1 for p in pnls if p > 0) / len(pnls) * 100):.1f}%")
