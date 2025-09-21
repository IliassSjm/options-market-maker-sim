import streamlit as st
import pandas as pd
from datetime import datetime

# --- Import Your Simulation Engine ---
# This brings in the main class from your orchestrator script
from main_simulator import MarketMakingSimulation

# --- App Configuration ---
st.set_page_config(
    page_title="Market Making Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI: Title ---
st.title("Market Making Simulator with Delta Hedging 🛡️")

# --- UI: Sidebar for Parameters ---
st.sidebar.header("Simulation Setup")

with st.sidebar.expander("Market Parameters", expanded=True):
    initial_price = st.number_input("Initial Stock Price ($)", value=100.0)
    strike_price = st.number_input("Option Strike Price ($)", value=105.0)
    drift = st.slider("Drift (μ)", 0.0, 0.15, 0.05, 0.01, help="Annual expected return of the stock.")
    volatility = st.slider("Volatility (σ)", 0.1, 0.5, 0.25, 0.01, help="Annual volatility of the stock.")
    time_horizon = st.slider("Time Horizon (Years)", 0.1, 2.0, 0.25, 0.05, help="Total length of the simulation.")
    option_maturity = st.number_input("Option Maturity (Years)", value=0.25, help="Must be >= Time Horizon.")

with st.sidebar.expander("Market Maker Parameters", expanded=True):
    starting_cash = st.number_input("Starting Cash ($)", value=50000)
    risk_free_rate = st.slider("Risk-Free Rate (r)", 0.0, 0.1, 0.05, 0.01)
    base_spread = st.slider("Base Spread ($)", 0.05, 1.0, 0.30, 0.05)
    gamma = st.slider("Risk Aversion (Gamma)", 0.01, 0.5, 0.1, 0.01, help="Higher gamma means wider spreads.")

# --- Main App Logic ---
if st.button("🚀 Run Full Simulation"):

    # 1. Assemble parameters from the UI into dictionaries
    simulation_params = {
        'initial_price': initial_price,
        'strike_price': strike_price,
        'drift': drift,
        'volatility': volatility,
        'time_horizon': time_horizon,
        'time_step': 1/252,
        'option_maturity': option_maturity,
        'option_type': 'call',
        'order_probability': 0.08,
        'seed': 42,
        'use_vol_surface': True,
        'base_vol': volatility,
        'smile_curvature': 0.1
    }

    market_maker_params = {
        'starting_cash': float(starting_cash),
        'gamma': gamma,
        'risk_free_rate': risk_free_rate,
        'base_spread': base_spread,
        'min_spread': 0.05,
        'max_spread': 1.50
    }

    # Use a spinner to show that the simulation is running
    with st.spinner("Running complex simulation... This may take a moment."):
        
        # 2. Initialize your main simulation class with the parameters
        sim = MarketMakingSimulation(simulation_params, market_maker_params)
        
        # 3. Run the simulation
        results = sim.run_simulation()
        
        # 4. Generate the performance report and plots
        report = sim.generate_performance_report()
        fig = sim.plot_simulation_results()

    st.success("✅ Simulation Complete!")

    # --- Display Results ---
    st.header("Performance Report")
    
    # Display key metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Final P&L", f"${report['pnl_metrics']['final_pnl']:,.2f}")
    col2.metric("Total Trades", f"{report['trading_metrics']['total_trades']}")
    col3.metric("Sharpe Ratio", f"{report['pnl_metrics']['sharpe_ratio']:.2f}")

    # Show the full report in an expandable section
    with st.expander("View Detailed Performance Report"):
        st.json(report)

    st.header("Simulation Visualizations")
    # 5. Display the Matplotlib figure in Streamlit
    st.pyplot(fig)