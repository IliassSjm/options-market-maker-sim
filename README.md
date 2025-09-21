
# Market Making Simulator for European Options 📈

This project is an interactive web application built with **Streamlit** that simulates the strategy of a market maker for European options. The simulation includes dynamic quote generation, trade execution, and a delta hedging strategy to manage risk. The entire application is containerized with **Docker** for easy and consistent deployment.

The core of the simulation uses a **Geometric Brownian Motion** model to generate a realistic stock price path. The market maker agent then provides bid/ask quotes based on the **Black-Scholes** model, adjusting for inventory risk. The application provides a comprehensive performance report and visualizations of the simulation results.

---

## Features

* **Interactive Dashboard**: Adjust key simulation parameters like volatility, drift, and risk aversion in real-time.
* **Dynamic Quoting**: The market maker's bid/ask spread adjusts based on inventory and market conditions.
* **Delta Hedging**: See how the agent trades the underlying stock to maintain a delta-neutral position.
* **Performance Analytics**: View detailed reports and charts on Profit & Loss (P&L), inventory, and portfolio value.
* **Containerized**: Fully containerized with Docker for seamless setup and execution.
* **Automated Testing**: Includes a CI/CD pipeline with **GitHub Actions** to automatically run unit tests on every push.

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need to have the following software installed:
* Git
* Python 3.9+
* Docker Desktop

### Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/options-market-maker-sim.git
    cd options-market-maker-sim
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *On Windows, use: `.venv\Scripts\activate`*

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    Your browser will automatically open to the application.

### Running with Docker 🐳

This is the recommended way to run the project as it requires no local Python setup.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/options-market-maker-sim.git
    cd options-market-maker-sim
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t market-maker-sim .
    ```

3.  **Run the container:**
    ```bash
    docker run -p 8501:8501 market-maker-sim
    ```

4.  **Open the application:**
    Navigate to **`http://localhost:8501`** in your web browser.