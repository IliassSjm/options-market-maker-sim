# Base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app


# This ensures setup.py is present when pip needs it.
COPY . .

# --- Now, install dependencies from requirements.txt ---
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]