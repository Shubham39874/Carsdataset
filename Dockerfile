# Use a slim version of Python to keep the pod light
FROM python:3.11-slim

# Set the folder inside the pod where our code will live
WORKDIR /app

# Install the ingredients
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (including the model and models folders!)
COPY . .

# Streamlit uses port 8501 by default
EXPOSE 8501

# The command to launch the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]