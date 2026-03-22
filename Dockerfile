FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for cartopy and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml requirements.txt ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[geo]"

# Copy remaining files
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["panel", "serve", "src/solar_intelligence/ui/panel_dashboard.py", \
     "--address=0.0.0.0", "--port=7860", \
     "--allow-websocket-origin=*", \
     "--prefix=/", \
     "--num-procs=1"]
