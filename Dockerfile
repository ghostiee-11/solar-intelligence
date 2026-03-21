FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for cartopy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e ".[geo]"

COPY . .

EXPOSE 5006

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5006/panel_dashboard || exit 1

CMD ["panel", "serve", "src/solar_intelligence/ui/panel_dashboard.py", \
     "--address=0.0.0.0", "--port=5006", \
     "--allow-websocket-origin=*", \
     "--num-procs=2"]
