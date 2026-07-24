# Hugging Face Docker Space running the Swarm-SAR Streamlit dashboard.
# (HF removed "streamlit" as a create-time SDK; Docker is the supported path.)
# Installs only the torch-free viz stack so the free CPU tier builds fast.
FROM python:3.11-slim

# Non-root user (HF Spaces run as uid 1000) with a writable home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR $HOME/app

# Install the package (self-contained sim + dashboard) straight from GitHub.
RUN pip install --no-cache-dir --user \
    "swarm-sar[viz] @ git+https://github.com/mathi0405/drone-swarm-sar.git@main"

COPY --chown=user app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
