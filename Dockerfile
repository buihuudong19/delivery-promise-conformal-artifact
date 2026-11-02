# Container for the pipeline, with the library versions of the reported run.
#   docker build -t p2du .
#   docker run --rm -v /path/to/olist-csv:/app/data/raw:ro -v /path/to/empty-folder:/app/outputs p2du --quick
# Mount an empty folder on /app/outputs: a run overwrites the tables, figures and manifest found there.
# XGBoost and the quantile forest use random subsampling whose streams differ between platforms; the reported run is
# the Windows run recorded in outputs/manifest.json.
# Without arguments the container runs the full pipeline (python run_pipeline.py).
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
ENTRYPOINT ["python", "run_pipeline.py"]
