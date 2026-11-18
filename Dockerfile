FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Use the virtual environment directly, so nothing is installed again when the container starts
ENV PATH="/app/.venv/bin:$PATH"

COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
