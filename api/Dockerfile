FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV ICP_DATABASE_URL=sqlite:////data/icp.db
EXPOSE 8080
CMD ["uvicorn", "icp_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
