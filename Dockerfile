FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-live.txt ./
RUN pip install --no-cache-dir -r requirements-live.txt
COPY run.py config*.json ./
COPY cryptomoni ./cryptomoni
COPY data/BTC-USD.csv ./data/BTC-USD.csv
COPY data/stocks ./data/stocks
ENV PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["python", "run.py", "up", "--host", "0.0.0.0"]
