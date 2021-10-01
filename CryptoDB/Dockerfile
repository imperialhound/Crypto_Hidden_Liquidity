FROM python:3.7-slim

WORKDIR /app
COPY . .

# upgrade pip, install requirements and update permissions for main.py
RUN pip install --upgrade pip \
    && pip install -r ./requirements.txt \
    && chmod +x /app/main.py

# Create CryptoFeed Kafka Producers
CMD ["python", "/app/main.py"]