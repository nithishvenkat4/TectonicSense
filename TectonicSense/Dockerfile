# TectonicSense container
# Final service configuration will be added after local inference is implemented.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["python", "-c", "print('TectonicSense container initialized')"]
