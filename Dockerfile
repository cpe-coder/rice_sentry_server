
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

EXPOSE 80


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
 