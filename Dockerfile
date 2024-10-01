
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
