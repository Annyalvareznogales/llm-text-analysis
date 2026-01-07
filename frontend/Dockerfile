FROM python:3.11

LABEL org.opencontainers.image.source=https://github.com/Annyalvareznogales/llm-text-analysis

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
COPY . /app/frontend

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]