FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 -

# 환경변수 설정
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH=/app

# 프로젝트 파일 복사
COPY pyproject.toml poetry.lock ./
COPY .env ./
COPY main.py ./
COPY rag/ ./rag/
COPY ./data/healthcare_data_ladi ./data/healthcare_data_ladi

# Poetry 설정 및 의존성 설치
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# 포트 설정
EXPOSE 8000

# 애플리케이션 실행
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]