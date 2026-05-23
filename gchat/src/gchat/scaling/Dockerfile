FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir "python-fasthtml>=0.13.4"

COPY src/gchat/__init__.py src/gchat/__init__.py
COPY src/gchat/scaling/ src/gchat/scaling/

ENV PYTHONPATH=/app/src

EXPOSE 5001

CMD ["python", "-m", "gchat.scaling.dashboard", "--host", "0.0.0.0", "--port", "5001", "--no-reload"]
