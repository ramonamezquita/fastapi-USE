# Run application as non-root process in the Docker container.

FROM python:3.10

RUN pip install --upgrade pip

RUN adduser --disabled-login worker
USER worker
WORKDIR /home/worker

COPY --chown=worker:worker requirements.txt requirements.txt
RUN pip install --user -r requirements.txt
ENV PATH="/home/worker/.local/bin:${PATH}"

# Endpoint to the model prediction service.
ENV MODEL_ENDPOINT=http://modelserving:8501/v1/models/universal-sentence-encoder:predict

VOLUME app /home/worker/app

COPY --chown=worker:worker . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]