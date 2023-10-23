FROM emacski/tensorflow-serving:2.5.1

# Copy and mount saved model.
COPY models /models
VOLUME models/universal-sentence-encoder models/universal-sentence-encoder

ENV MODEL_NAME=universal-sentence-encoder
