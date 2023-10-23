# Data Science Coding Challenge
The purpose of this project is to containerize the solution to the Data Science
coding test given by OpenTable.

A docker-compose.yml file is provided for building the required services
which include,

|     Name     |         Description         | Port |
|:------------:|:---------------------------:|:----:|
|    webapp    | Web app built using FastAPI | 5000 |
| modelserving |   Model inference service   | 8501 |


The final user interacts only with the `webapp` service which in turn propagates
the user's input to the model inference service. This helps us to keep
the web application light and responsive while keeping the heavy dependencies
to the inference service container.


# Run project
First, clone this repo with
```
git clone https://github.com/ramonamezquita/opentable-challenge.git
```

CD into the project, download and decompress the Universal Sentence Encoder inside `MODELS_DIR`.
```
cd opentable-challenge
MODELS_DIR="$(pwd)/modelserving/models/universal-sentence-encoder/01"
wget -P $MODELS_DIR 'https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed'
tar -xvf $MODELS_DIR/4\?tf-hub-format=compressed -C $MODELS_DIR
```
> [!WARNING]  
> This model is about 1GB. Depending on your network speed, it might take a while.


Build and run services.
```
docker-compose up
```


# Endpoints
Once services are running, go to http://localhost:5000/docs to try out the available endpoints which are

|          Path          |                              Description                              | Method |
|:----------------------:|:---------------------------------------------------------------------:|:------:|
|      /embeddings/      | Returns a JSON encoded array of the embedding for the given sentence. |  GET   |
|    /embeddings/bulk    |              Returns embeddings for a list of sentences.              |  POST  |
| /embeddings/similarity |      Returns the cosine similarity between a pair of sentences.       |  POST  |


# Payload examples
A couple of sample payloads are stored in the *payloads* directory which can
be executed using the `curl` command.

```
curl -X POST -H "Content-Type: application/json" -d @payloads/payload_2.json http://localhost:5000/embeddings/bulk
curl -X POST -H "Content-Type: application/json" -d @payloads/payload_3.json http://localhost:5000/embeddings/similarity
```


# Tests
A collection of unit tests are given for the webapp service (see `webapp/app/test_main.py`). 
To run them you need to execute `pytest` inside the webapp container with
```
docker exec web pytest app
```









