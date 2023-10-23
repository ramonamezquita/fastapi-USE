from requests import Session


class TensorFlowServingClient:
    """TensorFlow Serving model client.

    Sends POST requests to service hosting the tensorflow model.

    Parameters
    ----------
    model_endpoint : str
        Endpoint in which the model is being served.
    """

    def __init__(self, model_endpoint: str):
        self.model_endpoint = model_endpoint
        self.session = Session()

        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

    def predict(self, sentences: list[str]) -> dict[str, list[list[float]]]:
        """Predicts input sentences.

        Parameters
        ----------
        data : list of str
            List of input sentences.

        Returns
        -------
        embeddings : list of lists of str
            Embeddings for each input sentence.
        """
        payload = {"instances": sentences}
        reponse = self.session.post(
            url=self.model_endpoint, json=payload, headers=self.headers
        )

        return reponse.json()["predictions"]
