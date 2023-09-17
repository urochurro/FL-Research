import flwr as fl
import utils
import pandas as pd
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional

df = pd.read_csv("train_split2.csv")
X_test, y_test = utils.preprocess(df)

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[tuple[float, Dict[str, Scalar]]]:
        utils.set_model_params(model, parameters)
        proba = model.predict_proba(X_test)
        loss = log_loss(y_test, proba)
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(server_address="localhost:3000", strategy=strategy, config=fl.server.ServerConfig(num_rounds=50))