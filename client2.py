
import warnings
import flwr as fl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


import utils

import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__":

    df_train=pd.read_csv("train_split1.csv",encoding='utf-8')
    X, y = utils.preprocess(df_train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 0)

    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    utils.set_initial_params(model)

    class HospitalClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}
        
    fl.client.start_numpy_client(server_address="localhost:8080", client=HospitalClient())