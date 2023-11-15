<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>FL-RESEARCH</h1>
<h3>◦ Diving Deep into Code with FL-Research-Uncovering The Future!</h3>
<h3>◦ Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikitlearn" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat-square&logo=SciPy&logoColor=white" alt="SciPy" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy" />
</p>
<img src="https://img.shields.io/github/license/urochurro/FL-Research?style=flat-square&color=5D6D7E" alt="GitHub license" />
<img src="https://img.shields.io/github/last-commit/urochurro/FL-Research?style=flat-square&color=5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/github/commit-activity/m/urochurro/FL-Research?style=flat-square&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/languages/top/urochurro/FL-Research?style=flat-square&color=5D6D7E" alt="GitHub top language" />
</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
<!-- - [📦 Features](#-features) -->
- [📂 repository Structure](#-repository-structure)
- [⚙️ Modules](#modules)
- [🚀 Getting Started](#-getting-started)
    - [🔧 Installation](#-installation)
    - [🤖 Running FL-Research](#-running-FL-Research)
<!-- - [🛣 Roadmap](#-roadmap) -->
<!-- - [🤝 Contributing](#-contributing) -->
- [📄 License](#-license)
<!-- - [👏 Acknowledgments](#-acknowledgments) -->

---


## 📍 Overview

The FL-Research GitHub repository is designed for a Federated Learning setup. It contains scripts for a server, multiple clients, and data processing utilities. The server operates using the Flower framework and handles client model's weights to achieve a custom aggregation strategy. Clients (like client0.py and client1.py) are configured to develop Neural Network models using Keras. A Jupyter notebook is included for data visualization and prediction on genetic datasets. The project extensively utilizes Python libraries for machine learning, data handling, and client-server communication.

---

<!-- ## 📦 Features

HTTPStatus Exception: 429

--- -->


## 📂 Repository Structure

```sh
└── FL-Research/
    ├── Client/
    │   ├── client0.py
    │   ├── client1.py
    │   ├── client2.py
    │   ├── client3.py
    │   └── client4.py
    ├── client.ipynb
    ├── Data/
    ├── genetic_dataset_prediction_using_lazypredict.ipynb
    ├── requirements.txt
    ├── server.py
    └── utils.py

```

---


## ⚙️ Modules

<details closed><summary>Root</summary>

| File                                                                                                                                                        | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                                                         | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [client.ipynb](https://github.com/urochurro/FL-Research/blob/main/client.ipynb)                                                                             | The code in client.ipynb loads a CSV dataset into a pandas DataFrame and splits it into 3 random parts using scikit-learn's train_test_split function. It then saves each split's training set as a separate CSV file for later use. This is important for machine learning processes where distinct subsets of data are required for training and validation purposes.                                                                                                                                                                                                         |
| [genetic_dataset_prediction_using_lazypredict.ipynb](https://github.com/urochurro/FL-Research/blob/main/genetic_dataset_prediction_using_lazypredict.ipynb) | The code snippet is from a Jupyter notebook used for data processing and visualization in a genetic dataset prediction task. Specifically, it imports necessary libraries such as numpy, pandas, seaborn, matplotlib, and disables warning messages. The overall directory structure suggests an application setup for client-server communication, a server module, client modules, and a notebook for genetic dataset prediction, indicating a machine learning project.                                                                                                      |
| [requirements.txt](https://github.com/urochurro/FL-Research/blob/main/requirements.txt)                                                                     | The presented software project structure contains multiple Python scripts for client-server communication, data analysis notebooks, and utility functions. The dependency list in "requirements.txt" indicates the use of libraries for data handling (pandas, numpy), machine learning (scikit-learn), server communication (flwr, grpcio), kernel and notebook support (jupyter_client, ipykernel) and various utilities (psutil, cryptography, joblib).                                                                                                                      |
| [server.py](https://github.com/urochurro/FL-Research/blob/main/server.py)                                                                                   | The code implements a Federated Learning server using the Flower framework. It loads weights from a pre-trained model and uses them as initial parameters for client models. A custom aggregation strategy is defined that averages client's model accuracies, weighted by the number of samples each client used. The server is started on localhost and runs for five rounds of learning, requiring a minimum of two clients for each round.                                                                                                                                  |
| [utils.py](https://github.com/urochurro/FL-Research/blob/main/utils.py)                                                                                     | The `utils.py` script contains a data preprocessing function for a medical dataset. This process involves label encoding categorical data, normalizing numerical data, and dealing with class imbalance using the SMOTE method. It splits the dataset into features and target, oversamples it using SMOTE, encodes the target, and converts it into a binary matrix. The function returns the processed features and target ready for model training and testing.                                                                                                              |
| [client0.py](https://github.com/urochurro/FL-Research/blob/main/Client\client0.py)                                                                          | The code presents a client in a federated learning setup that runs a multi-layer perceptron (MLP) neural network for a prediction task on a dataset preprocessed via a utility method. It adjusts MLP weights using training data, and assesses accuracy on test data. The federated learning setup uses the Flower library to manage distribution of model parameters across clients. The client evaluates its local model, sends updated weights to a central server, and integrates updates it receives from the server.                                                     |
| [client1.py](https://github.com/urochurro/FL-Research/blob/main/Client\client1.py)                                                                          | The code represents a client in a federated learning network configured using the Flower library. It prepares a dataset, splits it for training/testing, and develops a Neural Network model using Keras. It defines a custom client class with specific fit, evaluate, and get_parameters methods for federated learning. The model is trained, tested, and evaluated on accuracy, returning weights to the server. It runs the federated learning client, connecting to a local server.                                                                                       |
| [client2.py](https://github.com/urochurro/FL-Research/blob/main/Client\client2.py)                                                                          | This code is part of a distributed machine learning system and represents a Flower client using Keras and TensorFlow libraries. It reads data from a CSV file and preprocesses it, splits it into training and test sets, then trains a deep learning model over 100 epochs. The client uses a sequential model with multiple dense layers and trains with the'categorical_crossentropy' loss function. The'HospitalClient' class enables this client to interface with the Flower server by setting/getting model weights, training the model, and evaluating its performance. |
| [client3.py](https://github.com/urochurro/FL-Research/blob/main/Client\client3.py)                                                                          | This code imports various Python packages and a utility function, reads data from a CSV file, then splits this data into training and testing subsets. It creates a multi-layered neural network model using Keras, compiled with attributes for loss, optimizer and metrics. The model is then integrated within a Flower client, which is used for federated learning. This client provides methods for: getting and setting model parameters, training the model and evaluating performance. The client then connects to a Server specified at a localhost IP address.       |

</details>

---

## 🚀 Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system:

`pip install flwr`

`pip install tensorflow`

### 🔧 Installation

1. Clone the FL-Research repository:
```sh
git clone https://github.com/urochurro/FL-Research
```

2. Change to the project directory:
```sh
cd FL-Research
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🤖 Running FL-Research

```sh
python server.py
```

Once the server is running we can start the clients in different terminals. Open a new terminal and start the first client:
```sh
python client0.py
```

Open another terminal and start the second client:
```sh
python client1.py
```

---

## 📄 License


This project is protected under the [MIT](https://choosealicense.com/licenses/mit) License. For more details, refer to the [LICENSE](LICENSE) file.

---

<!-- ## 👏 Acknowledgments

- List any resources, contributors, inspiration, etc. here. -->

[**Return**](#Top)

---

