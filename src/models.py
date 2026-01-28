# HERE BE THE MODELS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from enum import Enum
from data_preprocessing import *
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# tried different from the DEFAULT call in dataset preproc. cast from from_string w/decorator
# w/e its fking python...

def load_rf_model(n_estimators = 100, max_depth = None, criterion = "absolute_error", random_state = 1508):
    return RandomForestRegressor(n_estimators= n_estimators, max_depth = max_depth, criterion=criterion,random_state=random_state)
# maybe this in processibng?
def extract_features_targets(dataset, representation, bits):
    X_dict = dataset.get_folded_dataset(representation, bits)
    y_dict = dataset.get_targets_dict()

    ids = list(X_dict.keys())
    X = [X_dict[i] for i in ids]
    y = [y_dict[i] for i in ids]

    return X, y

# Close to default RF
def run_rf_model(train, test, representation, bits, n_estimators=100, max_depth=None):
    try:
        X_train, y_train = extract_features_targets(train, representation, bits)
        X_test, y_test = extract_features_targets(test, representation, bits)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to extract features/targets: {e}")
    print(f"running model: random forest {representation} {bits}")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion="absolute_error",
        random_state=1508
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    return results
# Sameish to RF
def run_xgb_model(train, test, representation, bits, n_estimators=100, max_depth=6, learning_rate=0.1):
    try:
        X_train, y_train = extract_features_targets(train, representation, bits)
        X_test, y_test = extract_features_targets(test, representation, bits)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to extract features/targets: {e}")

    print(f"running model: XGBoost {representation} {bits}")

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:absoluteerror", #abs error should be opt target.
        random_state=1508
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    return results

# Three-layer fully connected feedforward network with ReLU activations
def run_nn_model(train, test, representation, bits, hidden_dims=[128, 64], epochs=100, batch_size=32, lr=1e-3):
    try:
        X_train, y_train = extract_features_targets(train, representation, bits)
        X_test, y_test = extract_features_targets(test, representation, bits)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to extract features/targets: {e}")

    print(f"running model: Neural Network {representation} {bits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    layers = []
    input_dim = X_train_tensor.shape[1]
    for h in hidden_dims:
        layers.append(nn.Linear(input_dim, h))
        layers.append(nn.ReLU())
        input_dim = h
    layers.append(nn.Linear(input_dim, 1))
    model = nn.Sequential(*layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()

    results = {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    return results
class ModelType(Enum):
    randomforest = ("randomforest", run_rf_model)
    xgb = ("xgb", run_xgb_model)
    nn = ("nn", run_nn_model)

    def __init__(self, label, loader_fn):
        self.label = label
        self.loader = loader_fn

    @classmethod
    def from_string(cls, s):
        try:
            return cls[s] #lazy subscript instead of matching fiwb
        except ValueError:
            raise ValueError(f"Unrecognized model: {s}")


