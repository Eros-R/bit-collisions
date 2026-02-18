# HERE BE THE MODELS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from enum import Enum
from data_preprocessing import *
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
# tried different from the DEFAULT call in dataset preproc. cast from from_string w/decorator
# w/e its fking python...

# maybe this in processibng?
def extract_features_targets(dataset, representation, bits):
    X_dict = dataset.get_folded_dataset(representation, bits)
    y_dict = dataset.get_targets_dict()

    ids = list(X_dict.keys())
    X = [X_dict[i] for i in ids]
    y = [y_dict[i] for i in ids]

    return X, y
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
# Close to default RF
def run_rf_model(representation, bits, n_estimators=200, max_depth=12, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', extract=True,
                 x_train=None, y_train=None, x_test=None, y_test=None, train=None, test=None):
    if extract:
        try:
            X_train, y_train = extract_features_targets(train, representation, bits)
            X_test, y_test = extract_features_targets(test, representation, bits)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to extract features/targets: {e}")
    else:
        X_train = x_train
        y_train = y_train
        X_test = x_test
        y_test = y_test

    print(f"running model: Random Forest {representation} {bits}")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion="absolute_error",
        random_state=1508,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
#juiced
def run_xgb_model(representation, bits, n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, gamma=1.0, min_child_weight=1, extract=True,
                  x_train=None, y_train=None, x_test=None, y_test=None, train=None, test=None):
    if extract:
        try:
            X_train, y_train = extract_features_targets(train, representation, bits)
            X_test, y_test = extract_features_targets(test, representation, bits)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to extract features/targets: {e}")
    else:
        X_train = x_train
        y_train = y_train
        X_test = x_test
        y_test = y_test

    print(f"running model: XGBoost {representation} {bits}")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        min_child_weight=min_child_weight,
        objective="reg:absoluteerror",
        random_state=1508,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

# Three-layer fully connected feedforward network with ReLU activations
def run_nn_model(representation, bits, hidden_dims=[256,128,64], epochs=200, batch_size=32, lr=1e-3,
                 extract=True, x_train=None, y_train=None, x_test=None, y_test=None, train=None, test=None):

    if extract:
        try:
            X_train, y_train = extract_features_targets(train, representation, bits)
            X_test, y_test = extract_features_targets(test, representation, bits)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to extract features/targets: {e}")
    else:
        X_train = x_train
        y_train = y_train
        X_test = x_test
        y_test = y_test

    print(f"running model: Neural Network {representation} {bits}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1,1)
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
        y_pred = model(X_test_tensor).cpu().numpy().flatten()

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
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


