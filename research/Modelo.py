import warnings
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import neptune

run = neptune.init_run( 
    project="aetcheverry/titanic",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZDMzZjc3OC1kZThkLTQyM2ItYmIzNC04YTczNzczMjFlZDMifQ==",
    )

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
     
    train_path = "C:\\Users\\Lejam\\Downloads\\train.csv"
    train_data = pd.read_csv(train_path)
    
    train_data["Sex"] = pd.factorize(train_data["Sex"])[0]

    def get_error_metric(value, prediction, errormetric):
            if(errormetric ==  my_mae):
                return(mean_absolute_error(value,prediction))
            elif(errormetric == my_r2):
                return(r2_score(value, prediction))
            elif(errormetric == my_rmse):
                return(mean_squared_error(value,prediction))


  
    def get_error(max_leaf_nodes, train_X, val_X, train_y, val_y, errormetric):
        model = HistGradientBoostingRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        errormetric = get_error_metric(val_y, preds_val, errormetric)
        return(errormetric)

    train, test = train_test_split(train_data)
    train_x = train[["Pclass","Sex", "Age","Parch"]]
    test_x = test[["Pclass","Sex", "Age", "Parch"]]
    train_y = train[["Survived"]]
    test_y = test[["Survived"]]
    
    my_mae = 0
    my_r2 = 0
    my_rmse = 0
    minMae = 0
    minR2 = 0
    minRmse = 0

    for max_leaf_nodes in range(2,25):
        my_mae = get_error(max_leaf_nodes, train_x, test_x, train_y, test_y, my_mae)
        my_r2 = get_error(max_leaf_nodes, train_x, test_x, train_y, test_y, my_r2)
        my_rmse = get_error(max_leaf_nodes, train_x, test_x, train_y, test_y, my_rmse)
        if((my_mae < minMae and my_r2 < minR2) or (my_mae < minMae and my_rmse < minRmse) or (my_r2 < minR2 and my_rmse < minRmse)) or (max_leaf_nodes == 2):
            minLeafNodes = max_leaf_nodes
            minMae = my_mae
            minR2 = my_r2
            minRmse = my_rmse
    
    print(minLeafNodes)

    params = {
        "max_leaf_nodes" : minLeafNodes,
        "random_state" : 1,
    }

    run["parameters"] = params

    run["train"] = train_x
    run["test"] = test_x

    lr = HistGradientBoostingClassifier(**params)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    for epoch in range(100):
        run["train/accuracy"].append(r2)

    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")

    for epoch in range(100):
        run["train/accuracy"].append(r2)

    predictions = lr.predict(train_x)

    run.stop()