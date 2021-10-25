# lib imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Assume-se que todo dataset tera o formato semelhante ao do beans

def split_data_from(path, test_size: float, eval_size: float):
    df = pd.read_excel(path)
    y = df["Class"]
    y = y.astype("category").cat.codes
    X = df.iloc[:, :-1]
    size_output = y.nunique()
    size_input = len(X.columns)

    # normalizando X
    X = (X - X.min()) / (X.max() - X.min())

    # spliting
    X_forTraining, X_test, y_forTraining, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=42)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_forTraining, y_forTraining, stratify=y_forTraining, test_size=0.1, random_state=33)

    test_pack = (X_test , y_test)
    eval_pack = (X_eval , y_eval)
    train_pack = (X_train , y_train)
    size_pack = (size_input,size_output)

    return train_pack, eval_pack, test_pack, size_pack
