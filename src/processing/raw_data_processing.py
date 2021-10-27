# lib imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Assume-se que todo dataset tera o formato semelhante ao do beans

def get_dataframe_from(path):
    return pd.read_excel(path) # Precisamos dessa funcao ainda? Só faz o read

def df_category_split(df:pd.DataFrame):
    y = df["Class"]
    y = y.astype("category")

    # TODO what do I do with it? Como eu passo?
    num_to_type_list = y.cat.categories.to_list() # Aqui a lista que tranforma o numero no tipo
    
    y = y.cat.codes
    X = df.drop("Class", axis=1) # Isso não prejudica y pois este já é outro objeto
    
    size_output = len(num_to_type_list)
    size_input = len(X.columns)
    size_pack = (size_input,size_output)

    return X, y, size_pack, num_to_type_list

def normalize_df(df:pd.DataFrame):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

# tet = Train, Eval, Test
def df_tet_split(df:pd.DataFrame, test_size: float, eval_size: float):
    X, y, size_pack, num_to_type_list = df_category_split(df)
    X = normalize_df(X)

    # spliting
    X_forTraining, X_test, y_forTraining, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=42)

    X_train, X_eval, y_train, y_eval = train_test_split(
        X_forTraining, y_forTraining, stratify=y_forTraining, test_size=eval_size, random_state=33)

    test_pack = (X_test, y_test)
    eval_pack = (X_eval, y_eval)
    train_pack = (X_train, y_train)

    return train_pack, eval_pack, test_pack, size_pack, num_to_type_list
