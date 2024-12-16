import pandas as pd
from sklearn import datasets

def load_and_save():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv('data/raw/iris.csv', index=False)

if __name__ == "__main__":
    load_and_save()
