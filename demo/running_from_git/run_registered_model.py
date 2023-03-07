import mlflow
import pandas as pd


def main():
    # Load model by version
    model_name = "Git_ElasticNet"

    model_version = 1
    model1 = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )
    print("\nmodel1:")
    print(model1)

    # Load model by stage
    stage = "Staging"
    model2 = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{stage}"
    )

    print("\nmodel2:")
    print(model2)

    print("\ndata:")
    data = pd.read_csv("./wine_quality_test.csv", index_col=None)
    X = data.drop("quality", axis=1)
    y = data["quality"].tolist()
    print(data.head())

    predictions1 = model1.predict(X)
    predictions2 = model2.predict(X)
    print(f"predictions1: {predictions1}")
    print(f"predictions2: {predictions2}")
    print(f"actual:       {y}")


if __name__ == "__main__":
    main()
