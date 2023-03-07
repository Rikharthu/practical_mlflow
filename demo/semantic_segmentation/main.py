import click
import mlflow

_steps = [
    "train_model",
    "download_model"
]


@click.command()
@click.option("--steps", default="all", type=str)
def main(steps):
    # Steps to execute
    active_steps = steps.split(",") if steps != "all" else _steps
    print(f"Active steps: {active_steps}")

    with mlflow.start_run(run_name="pipeline", nested=True):

        # TODO: Add download data run as a separate step

        if "train_model" in active_steps:
            print("Training model")

            train_model_run = mlflow.run(
                ".",
                "train_model",
                run_name="train_model",
                parameters={}
            )
            train_model_run_id = train_model_run.run_id

            # If we would need to extract some data from this run:
            # train_model_run = mlflow.tracking.MlflowClient().get_run(train_model_run_id)
            # model_uri = train_model_run.params['local_folder']

        if "download_model" in active_steps:
            print("Downloading model")
            download_model_run = mlflow.run(
                ".",
                "download_model",
                run_name="download_model",
                # MLflow will automatically log these parameters as well
                parameters={
                    "run_id": train_model_run_id
                }
            )


if __name__ == "__main__":
    main()
