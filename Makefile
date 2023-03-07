#start_mlflow_ui:
#	mlflow ui

start_mlflow:
	cd ./mlflow_docker_setup && docker-compose up -d --build

stop_mlflow:
	cd ./mlflow_docker_setup && docker-compose down

export_vars:
	export MLFLOW_TRACKING_URI=http://192.168.0.97
	export MLFLOW_S3_ENDPOINT_URL=http://192.168.0.97:9000
	export AWS_ACCESS_KEY_ID="minio"
	export AWS_SECRET_ACCESS_KEY="minio123"