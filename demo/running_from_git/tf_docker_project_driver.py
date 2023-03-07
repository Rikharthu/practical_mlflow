# Taken from this Medium post by George Novack
#   https://towardsdatascience.com/create-reusable-ml-modules-with-mlflow-projects-docker-33cd722c93c4

import mlflow

# FIXME: Running experiments in Docker seems to be supported only with K8s backend for now.
# mlflow.set_experiment("celeb-cnn-project")

# synchronous=false means that these run in parallel
mlflow.projects.run(
    'https://github.com/Rikharthu/celeb-cnn-project',
    backend='local',
    synchronous=False,
    parameters={
        'batch_size': 32,
        'epochs': 10,
        'convolutions': 3,
        'training_samples': 15000,
        'validation_samples': 2000,
        'randomize_images': True
    },
    build_image=True
)

# mlflow.projects.run(
#     'https://github.com/Rikharthu/celeb-cnn-project',
#     backend='local',
#     synchronous=False,
#     parameters={
#         'batch_size': 32,
#         'epochs': 10,
#         'convolutions': 2,
#         'training_samples': 15000,
#         'validation_samples': 2000,
#         'randomize_images': False
#     },
#     build_image=True
# )
#
# mlflow.projects.run(
#     'https://github.com/Rikharthu/celeb-cnn-project',
#     backend='local',
#     synchronous=False,
#     parameters={
#         'batch_size': 32,
#         'epochs': 10,
#         'convolutions': 0,
#         'training_samples': 15000,
#         'validation_samples': 2000,
#         'randomize_images': False
#     },
#     build_image=True
# )

# docker run --rm -it -e MLFLOW_RUN_ID=9df6ee23a5af4cb6b70730bce05cdcc4 -e MLFLOW_TRACKING_URI=http://localhost -e MLFLOW_EXPERIMENT_ID=5 -e AWS_SECRET_ACCESS_KEY=minio123 -e AWS_ACCESS_KEY_ID=minio -e MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 gnovack/celebs-cnn:latest