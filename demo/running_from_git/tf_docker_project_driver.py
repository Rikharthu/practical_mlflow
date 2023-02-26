import mlflow

mlflow.set_experiment("celeb-cnn-project")

mlflow.projects.run(
    'https://github.com/gnovack/celeb-cnn-project',
    backend='local',
    synchronous=False,
    parameters={
        'batch_size': 32,
        'epochs': 10,
        'convolutions': 3,
        'training_samples': 15000,
        'validation_samples': 2000,
        'randomize_images': True
    })

mlflow.projects.run(
    'https://github.com/gnovack/celeb-cnn-project',
    backend='local',
    synchronous=False,
    parameters={
        'batch_size': 32,
        'epochs': 10,
        'convolutions': 2,
        'training_samples': 15000,
        'validation_samples': 2000,
        'randomize_images': False
    })

mlflow.projects.run(
    'https://github.com/gnovack/celeb-cnn-project',
    backend='local',
    synchronous=False,
    parameters={
        'batch_size': 32,
        'epochs': 10,
        'convolutions': 0,
        'training_samples': 15000,
        'validation_samples': 2000,
        'randomize_images': False
    })
