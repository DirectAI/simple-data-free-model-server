# simple-data-free-model-server

### On Pre-Commit Hooks
- Make sure you run `pip install pre-commit` followed by `pre-commit install` before attempting to commit to the repo.

### Launching Production Service
- Set your logging level preference in `directai_fastapi/.env`. See options on [python's logging documentation](https://docs.python.org/3/library/logging.html#levels). An empty string input defaults to `logging.INFO`.
- `sudo docker-compose build && sudo docker-compose up`

### Launching Integration Tests
- `docker-compose -f testing-docker-compose.yml build && docker-compose -f testing-docker-compose.yml up`

### Running Offline Batch Classification
We've built infrastructure to make it easy to quickly run an arbitrary classifier against a dataset. If your images are organized like so:

    /dataset_directory
    │
    ├── image1.jpg
    ├── image2.jpg
    ├── image3.jpg
    ├── ...
    └── imageN.jpg
and you have a JSON file defining the image classifier you'd like to run at `classifier_config.json`, you can dump classification labels to an `output.csv` via:

 - `docker-compose build && docker-compose run local_fastapi
   python classify_directory.py --root=dataset_directory
   --classifier_json_file=classifier_config.json --output_file=output.csv`

Make sure that all the files are mounted within the Docker container. You can do that by either modifying the volumes specified in `docker-compose.yml`, or by placing them all within the `.cache` directory which is mounted by default.

If your images have labels and are organized like so:

    /dataset_directory
    │
    ├── /label1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    ├── /label2
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    └── /labelN
        ├── image1.jpg
        ├── image2.jpg
        └── ...
You can run an evaluation against the labels by running the command

 - `docker-compose build && docker-compose run local_fastapi
   python classify_directory.py --root=dataset_directory
   --classifier_json_file=classifier_config.json --eval_only=True`

If you want to run classifications on a custom dataset, you can either use our API or build a custom Ray Dataset and use the utilities defined in `batch_processing.py`.