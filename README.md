# simple-data-free-model-server

### On Pre-Commit Hooks
- Make sure you run `pip install pre-commit` followed by `pre-commit install` before attempting to commit to the repo.

### Launching Production Service
- `sudo docker-compose build && sudo docker-compose up`

### Launching Integration Tests
- `docker-compose -f testing-docker-compose.yml build && docker-compose -f testing-docker-compose.yml up`