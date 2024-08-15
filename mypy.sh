BUILD=false
MYPY_FULL_APP=false
MYPY_FULL_APP=false

while getopts "b" opt; do
  case $opt in
    b)
      BUILD=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

build_app() {
    if $1; then
        docker compose -f docker-compose.yml build
    fi
    echo "running mypy on local_fastapi"
    fastapi_output=$(docker run --rm simple-data-free-model-server-local_fastapi:latest mypy .)
    echo "$fastapi_output"
    if echo "$fastapi_output" | grep -q "Success: no issues found"; then
        echo "local_fastapi looks good"
    else
        echo "local_fastapi failed. exiting."
        exit 1
    fi
}

build_testing() {
    if $1; then
        docker compose -f testing-docker-compose.yml build
    fi
    echo "running mypy on local_tests"
    integration_testing_output=$(docker run --rm simple-data-free-model-server-tests:latest mypy /integration_tests)
    echo "$integration_testing_output"
    if echo "$integration_testing_output" | grep -q "Success: no issues found"; then
        echo "tests looks good"
    else
        echo "tests failed. exiting."
        exit 1
    fi
}

build_app $BUILD
build_testing $BUILD
