#!/bin/bash
BUILD=false
MYPY_FULL_APP=false
MYPY_TESTS=false



while getopts "bta" opt; do
  case $opt in
    b)
      BUILD=true
      ;;
    t)
      MYPY_TESTS=true
      ;;
    a)
      MYPY_FULL_APP=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

if ! $MYPY_FULL_APP && ! $MYPY_TESTS; then
  MYPY_FULL_APP=true
  MYPY_TESTS=true
fi


build_app() {
    if $1; then
        sudo docker-compose -f docker-compose.yml build
    fi
    echo "running mypy on local_fastapi"
    fastapi_output=$(sudo docker run --rm simple-data-free-model-server_local_fastapi:latest mypy . --disallow-untyped-defs --disallow-incomplete-defs)
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
        sudo docker-compose -f docker-compose.yml build
    fi
    echo "running mypy on local_tests"
    integration_testing_output=$(docker run --rm simple-data-free-model-server_tests:latest mypy integration_tests/test.py test_modules --disallow-untyped-defs --disallow-incomplete-defs)
    echo "$integration_testing_output"
    if echo "$integration_testing_output" | grep -q "Success: no issues found"; then
        echo "tests looks good"
    else
        echo "tests failed. exiting."
        exit 1
    fi
}

if $MYPY_FULL_APP; then
    build_app $BUILD
fi
# if $MYPY_TESTS; then
#     build_testing $BUILD
# fi
