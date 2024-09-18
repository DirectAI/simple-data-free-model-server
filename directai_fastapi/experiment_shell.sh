if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_number> [run_classification]"
    exit 1
fi

EXPERIMENT_NUMBER=$1

if [ "$2" == "run_classification" ]; then
    sudo cp breakfast_classifier_config.json ../.cache/breakfast_classifier_config_v${EXPERIMENT_NUMBER}.json
    sudo docker-compose build && sudo docker-compose run local_fastapi_ben python classify_directory.py --root=.cache/food101_validation_data --classifier_json_file=.cache/breakfast_classifier_config_v${EXPERIMENT_NUMBER}.json --eval_only=False --output_file=.cache/breakfast_output_v${EXPERIMENT_NUMBER}.csv --batch_size=64
fi

python analyze_breakfast_results.py -e $EXPERIMENT_NUMBER
