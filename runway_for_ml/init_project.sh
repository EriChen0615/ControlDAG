#!bin/bash

echo "[INFO]Runway Initialization Starts!"

RUNWAY_ASSET=runway_for_ml/assets

mkdir cache
mkdir data
mkdir third_party
mkdir experiments
mkdir configs 
mkdir scripts

mkdir src 
mkdir src/data_ops
cp $RUNWAY_ASSET/__init__.py src/data_ops/__init__.py
mkdir src/executors
cp $RUNWAY_ASSET/__init__.py src/executors/__init__.py
mkdir src/models
cp $RUNWAY_ASSET/__init__.py src/models/__init__.py

cp $RUNWAY_ASSET/example_configs/* configs/
cp $RUNWAY_ASSET/example_scripts/* scripts/
cp $RUNWAY_ASSET/main.py src/
cp $RUNWAY_ASSET/example_data_ops/* src/data_ops/
cp $RUNWAY_ASSET/example_executors/* src/executors/

echo "[INFO]Skeleton Initialized. You can check example configuration files in config/ and codes in src/data_ops/ and src/executors/"
echo "[INFO]Runway Initialization Completes!"



