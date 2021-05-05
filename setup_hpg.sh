export PYTHONPATH=${PYTHONPATH}:${PWD}/
export BASE_PATH=${PWD}

if [[ "$HOSTNAME" == login*ufhpc ]]; then
    echo "Loading modules"
    module load tensorflow/2.4.1
    module load git
fi
