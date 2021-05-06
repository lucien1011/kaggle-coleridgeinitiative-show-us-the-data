export PYTHONPATH=${PYTHONPATH}:${PWD}/
export BASE_PATH=${PWD}

if [[ "$HOSTNAME" == login*ufhpc ]]; then
    echo "Loading modules"
    #module load tensorflow/2.4.1
    module load python/3.8
    module load cuda/11.1.0
    module load git
fi
