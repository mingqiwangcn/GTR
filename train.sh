if [ "$#" -ne 2 ]; then
    echo "Usage: ./train.sh <dataset> <real or syt>"
    exit
fi
dataset=$1
mode=$2
python run.py --exp train_test --config configs/${dataset}_train_${mode}.json
