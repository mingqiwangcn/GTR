if [ "$#" -ne 3 ]; then
    echo "Usage: ./eval.sh <train_dataset> <real/syt> <test_dataset>"
    exit
fi
train_dataset=$1
mode=$2
test_dataset=$3
python run.py --exp eval --config configs/${train_dataset}_eval_${mode}_${test_dataset}.json
