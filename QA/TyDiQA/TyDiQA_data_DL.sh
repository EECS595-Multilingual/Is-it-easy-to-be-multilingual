REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR


function download_tydiqa {
    echo "download tydiqa-goldp"
    base_dir=$DIR/tydiqa/
    mkdir -p $base_dir && cd $base_dir
    tydiqa_train_file=tydiqa-goldp-v1.1-train.json
    tydiqa_dev_file=tydiqa-goldp-v1.1-dev.tgz
    wget https://storage.googleapis.com/tydiqa/v1.1/${tydiqa_train_file} -q --show-progress
    wget https://storage.googleapis.com/tydiqa/v1.1/${tydiqa_dev_file} -q --show-progress
    tar -xf ${tydiqa_dev_file}
    rm ${tydiqa_dev_file}
    out_dir=$base_dir/tydiqa-goldp-v1.1-train
    python $REPO/utils_preprocess.py --data_dir $base_dir --output_dir $out_dir --task tydiqa
    mv $base_dir/$tydiqa_train_file $out_dir/
    echo "Successfully Download data at $DIR/tydiqa"  >> $DIR/download.log
}

download_tydiqa
