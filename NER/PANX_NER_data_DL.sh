REPO=$PWD
DIR=$REPO/download/panx
mkdir -p $DIR
echo "Download panx NER dataset"
if [ -f $DIR/AmazonPhotos.zip ]; then
    unzip -qq $DIR/AmazonPhotos.zip -d $DIR/
    base_dir=$DIR  
    mkdir -p $base_dir
    cd $base_dir
    langs=(en fi ar bn id ko ru sw te)
    for lg in ${langs[@]}; do
        tar xzf $base_dir/${lg}.tar.gz
        for f in dev test train; do mv $base_dir/$f $base_dir/${lg}-${f}; done
    done
    python $REPO/utils_preprocess.py \
    --data_dir $base_dir/NER_data \
        --output_dir $DIR/panx \
        --task panx
    #rm -rf $base_dir
    echo "Successfully download data at $DIR/panx" >> $DIR/download.log
else
    echo "Please download the AmazonPhotos.zip file on Amazon Cloud Drive mannually and save it to $DIR/AmazonPhotos.zip"
    echo "https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN"
fi