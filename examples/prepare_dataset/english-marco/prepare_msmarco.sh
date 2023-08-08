set -e

dataset_dir="./data/datasets/msmarco-passage"

mkdir -p $dataset_dir
cd $dataset_dir

echo "download MSMARCO passage data"

echo "start downloading corpus, this may take some time depending on the network"
wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar --skip-old-files -zxvf collectionandqueries.tar.gz -C ./

echo "start downloading queries and qrels"

wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip ./msmarco-test2019-queries.tsv.gz 

wget -nc --no-check-certificate https://trec.nist.gov/data/deep/2019qrels-pass.txt

wget -nc --no-check-certificate https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gunzip ./msmarco-test2020-queries.tsv.gz 

wget -nc --no-check-certificate https://trec.nist.gov/data/deep/2020qrels-pass.txt

echo "finish downloading"

echo "Create symbolic link"

ln -sf collection.tsv corpus.tsv

ln -sf queries.train.tsv query.train
ln -sf qrels.train.tsv qrels.train

ln -sf queries.dev.small.tsv query.dev
ln -sf qrels.dev.small.tsv qrels.dev

ln -sf msmarco-test2019-queries.tsv query.trec19
ln -sf 2019qrels-pass.txt qrels.trec19

ln -sf msmarco-test2020-queries.tsv query.trec20
ln -sf 2020qrels-pass.txt qrels.trec20

echo "Done"

