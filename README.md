aag2graph (directly convert from aag):
/data/hongcezh/clause-learning/data-collect/data2dataset

aag+inv to cnf
/data/hongcezh/clause-learning/data-collect/aig2cexcnfmap

script to run ic3ref
/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/run_single.sh
/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/collect.sh

clone modified ic3ref to utils
https://github.com/zhanghongce/IC3ref - modified Makefile (may not be necessary)

usage: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -v -f xxx.cnf < xxx.aig

build_data.py: build data from aag+inv to graph

train_data.py: train data from graph

main.py: predict the induction invariant from SAT models

clean_trival_log.py: clean the trivial logs (line <= 20)