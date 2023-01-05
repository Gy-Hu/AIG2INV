aag2graph (directly convert from aag):
/data/hongcezh/clause-learning/data-collect/data2dataset

aag+inv to cnf
/data/hongcezh/clause-learning/data-collect/aig2cexcnfmap

clone modified ic3ref to utils
https://github.com/zhanghongce/IC3ref - modified Makefile (may not be necessary)

usage: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -v -f xxx.cnf < xxx.aig

build_data.py: build data from aag+inv to graph

train_data.py: train data from graph

main.py: predict the induction invariant from SAT models