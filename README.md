# AIG2INV

Accelerating IC3 by inducive clauses prediction

### For important dir
**aag2graph (directly convert from aag):**
* `/data/hongcezh/clause-learning/data-collect/data2dataset`

**aag+inv to cnf**
* `/data/hongcezh/clause-learning/data-collect/aig2cexcnfmap`

**script to run ic3ref**
* `/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/run_single.sh`
* `/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/collect.sh`

**for ground truth**
* `/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-abc-result/output/tip/`
* `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/hwmcc20_abc_7200_result/`

### For deps
clone modified ic3ref to utils
https://github.com/zhanghongce/IC3ref - modified Makefile (may not be necessary)

> usage: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -v -f xxx.cnf < xxx.aig

### For scripts
* `build_data.py` : build data from aag+inv to graph
* `train_data.py` : train data from graph
* `main.py` : predict the induction invariant from SAT models
* `clean_trival_log.py` : clean the trivial logs (line <= 20)

### For dataset
* `dataset_20230106_014957_toy` : contains only one aiger's cex
* `dataset_20230106_025223_small`: contains 5 simplest aigers from hwmcc07
* `dataset_hwmcc07_almost_complete`: contains almost all aigers from hwmcc07
* `dataset_hwmcc07_0_1_2_3_4`: contains 25 simplest aigers from hwmcc07

### For converted aiger
* `case4test/hwmcc2007_big_comp_for_prediction`: hwmcc07 tip safety cases (only UNSAT cases) all aiger1.0 format, only use to dump predicted clauses
* `case4test/hwmcc2007`: hwmcc07 tip safety cases (all aiger1.0 format)
* `case4test/hwmcc2020_all`: hwmcc20 safety cases (all aiger1.0 format)