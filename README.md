# AIG2INV

Accelerating IC3 by inducive clauses prediction

### For important dir
**aag2graph (directly convert from aag):**
* `/data/hongcezh/clause-learning/data-collect/data2dataset`

**aag+inv to cnf**
* `/data/hongcezh/clause-learning/data-collect/aig2cexcnfmap`

**script to run ic3ref in collecting hwmcc07 dataset**
* `/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/run_single.sh`
* `/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/collect.sh`

**for ground truth**
* abc hwmcc07: `/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-abc-result/output/tip/`
* abc hwmcc20: `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/hwmcc20_abc_7200_result/`
* ic3ref hwmcc07: `/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`
* ic3ref hwmcc20: `/data/hongcezh/clause-learning/data-collect/hwmcc20-7200-result/output/aig`

### For deps

**For modified ic3ref:**

clone modified ic3ref to utils
https://github.com/zhanghongce/IC3ref - modified Makefile (may not be necessary)

> usage: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -v -f xxx.cnf < xxx.aig

**For modified abc:**

clone modified abc to utils - https://github.com/zhanghongce/abc

> usage: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/abc/abc -c "&r xxx.aig; &put; fold ; pdr -v"


### For scripts
* `build_data.py` : build data from aag+inv to graph
* `train_data.py` : train data from graph
* `main.py` : predict the induction invariant from SAT models
* `tool_box.py` : some useful functions (e.g. clean trivial log)
* `utils/fetch_aiger.py` : fetch aiger, this script must be ran in the same dir as `utils`

### Usage
**build dataset**
* Example command to construct hwmcc20 abc training data:
    * `python build_data.py --model-checker abc --dataset-folder-prefix dataset_hwmcc20_big --simplification-label slight --benchmark hwmcc2020_all --ground_truth_folder_prefix /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/hwmcc20_abc_7200_result`
    * `python build_data.py --model-checker abc --dataset-folder-prefix dataset_hwmcc20_small --simplification-label slight --benchmark hwmcc2020_small --ground_truth_folder_prefix /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/hwmcc20_abc_7200_result`
* Example command to construct hwmcc07 ic3ref training data:
    * `python build_data.py --model-checker ic3ref --dataset-folder-prefix dataset_hwmcc07_big --simplification-label slight --benchmark hwmcc2007 --ground_truth_folder_prefix /data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`
    * `python build_data.py --model-checker ic3ref --dataset-folder-prefix dataset_hwmcc07_small --simplification-label slight --benchmark hwmcc2007_small --ground_truth_folder_prefix /data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`

### For dataset
* `dataset_hwmcc07_toy_no_simplification` : contains only one aiger's cex
* `dataset_hwmcc07_small_no_simplification`: contains 5 simplest aigers from hwmcc07

### For converted aiger
* `cnt1` , `cnt2` and `cnt-zeros` : For toy experiments
* `case4test/hwmcc2007_all_comp_for_prediction_ic3ref`: hwmcc07 tip safety cases (only UNSAT cases) all aiger1.0 format, only use to dump predicted clauses, for ic3ref prediction
* `case4test/hwmcc2007_all`: hwmcc07 tip all safety cases, including both UNSAT and SAT cases (all aiger1.0 format)
* `case4test/hwmcc2020_all`: hwmcc20 safety cases (all aiger1.0 format)
* `case4test/hwmcc2020_all_without_sat_unknown`: hwmcc20 safety cases without SAT and UNKNOWN cases (all aiger1.0 format)