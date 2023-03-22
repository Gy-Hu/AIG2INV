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
* ic3ref hwmcc20: `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/hwmcc20-ic3ref_7200-result/`

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
* `tool_box.py` : some useful functions (e.g. clean trivial log, counterexample cube visualization)
* `utils/fetch_aiger.py` : fetch aiger, this script must be ran in the same dir as `utils`

### Usage
**build dataset**
* Example command to construct hwmcc20 abc training data:
    * `python build_data.py --model-checker abc --dataset-folder-prefix dataset_hwmcc20_big_abc_1 --simplification-label slight --benchmark hwmcc2020_all --ground_truth_folder_prefix /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/hwmcc20_abc_7200_result --subset_range 1`
    * `python build_data.py --model-checker abc --dataset-folder-prefix dataset_hwmcc20_small_abc_1 --simplification-label slight --benchmark hwmcc2020_small --ground_truth_folder_prefix /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/hwmcc20_abc_7200_result --subset_range 1`
* Example command to construct hwmcc07 ic3ref training data:
    * `python build_data.py --model-checker ic3ref --dataset-folder-prefix dataset_hwmcc07_big_ic3ref_1 --simplification-label slight --benchmark hwmcc2007 --ground_truth_folder_prefix /data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/ --subset_range 1`
    * `python build_data.py --model-checker ic3ref --dataset-folder-prefix dataset_hwmcc07_small_ic3ref_1 --simplification-label slight --benchmark hwmcc2007_small --ground_truth_folder_prefix /data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/ --subset_range 1`

### For dataset
* `dataset_hwmcc07_toy_no_simplification` : contains only one aiger's cex
* `dataset_hwmcc07_small_no_simplification`: contains 5 simplest aigers from hwmcc07
* `dataset_{BENCHMARK}_{MODEL_CHECKER}_{SIMPLIFICATION_LEVEL}_{SUBSET_RANGE}`: contains aigers of `{BENCHMARK}` with `{SIMPLIFICATION_LEVEL}` simplification, and ground truth from `{MODEL_CHECKER}` model checker

### For converted aiger
* `cnt1` , `cnt2` and `cnt-zeros` : For toy experiments
* `benchmark_folder/hwmcc2007_all`: hwmcc07 tip all safety cases, including both UNSAT and SAT cases (all aiger1.0 format)
* `benchmark_folder/hwmcc2007_all_comp_for_prediction_ic3ref`: hwmcc07 tip safety cases (only UNSAT cases) all aiger1.0 format, only use to dump predicted clauses, for ic3ref prediction
* `benchmark_folder/hwmcc2020_all`: hwmcc20 safety cases (all aiger1.0 format)
* `benchmark_folder/hwmcc2020_small`: part of hwmcc20 safety cases (all aiger1.0 format), only contains 50 simple cases
* `benchmark_folder/hwmcc2020_all_without_sat_unknown`: hwmcc20 safety cases without SAT and UNKNOWN cases (all aiger1.0 format)

### For validate result
* `case4comp/xxx`: contains the aiger and its predicted clauses

### Simplification Level
* `thorough`: use sympy in transition relation simplification + aig operator simplification during transition relation construction + z3 simplification + counterexample cube simplification
* `deep`: use sympy in transition relation simplification + aig operator simplification during transition relation construction + z3 simplification
* `moderate`: aig operator simplification during transition relation construction + z3 simplification
* `slight`: z3 simplification + ternary simulation
* `naive`: only use sympy to simplify the counterexample cube
