# AIG2INV

Accelerating IC3 by inducive clauses prediction

### WIP
* [x] `dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22` test
* [ ] `dataset_hwmcc2007_tip_abc_no_simplification_0-22` test
* [ ] `dataset_hwmcc2020_all_only_unsat_abc_no_simplification_0-38` test
* [x] `dataset_hwmcc2020_all_only_unsat_ic3ref_no_simplification_0-38` test


### For important dir

**for ground truth**
* abc hwmcc07: `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-abc-result/output/tip/`
* abc hwmcc20: `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/ground_truth/hwmcc20_abc_7200_result/`
* ic3ref hwmcc07: `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`
* ic3ref hwmcc20: `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/ground_truth/hwmcc20-ic3ref_7200-result/`

### For deps

**For modified ic3ref:**

clone modified ic3ref to utils
https://github.com/zhanghongce/IC3ref - modified Makefile (may not be necessary)

> usage: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/IC3ref/IC3 -v -f xxx.cnf < xxx.aig

**For modified abc:**

clone modified abc to utils - https://github.com/zhanghongce/abc

> usage: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/utils/abc/abc -c "&r xxx.aig; &put; fold ; pdr -v"

**For construct benchmark and ground truth:**
`clause-learning`: Contains tables that can be used to construct the benchmark. And it also contains the inv.cnf ground truth.

### For scripts
* `build_data.py` : build data from aag+inv to graph
* `train_neurograph/train.py` : train data from graph
* `main.py` : predict the induction invariant from SAT models
* `tool_box.py` : some useful functions (e.g. clean trivial log, counterexample cube visualization)
* `utils/fetch_aiger.py` : fetch aiger, this script must be ran in the same dir as `utils`
* `utils/graph_size_comp.py` : compare the graph size of different simplification level

### Example Usage (Details in [USAGE.md](./USAGE.md))
**build dataset**
* Example command to construct hwmcc20 abc training data:
    * `python build_data.py --model-checker abc --simplification-label slight --benchmark hwmcc2020_all --ground_truth_folder_prefix /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/ground_truth/hwmcc20_abc_7200_result --subset_range 1`

**validate the prediction**
* `python main.py --threshold 0.5 --selected-built-dataset dataset_hwmcc2020_small_abc_slight_1 --NN-model neuropdr_2023-01-06_07:56:51_last.pth.tar --gpu-id 1 --compare_with_abc --re-predict`

### For dataset
* `dataset_{BENCHMARK}_{MODEL_CHECKER}_{SIMPLIFICATION_LEVEL}_{SUBSET_RANGE}`: contains aigers of `{BENCHMARK}` with `{SIMPLIFICATION_LEVEL}` simplification, and ground truth from `{MODEL_CHECKER}` model checker

### For converted aiger
* `cnt1` , `cnt2` and `cnt-zeros` : For toy experiments
* `benchmark_folder/hwmcc2007_all`: hwmcc07 tip all safety cases, including both UNSAT and SAT cases (all aiger1.0 format)
* `benchmark_folder/hwmcc2020_all`: hwmcc20 safety cases (all aiger1.0 format)
* `benchmark_folder/hwmcc2020_small`: part of hwmcc20 safety cases (all aiger1.0 format), only contains 50 simple cases
* `benchmark_folder/hwmcc2020_all_only_unsat`: hwmcc20 safety cases without SAT and UNKNOWN cases (all aiger1.0 format)

### For validate result
* `case4comp/xxx_comp`: contains the aiger and its predicted clauses (`xxx` normally is the corresponding dataset name)

### Simplification Level
* `thorough`: use sympy in transition relation simplification + aig operator simplification during transition relation construction + z3 simplification + counterexample cube simplification
* `deep`: use sympy in transition relation simplification + aig operator simplification during transition relation construction + z3 simplification
* `moderate`: aig operator simplification during transition relation construction + z3 simplification
* `slight`: z3 simplification + ternary simulation
* `naive`: only use sympy to simplify the counterexample cube

### For Log
* `log/error_handle/abnormal_header.log.xxx`: contains the aiger that has abnormal header (e.g. `SAT`)
* `log/error_handle/bad_model.log`: contains the aiger that has bad model (e.g. `v2==T, false, v4==F`, SAT model contains `false`)
* `log/error_handle/graph_pickle_incomplete.log`: contains the aiger that has incomplete graph pickle (e.g. number of CTI is not equal to number of graph generated)
* `log/error_handle/mismatched_inv.log.xxx`: contains the aiger that has mismatched inv (e.g. could not find a inductive clause in the inv to block the counterexample)