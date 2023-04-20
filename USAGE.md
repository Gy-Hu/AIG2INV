# How to use the scripts

## Generate Benchmark (Simple preprocessing)

* `utils/fetch_aiger.py` : Fetch the benchmark from the hwmcc, and simplly preprocess it (processed data should be in `benchmark_folder/` folder)
    * **usage:** `cd utils && python fetch_aiger.py`
    * **note:** 
        * change `aag_dir` (dump folder), `csv_file` (running info and aiger location prefix) and `aag_list[i]` (aiger files location) in the script to fetch different benchmarks. Modify `fetch_aig_from_csv` to ajust the preprocessing methods (only unsat? only sat? Only hard cases?). Modify `simplify` to determine whether convert `aig` to `aag`
        * Comment out `aag_list = [aag for aag in aag_list if aag.split('/')[-1] in hard_aag_list]` if we don't want to filter out the some cases
        * Choose what filter method you want to use in `fetch_aig_from_csv()`

## Build Dataset (Cex -> Graph)

* `build_dataset.py` : Build the dataset from the benchmark
    * **usage:** `python build_dataset.py --model-checker <model-checker> --simplification-level <simplification-level> --benchmark <benchmark> --ground_truth_folder_prefix <ground_truth_folder_prefix> --subset_range <subset_range>`
    * **parameters:**
        * `--model-checker` : The model checker to use (e.g. `abc`, `ic3ref`)
        * `--simplification-level` : The simplification level to use (e.g. `thorough`, `deep`, `moderate`, `slight` , `naive`)
        * `--benchmark` : The benchmark to use in benchmark_folder (e.g. `hwmcc2007_all`, `hwmcc2020_all`, `hwmcc2020_all_only_unsat`, `hwmcc2020_small`)
        * `--ground_truth_folder_prefix` : The prefix of the ground truth folder (e.g. `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/ground_truth/hwmcc20_abc_7200_result`, `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/ground_truth/hwmcc20_abc_7200_result`, `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`, `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`)
        * `--subset_range` : The range of data in benchmark to generate (e.g. `1`, `23`)
    * **note:**
        * There exists a list named `AigCaseBlackList`, which can exclude some cases from the benchmark (e.g. using `utils/check_proc.sh` to find those `hard-extraction` cases, and add them to the list. `echo "$count"` or `echo "$aag_files_and_count"` after sourcing)
        * Delete the existing `dataset_xxxx/` folder before running the script

## Train the Model

select which model to use (currently only `neurograph` is available)

* `train_neurograph/train.py`: Train the model
    * **usage:** `python train.py --task-name <task-name> --local_rank <local_rank> --dim <dim> --n_rounds <n_rounds> --epochs <epochs> --inf_dev <inf_dev> --gen_log <gen_log> --log-dir <log-dir> --model-dir <model-dir> --data-dir <data-dir> --restore <restore> --train-file <train-file> --val-file <val-file> --mode <mode> --gpu-id <gpu-id> --batch-size <batch-size> --pos-weight <pos-weight> --lr <lr>`
    * **parameters:**
        * `--task-name` : The task name (e.g. `neuropdr`)
        * `--local_rank` : The local rank for dpp (e.g. `-1`)
        * `--dim` : The dimension of variable and clause embeddings (e.g. `128`)
        * `--n_rounds` : The number of rounds of message passing (e.g. `26`)
        * `--epochs` : The number of epochs (e.g. `10`)
        * `--inf_dev` : The device to use (e.g. `gpu`)
        * `--gen_log` : The log file to use (e.g. `log/data_maker.log`)
        * `--log-dir` : The log folder dir (e.g. `log/`)
        * `--model-dir` : The model folder dir (e.g. `neurograph_model/`)
        * `--data-dir` : The data folder dir (e.g. `dataset/`)
        * `--restore` : The model to continue training (e.g. `neuropdr_2021-01-06_07:56:51_last.pth.tar`)
        * `--train-file` : The train file dir (e.g. `dataset_hwmcc2020_small_abc_slight_1`)
        * `--val-file` : The validation file dir (e.g. `dataset_hwmcc2020_small_abc_slight_1`)
        * `--mode` : The mode to use (e.g. `train`, `debug`)
        * `--gpu-id` : The gpu id to use (e.g. `0`,`1`)
        * `--batch-size` : The batch size to use (e.g. `2`)
        * `--pos-weight` : The positive weight in BCEWithLogitsLoss (e.g. `1.0`)
        * `--lr` : The learning rate to use (e.g. `0.00001`)

* `train_gcn/train.py`
    parser.add_argument('--dataset', type=str, default=None, help='dataset name') # no need if load pickle
    parser.add_argument('--load-pickle', type=str, default=None, help='load pickle file name')
    parser.add_argument('--dump-pickle-name', type=str, default=None, help='dump pickle file name') # no need if load pickle
    parser.add_argument('--model-name', type=str, default=None, help='model name to save')
    * **usage:** `python train.py --dataset <dataset>  (optional: --load-pickle, --model-name, --dump-pickle-name <dump-pickle-name> --dual , etc.)`
    * **parameters:**
        * `--dataset` : The dataset to use (e.g. `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22/bad_cube_cex2graph/expr_to_build_graph/`)
        * `--load-pickle` : The pickle file to load (e.g. `hwmcc07_tip_ic3ref_no_simplification_0-22.pickle`)
        * `--dump-pickle-name` : The pickle file to dump (e.g. `hwmcc07_tip_ic3ref_no_simplification_0-22.pickle`)
        * `--model-name` : The model name to save (e.g. `hwmcc07_ic3ref_tip_no_simplification_0-22.pt`)
        * `--dual` : Use dual GNN or not (One for pre-defined feature, one for unsuperivsed embedding)
    * **note:**
        * Adjust `DATASET_SPLIT` if you just want to test the code
        * Adjust `h = self.dropout(h)` in `train_gcn/GNN_Model.py` and `WEIGHT_DECAY` in `train_gcn/train.py` if you want to prevent overfitting
        * Define how to split the dataset by yourself (using `train_test_split()`)
        * Uncomment `prof.step()` in training loop if you want to profile the code
        * Check `threshold_finder = ThresholdFinder(val_dataloader, model, device)`, confirm which dataloader you want to use in testing
        * Check `Validation loop` in training loop, uncomment if you need it (e.g. early stop or hyperparameter fine tuning)
        * Check node embedding method that you want to employ in `graph_list = employ_graph_embedding(graph_list,args)`
        * Comment out `default args` if you want to input them in command line
        * Comment out `G = G.to_undirected()` if you want to use directed graph
        * Uncomment `graph_list_struc_feat = employ_graph_embedding(graph_list,args)` if you want to use additional node embedding
        * If using `dual` GNN, choose the GNN model start with `dual` (), other will fail. (e.g.`DualGCNModle`,`DualGraphSAGEModel`,etc.) 

## Validate the Model

* `main.py` : Validate the model
    * **usage:** `python main.py --threshold <threshold> --selected-built-dataset <selected-built-dataset> --NN-model <NN-model> --gpu-id <gpu-id>` (optional: `--compare_with_abc`, `--re-predict`, etc.)
    * **parameters:** 
        * `--threshold` : The threshold to use (e.g. `0.5`)
        * `--selected-built-dataset` : The built dataset with graph generated to use (e.g. `dataset_hwmcc2020_small_abc_slight_1`)
        * `--NN-model` : The NN model to use (e.g. `neuropdr_2023-01-06_07:56:51_last.pth.tar`)
        * `--gpu-id` : The gpu id to use (e.g. `0`,`1`)
        * `--compare_with_abc` : Compare the result with abc (optional)
        * `--re-predict` : Re-predict the data (optional)
        * `--compare_with_ic3ref_basic_generalization`: Compare the result with ic3ref basic generalization (optional)
        * `--compare_with_nnic3_basic_generalization'`: Compare the result with nnic3 basic generalization (optional)
        * `--aig-case-name`: Test the single aiger (optional)
    * **note:**
        * delete the duplicate data in `case4comp/` folder before runing the script (if this script has been run before with the same arguments)


## Analyze the Result
* `result_analyzer` : analyze the result
    * **usage:** `python result_analyzer.py --log-file <log-file>`
    * **parameters:**
        * `--log-file` : The log file to use (e.g. `compare_with_ic3ref.csv`)

## Debug

### Debug the collect.py
* `data2dataset/cex2smt2/collect.py`: Convert aig+inv to graph
    * **usage:** `python collect.py --aag <aag> --generalize <generalize> --cnf <cnf> --generate_smt2 <generate_smt2> --inv-correctness-check <inv-correctness-check> --run-mode <run-mode> --model-checker <model-checker> --thorough-simplification <thorough-simplification> --deep-simplification <deep-simplification> --moderate-simplification <moderate-simplification> --slight-simplification <slight-simplification> --naive-simplification <naive-simplification> --ground-truth-folder-prefix <ground-truth-folder-prefix> --dump-folder-prefix <dump-folder-prefix>`
    * **parameters:**
        * `--aag` : The aag file to use (e.g. `tip2_2.aag`)
        * `--generalize` : Generalize the predesessor (e.g. `True`)
        * `--cnf` : The cnf file to use (e.g. `tip2_2.cnf`)
        * `--generate_smt2` : Generate smt2 file (e.g. `True`)
        * `--inv-correctness-check` : Check the correctness of the invariant (e.g. `True`)
        * `--run-mode` : Normal or debug. Debug model will exit after inv correctness check (e.g. `debug`)
        * `--model-checker` : The model checker to use (e.g. `ic3ref`)
        * `--thorough-simplification` : Use sympy in tr simplification + aig operator simplification during tr construction + z3 simplification + counterexample cube simplification (e.g. `False`)
        * `--deep-simplification` : Use sympy in tr simplification + aig operator simplification during tr construction + z3 simplification (e.g. `False`)
        * `--moderate-simplification` : Aig operator simplification during tr construction + z3 simplification (e.g. `False`)
        * `--slight-simplification` : Z3 simplification + ternary simulation (e.g. `False`)
        * `--naive-simplification` : Only use sympy to simplify the counterexample cube (e.g. `False`)
        * `--ground-truth-folder-prefix` : The prefix of the ground truth folder (e.g. `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`)
        * `--dump-folder-prefix` : The prefix of the dump folder (e.g. `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset`)
    * **note:**
        * `_check_tr_correctness_after_simplification()` should be called if the simplification is turned on or debug 

## Others

### Visualize the Result/Clean Log
* `tool_box.py`: Visualize the result/clean log
    * **usage:** `python tool_box.py --clean_trivial_log <clean_trivial_log> --calculate_pickle_number <calculate_pickle_number> --json2mermaid <json2mermaid> --file_path <file_path>`
    * **parameters:**
        * `--clean_trivial_log` : Clean trivial log (e.g. `True`)
        * `--calculate_pickle_number` : Calculate the number of pickle files (e.g. `True`)
        * `--json2mermaid` : Convert json to mermaid (e.g. `True`)
        * `--file_path` : The file path to use (e.g. `dataset_hwmcc2020_small_abc_slight_1`)

### Symbolic Regression
* `data2dataset/cex2smt2/symbolic_reg_model.py`: Symbolic regression model
    * **usage:** `python symbolic_reg_model.py --model-file <model-file> --validate <validate> --model <model>`
    * **parameters:**
        * `--model-file` : The path to the model file (e.g. `symbolic_reg_model_2021-01-06_07:56:51_last.pth.tar`)
        * `--validate` : Determin whether to validate the model (e.g. `True`)
        * `--model` : Determin which model to use (e.g. `1`)

### AutoRegressor
* `data2dataset/cex2smt2/pytorch_auto_regression.py`: Auto-pytorch regression model
* `data2dataset/cex2smt2/sklearn_auto_regression.py`: Auto-sklearn regression model

### SAT Model Uniform Distribution Sampling
* `data2dataset/cex2smt2/ModelSampler.py`: SAT model uniform distribution sampling
    * **usage:** `python ModelSampler.py --smt2-file <smt2-file>`
    * **parameters:**
        * `--smt2-file` : The path to the smt2 file (e.g. `tip2_2.smt2`)
* `data2dataset/cex2smt2/ModelSample_real_ud` : Using crytominisat to perform real uniform distribution sampling
    * **note:** This script is under development. It is not used in the paper.

### Find complex case (not used in the paper)
* `utils/check_proc.sh`: check the process (normally used to find complex case if `build_data.py` has been run for a long time)
    * **usage:** `source ./check_proc.sh && echo "$count"` or `source ./check_proc.sh && echo "$aag_files_and_count"`
