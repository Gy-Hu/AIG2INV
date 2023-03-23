# How to use the scripts

## Generate Benchmark (Simple preprocessing)

* `utils/fetch_aiger.py` : Fetch the benchmark from the website (processed data should be in `benchmark_folder/` folder)
    * **usage:** `cd utils && python fetch_aiger.py`

## Build Dataset (Cex -> Graph)

* `build_dataset.py` : Build the dataset from the benchmark
    * **usage:** `python build_dataset.py --model-checker <model-checker> --simplification-level <simplification-level> --benchmark <benchmark> --ground_truth_folder_prefix <ground_truth_folder_prefix> --subset_range <subset_range>`
        * **parameters:**
            * `--model-checker` : The model checker to use (e.g. `abc`, `ic3ref`)
            * `--simplification-level` : The simplification level to use (e.g. `thorough`, `deep`, `moderate`, `slight` , `naive`)
            * `--benchmark` : The benchmark to use in benchmark_folder (e.g. `hwmcc2007_all`, `hwmcc2020_all`, `hwmcc2020_all_only_unsat`, `hwmcc2020_small`)
            * `--ground_truth_folder_prefix` : The prefix of the ground truth folder (e.g. `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/ground_truth/hwmcc20_abc_7200_result`, `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/ground_truth/hwmcc20_abc_7200_result`, `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`, `/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/clause-learning/data-collect/hwmcc07-7200-result/output/tip/`)
            * `--subset_range` : The range of data in benchmark to generate (e.g. `1`, `23`)

## Train

* `train_data.py`: Train the model
    * **usage:** 
        * **parameters:**

## Validate the Model

* `main.py` : Validate the model
    * **usage:** 
        * **parameters:**


## Analyze the Result

* `result_analyzer` : analyze the result
    * **usage:** 
        * **parameters:**

## Others

### Visualize the Result

### Clean Log

### Test Data Collection Script

### Symbolic Regression