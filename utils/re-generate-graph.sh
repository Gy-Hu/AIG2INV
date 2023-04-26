#!/bin/bash

#watch -n 1 "ps -aux | grep execute_model2graph.sh"

# Set the base directory and model2graph path
base_dir="/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2007_tip_ic3ref_no_simplification_0-22/bad_cube_cex2graph/expr_to_build_graph/"
model2graph_path="/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/data2dataset/smt2_cex2graph/model2graph"

# Change to the base directory
cd "$base_dir"

# Count the number of files
total_files=$(ls | wc -l)
current_file=0

# Loop through each file in the base directory
for file in *; do
    # Increment the current file counter
    current_file=$((current_file + 1))

    # Calculate the progress percentage
    progress=$((current_file * 100 / total_files))

    # Print the progress bar
    printf "\r[%-50s] %d%%" "$(printf "%${progress}s" | tr ' ' '#')" "$progress"

    # Execute model2graph for the current file
    "$model2graph_path" "$file" "$base_dir" false
done

# Print a newline after the progress bar
echo ""
