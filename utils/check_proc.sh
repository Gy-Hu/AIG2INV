# aag_files_and_count=$(ps -ef | grep collect.py | grep 2007 | awk -F ' ' '{for(i=1; i<=NF; i++) if($i ~ /\.aag$/) {split($i, a, "/"); gsub(".aag", "", a[length(a)]); print "'"'"'" a[length(a)] "'"'"',"}}' | tee >(wc -l))
#count=$(echo "$aag_files_and_count" | tail -n 1)
aag_stuck_in_smt2_generate=$(ps -ef | grep collect.py | awk -F ' ' '{for(i=1; i<=NF; i++) if($i ~ /\.aag$/) {split($i, a, "/"); gsub(".aag", "", a[length(a)]); print "'"'"'" a[length(a)] "'"'"',"}}' | tee >(wc -l))
aag_stuck_in_model2graph=$(ps -ef | grep model2graph | awk '{print $9}')
aag_stuck_json2networkx=$(ps -ef | grep json2networkx | grep -oP '(?<=expr_to_build_graph/)[^/]*' | sed 's/ --.*//')
kill_stuck_json2networkx=$(ps -ef | grep json2networkx | grep -oP '(?<=expr_to_build_graph/)[^/]*' | sed 's/ --.*//' | xargs -I % pkill -f %)
kill_stuck_in_smt2_generate_2007=$(ps -ef | grep collect.py | grep 2007 | awk '{print $2}' | xargs kill) # add | grep debug to find some unstopped proc
kill_stuck_in_smt2_generate_2020=$(ps -ef | grep collect.py | grep 2020 | awk '{print $2}' | xargs kill)
