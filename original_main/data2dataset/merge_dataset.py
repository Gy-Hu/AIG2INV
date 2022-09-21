import pickle

fin_list=['newgraph/hwmcc07dataset.pkl','newgraph/hwmcc10dataset.pkl','newgraph/hwmcc17dataset.pkl','newgraph/hwmcc20dataset.pkl']
outputname = 'newgraph/hwmcc07_10_17_20_5000node.pkl'
node_threshold = 5000

output_glist = []
for fname in fin_list:
  with open(fname, 'rb') as fin:
    print ("Processing",fname)
    all_graphs=pickle.load(fin)
    for g in all_graphs:
      num_node = g.x.shape[0]
      if num_node < node_threshold:
        output_glist.append(g)
    del all_graphs

print('Saving Graph dataset to %s' % outputname)
with open(outputname, 'wb') as f:
    pickle.dump(output_glist, f)
print('DONE.')
    

