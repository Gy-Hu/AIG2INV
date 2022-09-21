import pickle

fname='hwmcc07dataset.pkl'
flist='../hwmcc07-7200-result/summary.txt'

timelist=dict()
with open(flist) as fin:
    for l in fin.readlines():
        line = l.split(',')
        name,time = line[0], line[1]
        timelist[name]=time
        

with open(fname, 'rb') as fin:
    all_graphs=pickle.load(fin)

print ('aag_name, num_node, num_layer, n_clause, avg_size, time')
for g in all_graphs:
    num_node = g.x.shape[0]
    
    num_layer = max(g.forward_layer_index[0]).item() + 1
    
    n_clause = len(g.clauses)
    avg_clause_size = sum([len(c) for c in g.clauses])/n_clause
    aag_name = g.aag_name
    
    path=aag_name.split('/')
    indexname = ('/'.join(path[2:])).replace('.aag','')
    time=timelist[indexname]
    
    print (aag_name,',',num_node,',', num_layer,',',n_clause,',', avg_clause_size,',',time)
    
    

