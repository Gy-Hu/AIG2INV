import aag2graph
import clause
import pickle
from os.path import join, splitext, exists
from os import listdir
import argparse


def convert_aag(aag_fname, inv_cnf):
  aagmodel = aag2graph.AAGmodel()
  succeeded = aagmodel.from_file(aag_fname)
  assert succeeded

  #load clause
  if not clause.check_header(inv_cnf):
    return None
  clauses = clause.Clauses(fname=inv_cnf, num_input=aagmodel.num_input, num_sv=aagmodel.num_sv)
  graph = aagmodel.to_dataframe(6, clauses.clauses, aag_fname) # 4 is the type of node (input, )
  return graph



def test():
  all_graphs = [ \
    convert_aag(aag_fname="testcase1/cnt.aag", inv_cnf="testcase1/inv.cnf"), \
    convert_aag(aag_fname="testcase2/cnt2.aag", inv_cnf="testcase2/inv.cnf"), \
    ]
  print (all_graphs[0].forward_layer_index)
  print (all_graphs[0].edge_index)
  with open("test.pkl", 'wb') as f:
      pickle.dump(all_graphs, f)
      

def test_zero():
  all_graphs = [ \
    convert_aag(aag_fname="testcase3-zeros/cnt.aag", inv_cnf="testcase3-zeros/inv.cnf"), \
    ]
  print (all_graphs[0].forward_layer_index)
  print (all_graphs[0].edge_index)
  with open("test3zeros.pkl", 'wb') as f:
      pickle.dump(all_graphs, f)
      
def test_intel001():
  all_graphs = [ \
    convert_aag(aag_fname="testcase1/cnt.aag", inv_cnf="testcase1/inv.cnf"), \
    convert_aag(aag_fname="../hwmcc07-mod/intel/intel_001.aag", inv_cnf="../hwmcc07-7200-result/output/intel/intel_001/inv.cnf"), \
    convert_aag(aag_fname="testcase2/cnt2.aag", inv_cnf="testcase2/inv.cnf"), \
    ]
  
  print (all_graphs[1].sv_node)
  print (all_graphs[1].sv_node.shape[0])
  print (all_graphs[1].aag_name)
  print (all_graphs[1].clauses)
  with open("test_intel001.pkl", 'wb') as f:
      pickle.dump(all_graphs, f)



if __name__ == '__main__':
    #test_zero()
    #exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('result_path', action='store', type=str, default="../hwmcc07-7200-result/")
    parser.add_argument('aag_path', action='store', type=str, default="../hwmcc07-mod/")
    parser.add_argument('flist', action='store', type=str, default="../flist07.txt")
    parser.add_argument('dataset', action='store', type=str, default='hwmcc07dataset.pkl')

    opts = parser.parse_args()
    allfile=open(opts.flist).readlines()
    all_graphs = []
    num_not_good = 0
    num_unknown_result = 0
    for aagname in allfile:
      aagname=aagname.replace('\n','').replace('\r','')
      inv_cnf=join(opts.result_path,'output',aagname,'inv.cnf')
      aag_fname = join(opts.aag_path, aagname+".aag")
      if not exists(inv_cnf):
        print (aag_fname, inv_cnf, '... unknown result, skipped')
        num_unknown_result += 1
        continue
      if not clause.check_header(inv_cnf):
        print (aag_fname, inv_cnf, '... not unsat >0, skipped')
        num_not_good += 1
        continue  # skip sat/unsat 0
      print (aag_fname, inv_cnf)
      g = convert_aag(aag_fname, inv_cnf)
      all_graphs.append(g)
    print ("Collect dataset %d (unsat) from %d (all) - %d (bad) - %d (unk)" % (len(all_graphs), len(allfile), num_not_good, num_unknown_result))
    pkl_name = opts.dataset
    print('Saving Graph dataset to %s' % pkl_name)
    with open(pkl_name, 'wb') as f:
        pickle.dump(all_graphs, f)
    print('DONE.')

