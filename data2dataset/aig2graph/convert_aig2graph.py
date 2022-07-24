import aigmodel
import aig2graph
import clause
import pickle
from os.path import join, splitext, exists
from os import listdir
import argparse


def convert_aag(aag_fname, inv_cnf):
  aagmodel = aigmodel.AAGmodel()
  succeeded = aagmodel.from_file(aag_fname)
  assert succeeded

  trans_per_sv=[aagmodel.latch2next[v] for v in aagmodel.svars]
  # no need to worry about init, all initialized to 0
  g = aig2graph.AigGraph(sv = aagmodel.svars, inpv = aagmodel.inputs, trans=trans_per_sv, output=aagmodel.output)

  #load clause
  if not clause.check_header(inv_cnf):
    return None
  clauses = clause.Clauses(fname=inv_cnf, num_input=len(aagmodel.inputs), num_sv=len(aagmodel.svars))
  graph = g.to_dataframe(6, clauses.clauses, aag_fname) # 4 is the type of node (input, )
  return graph



def test():
  all_graphs = [ \
    convert_aag(aag_fname="../../deprecated/data2dataset/testcase1/cnt.aag", inv_cnf="../../deprecated/data2dataset/testcase1/inv.cnf"), \
    convert_aag(aag_fname="../../deprecated/data2dataset/testcase2/cnt2.aag", inv_cnf="../../deprecated/data2dataset/testcase2/inv.cnf"), \
    ]
  with open("test.pkl", 'wb') as f:
      pickle.dump(all_graphs, f)


if __name__ == '__main__':
    test()
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
      if not exists(inv_cnf):
        print (aag_fname, inv_cnf, '... unknown result, skipped')
        num_unknown_result += 1
        continue
      if not clause.check_header(inv_cnf):
        print (aag_fname, inv_cnf, '... not unsat >0, skipped')
        num_not_good += 1
        continue  # skip sat/unsat 0
      aag_fname = join(opts.aag_path, aagname+".aag")
      print (aag_fname, inv_cnf)
      g = convert_aag(aag_fname, inv_cnf)
      all_graphs.append(g)
    print ("Collect dataset %d (unsat) from %d (all) - %d (bad) - %d (unk)" % (len(all_graphs), len(allfile), num_not_good, num_unknown_result))
    pkl_name = opts.dataset
    print('Saving Graph dataset to %s' % pkl_name)
    with open(pkl_name, 'wb') as f:
        pickle.dump(all_graphs, f)

