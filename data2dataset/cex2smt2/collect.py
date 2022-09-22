from cnfextract import ExtractCnf
from clause import Clauses
from aigmodel import AAGmodel
from aig2graph import AigGraph
import z3

total_num_node_types = 6

def dump2smt2(cex_clause_pair_list_prop):
    for _,_, prop in cex_clause_pair_list_prop:
        print (prop)
        

def convert_one_aag(aag_name, cnf_name, g1_list, g2_list):
    m = AAGmodel()
    m.from_file(aag_name)
    inv_cnf = Clauses(fname=cnf_name, num_sv = len(m.svars), num_input = len(m.inputs))
    extractor = ExtractCnf(aagmodel = m, clause = inv_cnf)
    cex_clause_pair_list_prop, cex_clause_pair_list_ind, is_inductive, has_fewer_clauses = extractor.get_clause_cex_pair()
    #dump2smt2(cex_clause_pair_list_prop)
    num_old_clause = len(inv_cnf.clauses)
    num_new_clause = len(cex_clause_pair_list_prop) + len(cex_clause_pair_list_ind)
    print ('CL %d -> %d' % (num_old_clause, num_new_clause))
    idx = 0
    for cex, clause, prop in cex_clause_pair_list_prop:
        graph = AigGraph(sv = m.svars, inpv = m.inputs, cex = cex, cnf = clause, prop = prop)
        g = graph.to_dataframe(total_num_node_types, aag_name+':p'+str(idx))
        g1_list.append(g)
        idx += 1
    
    idx = 0
    for cex, clause, prop in cex_clause_pair_list_ind:
        graph = AigGraph(sv = m.svars, inpv = m.inputs, cex = cex, cnf = clause, prop = prop)
        g = graph.to_dataframe(total_num_node_types, aag_name+':i'+str(idx))
        g2_list.append(g)
        idx += 1

def test():
    case = "nusmv.syncarb5^2.B"
    g1_list = []
    g2_list = []
    convert_one_aag(f"../../case4test/hwmcc_simple/{case}/{case}.aag", f"/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/{case}/inv.cnf", g1_list, g2_list)
    

if __name__ == "__main__":
    test()


