from cnfextract import ExtractCnf
from clause import Clauses
from aigmodel import AAGmodel
from aig2graph import AigGraph
import z3

total_num_node_types = 6

def dump2smt2(cex_clause_pair_list_prop,aag_name,m, return_after_finished = False):
    cubeliteral_to_str = lambda cube_literals: ','.join(map
                                (lambda x: str(m.svars[x[0]]).replace('v','') 
                                if str(x[1])=='1' 
                                else str(int(str(m.svars[x[0]]).replace('v',''))+1),cube_literals))
    # open a file for writing
    with open(f"./{aag_name}_inv_CTI.txt", "w") as text_file:
        for cti in cex_clause_pair_list_prop:
            text_file.write(cubeliteral_to_str(cti[0]) + "\n")
    
    if return_after_finished:
        print('program finished, only dump cti to file')
        return 'Finish all the work!'
        

def convert_one_aag(aag_name, cnf_name, g1_list, g2_list):
    m = AAGmodel()
    m.from_file(aag_name)
    inv_cnf = Clauses(fname=cnf_name, num_sv = len(m.svars), num_input = len(m.inputs))
    extractor = ExtractCnf(aagmodel = m, clause = inv_cnf)
    cex_clause_pair_list_prop, cex_clause_pair_list_ind, is_inductive, has_fewer_clauses = extractor.get_clause_cex_pair()
    if dump2smt2(cex_clause_pair_list_prop,aag_name,m, return_after_finished = True)!= None: return
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
    case = "nusmv.reactor^4.C"
    g1_list = []
    g2_list = []
    convert_one_aag(f"../../case4test/hwmcc_simple/{case}/{case}.aag", f"/data/hongcezh/clause-learning/data-collect/hwmcc07-7200-result/output/tip/{case}/inv.cnf", g1_list, g2_list)
    

if __name__ == "__main__":
    test()


