
#include <iostream>
#include <string>
#include <time.h>
#include <z3++.h>
#include <fstream>
#include <vector>
#include <map>
#include <vector>
#include <bits/stdc++.h>
//#include "prop_formula.h"
#include <sstream>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace z3;
using namespace std;
#include "json.hpp"
using json = nlohmann::json;
// #include <filesystem>
// namespace recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

/*
Some idea relate to minimize the graph of (!cex & cex') graph
* - design pattern to match rules like !(!a && !b) => a || b in transition relation in aigmodel.py
* - use z3 to simplify the expression, simplify()
* - unsat core until fixed point
* - ternary simulation multiple times
* - mus to extract the minimal sat model (z3 can set minimal sat model)
* - use aig simplification in Z3 C++/python API (I only found C++ API)
* - use sympy to minimize the graph transition relation (bottleneck is translate z3 to sympy), then use z3 simplify()
* - use aig_simple_parser.cpp to minimize the graph transition relation, and implement simplify() in here
* - use PySMT or boolector (bottleneck is translation of z3 to pySMT/boolector)
* - use  Tactic "aig" and repeat function in z3 (https://stackoverflow.com/questions/33599423/optimize-solver-tactics-for-circuit-sat)
*/
//XXX: Double check before running the script
//void walk(int tab, expr e, vector<expr> & bfs_queue)
void walk(expr e, vector<expr> & bfs_queue, unordered_set<unsigned> & visited)
{
    if (visited.count(e.id()) > 0) {
        return; // already visited this node
    }
    visited.insert(e.id());

    //XXX: Double check before running the script
    //string blanks(tab, ' ');

    //if size of bfs_queue is 1000x, then print its size
    if(bfs_queue.size() % 1000 == 0)
    {
        cout<<"walking...bfs_queue size:"<<bfs_queue.size()<<endl;
    }

    if(e.is_const())
    {
        //XXX: Double check before running the script
        //cout << blanks << "ARGUMENT"<<"(id:"<<e.id()<<")"<<": "<< e << endl;
        /*
        if(bfs_queue.size() > 100000){
        // print the arguement - for debug only
        cout << "Printing children: "<< "ARGUMENT"<<"(id:"<<e.id()<<")"<<": "<< e << endl;
        }
        */
        bfs_queue.push_back(e);
    }
    else
    {
        //XXX: Double check before running the script
        //cout << blanks << "APP: " <<"(id:"<<e.id()<<")"<<": "<< e.decl().name() << endl;
        bfs_queue.push_back(e);
        for(int i = 0; i < e.num_args(); i++)
        {
            //XXX: Double check before running the script
            //walk(tab + 5, e.arg(i),bfs_queue);
            /*
            if(bfs_queue.size() > 1000000){
            // print the arguement - for debug only
            cout << "Printing parents: "<< "ARGUMENT"<<"(id:"<<e.id()<<")"<<": "<< e << endl;
            }
            */
            walk(e.arg(i),bfs_queue,visited);
        }
    }
}

struct cmpByZ3ExprID {
    bool operator()(const z3::expr& a, const z3::expr& b) const {
        return a.id() < b.id();
    }
};

int get_nid(expr const & e, map<expr, int,cmpByZ3ExprID> & nid_map) {
    if (e.is_app()) {
      // determine the node is in nid_map or not
        if(nid_map.count(e) == 0) {
            int nid = nid_map.size();
            nid_map.insert(make_pair(e, nid));
            //cout<<"new node: "<<e<<" "<<nid<<endl;
            return nid;
        }
        else if(nid_map.find(e) != nid_map.end()){
            //cout<<"find the node in nid_map, id:"<<e.id()<<endl;
            return nid_map[e];
        }
    }
    else {
        assert(false);
    }
}

void GetFileNames(string path,vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder doesn't Exist!"<<endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
    }
    }
    closedir(pDir);
}

void TestSimplify()
{
    z3::context c;
    
    expr v18 = c.bool_const("v18");
    expr v16 = c.bool_const("v16");

    

    expr and_expr = !( !v18 && !v16);
    std::cout << "original expr: " << and_expr << std::endl;


    z3::tactic t = tactic(c, "ctx-solver-simplify");
    goal g(c);
    g.add(and_expr);
    apply_result res = t(g);
    expr or_expr = res[0].as_expr();
    std::cout << "simplified expr: " << or_expr << std::endl;

    //expr or_expr = and_expr.simplify();
    //std::cout << "simplified expr: " << or_expr << std::endl;
}

int main(int argc, char ** argv) {
    //XXX: Double check before running the script
    //TestSimplify();
    // get the file name from argv

    // Define arguments
    string file_name = argv[1]; // such as "simple_alu"
    string file_path = argv[2]; // such as "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_naive_0/bad_cube_cex2graph/expr_to_build_graph/"
    // read a argument from command line to decide whether to simplify the graph
    std::stringstream ss(argv[3]); // such as "true"
    
    // pre-defined file_name, file_path, is_simplify
    // string file_name = "simple_alu";
    // string file_path = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc2020_all_only_unsat_abc_naive_0/bad_cube_cex2graph/expr_to_build_graph/";
    // std::stringstream ss("true");

    bool is_simplify = false;
    ss >> std::boolalpha >> is_simplify;
    // if `is_simplify` is true, then print this information
    if(is_simplify){
        cout<<"simplification of graph is on"<<endl;
    }
    cout<<"\nfile name:"<<file_name<<endl;
    //print if simplify the graph
    cout<<"ready to simplify: "<<is_simplify<<endl;

    z3::context ctx;
    //auto&& opt = z3::optimize(ctx);
    
    //Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, "dataset/IG2graph/generalize_IG_no_enumerate/nusmv.reactor^4.C_0.smt2", 0, 0, 0, 0, 0, 0);
    //Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, "nusmv.reactor^4.C_0.smt2", 0, 0, 0, 0, 0, 0);
    //Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, "nusmv.syncarb5^2.B_10.smt2", 0, 0, 0, 0, 0, 0);


    //travsersal the smt2 file in "../../dataset/IG2graph/generalize_IG_no_enumerate/" and store file name to a vector
    //const char* filePath = "../../dataset/bad_cube_cex2graph/expr_to_build_graph/nusmv.reactor^4.C";
    //const char* filePath = "../../dataset/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B";
    
    //XXX: Double check before running the script
    //const char* filePath = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_20230106_014957_toy/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B";
    //string filePath = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/expr_to_build_graph/" + file_name;
    string filePath = file_path + file_name;
    //const char* filePath = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B";
    //const char* filePath = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc07_almost_complete/bad_cube_cex2graph/expr_to_build_graph/eijk.bs4863.S";

    vector<string> filenames;
    GetFileNames(filePath,filenames);

    //traversal the file name vector and get the graph
    for(int i=0;i<filenames.size();i++){
        cout<<"\nfile name:"<<filenames[i]<<endl;
        //if file end with .smt2, then parse it
        if(filenames[i].find(".smt2") == string::npos){
            //skip this loop
            continue;}
            
        Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, filenames[i].c_str(), 0, 0, 0, 0, 0, 0);

        // fetch the last constriant - one line method
        Z3_ast result = Z3_ast_vector_get(ctx, b, Z3_ast_vector_size(ctx, b)-1);

        ctx.check_error();
        
        expr k(ctx, result);
        std::cout << "expr num. args (before simplify): " << k.num_args() << "\n";
        //XXX: Double check before running the script
        //print all the args of k
        // for(int i = 0; i < k.num_args(); i++)
        // {
        //     std::cout << "arg " << i << ": " << k.arg(i) << "\n";
        // }

        //XXX: Double check before running the script
        //HACK: This will ignore a "and" operation of k, but why?
        //cout<<"k: \n"<<k<<endl;
        expr simplified_expr = is_simplify ? k.simplify() : k;
        
        //XXX: Double check before running the script
        //std::cout << "expr num. args (after simplify): " << simplified_expr.num_args() << "\n";
        //std::cout<<"simplified_expr: \n"<<simplified_expr<<endl;
        
        vector<expr> bfs_queue; // record all the node in bfs order
        unordered_set<unsigned> visited; //record all the visited node to avoid re-visit
        //XXX: Double check before running the script
        //walk(0,k, bfs_queue);
        walk(simplified_expr, bfs_queue, visited);
        
        cout<<"bfs_queue size: "<<bfs_queue.size()<<endl;
        map<expr,int,cmpByZ3ExprID> map_expr;
        //map_expr.insert(make_pair(bfs_queue[0],0));
        set<pair<int, int>> set_expr_edge;
        for(int i = 0; i < bfs_queue.size(); i++)
        {
            int node_id = get_nid(bfs_queue[i], map_expr);
            //cout<<bfs_queue[i].decl().name()<<endl;
            for(int j = 0; j < bfs_queue[i].num_args(); j++)
            {
                int children_nid = get_nid(bfs_queue[i].arg(j),map_expr);
                //self.edges.add((node_id, children_nid))
                set_expr_edge.insert(make_pair(node_id, children_nid));
            }
        }
        cout<<"map_expr size: "<<map_expr.size()<<endl;
        cout<<"set_expr_edge size: "<<set_expr_edge.size()<<endl;
        //print all edge in the set set_expr_edge
        set<pair<int, int>>::iterator it;
        // print edge list
        // for(it=set_expr_edge.begin();it!=set_expr_edge.end();it++)
        // {
        //     printf("%d %d\n",it->first,it->second);
        // }

        //export to json
        json json_nodes;
        //iterate through the map map_expr
        for(auto it = map_expr.begin(); it != map_expr.end(); it++)
        {
            //cout<<it->first<<" "<<it->second<<endl;
            //j[it->second] = it->first.decl().name();

            json json_node;
            
            
            //find the id in set_expr_edge
            bool flag = false; // the node has children? -> not state variable
            for(auto it_edge = set_expr_edge.begin(); it_edge != set_expr_edge.end(); it_edge++)
            {
                if(it_edge->first == it->second)
                {//find children, this node is not a state variable
                    json_node["data"]["to"]["children_id"].push_back(it_edge->second);
                    json_node["data"]["type"] = "node";
                    flag = true;
                }
            }
            // if the decl().name().str() is false or true, it is a boolean variable, add "constant_" to the name
            // this is for keep the sequence of the graph, like node -> constant boolean -> input -> latch
            if(it->first.decl().name().str() == "false" || it->first.decl().name().str() == "true")
            {
                json_node["data"]["application"] = "constant_" + it->first.decl().name().str();
            }
            else{
                json_node["data"]["application"] = it->first.decl().name().str();
            }
            json_node["data"]["id"] = it->second;

            if(!flag){ //if this node is a state variable
                json_node["data"]["type"] = "variable";
            }
            

            json_nodes.push_back(json_node);
        }

        //export to json, replace the filename extension
        ofstream o(filenames[i].replace(filenames[i].find(".smt2"),5,".json"));
        // for(auto it = json_nodes.begin(); it != json_nodes.end(); it++)
        // {
        //     file<<*it<<endl;
        // }
        o<<json_nodes<<endl;
    }

    return 0;
}