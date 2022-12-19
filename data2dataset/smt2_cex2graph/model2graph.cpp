
#include <iostream>
#include <string>
#include <time.h>
#include <z3++.h>
#include <fstream>
#include <vector>
#include <map>
#include <vector>
#include <bits/stdc++.h>

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


void walk(int tab, expr e, vector<expr> & bfs_queue)
{
    string blanks(tab, ' ');

    if(e.is_const())
    {
        //cout << blanks << "ARGUMENT"<<"(id:"<<e.id()<<")"<<": "<< e << endl;
        bfs_queue.push_back(e);
    }
    else
    {
        //cout << blanks << "APP: " <<"(id:"<<e.id()<<")"<<": "<< e.decl().name() << endl;
        bfs_queue.push_back(e);
        for(int i = 0; i < e.num_args(); i++)
        {
            walk(tab + 5, e.arg(i),bfs_queue);
        }
    }
}

void visit(expr const & e) {
    if (e.is_app()) {
        unsigned num = e.num_args();
        for (unsigned i = 0; i < num; i++) {
            visit(e.arg(i));
        }
        // do something
        // Example: print the visited expression
        func_decl f = e.decl();
        std::cout << "application of " << f.name() << ": " << e << "\n";
    }
    else if (e.is_quantifier()) {
        visit(e.body());
        // do something
    }
    else { 
        assert(e.is_var());
        // do something
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

int main(int argc, char ** argv) {


    z3::context ctx;
    auto&& opt = z3::optimize(ctx);
    //Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, "dataset/IG2graph/generalize_IG_no_enumerate/nusmv.reactor^4.C_0.smt2", 0, 0, 0, 0, 0, 0);
    //Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, "nusmv.reactor^4.C_0.smt2", 0, 0, 0, 0, 0, 0);
    //Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, "nusmv.syncarb5^2.B_10.smt2", 0, 0, 0, 0, 0, 0);

    //travsersal the smt2 file in "../../dataset/IG2graph/generalize_IG_no_enumerate/" and store file name to a vector
    //const char* filePath = "../../dataset/bad_cube_cex2graph/expr_to_build_graph/nusmv.reactor^4.C";
    const char* filePath = "../../dataset/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B";
    vector<string> filenames;
    GetFileNames(filePath,filenames);

    //traversal the file name vector and get the graph
    for(int i=0;i<filenames.size();i++){
        cout<<"file name:"<<filenames[i]<<endl;
        Z3_ast_vector b = Z3_parse_smtlib2_file(ctx, filenames[i].c_str(), 0, 0, 0, 0, 0, 0);


        // Get all the constriants
        // Z3_ast* args = new Z3_ast[Z3_ast_vector_size(ctx, b)];
        // for (unsigned i = 0; i < Z3_ast_vector_size(ctx, b); ++i) { //execute from 0 to size of b
        //     args[i] = Z3_ast_vector_get(ctx, b, i);
        // }
        // z3::ast result(ctx, Z3_mk_and(ctx, Z3_ast_vector_size(ctx, b), args));

        // Get only the last constriant
        // Z3_ast* args = new Z3_ast[1];
        // unsigned i = Z3_ast_vector_size(ctx, b)-1;
        // cout<<"i: "<<int(i)<<endl;
        // args[0] = Z3_ast_vector_get(ctx, b, i);
        // z3::ast result(ctx, Z3_mk_and(ctx, 1, args));

        // fetch the last constriant - one line method
        Z3_ast result = Z3_ast_vector_get(ctx, b, Z3_ast_vector_size(ctx, b)-1);

        ctx.check_error();
        //walk(0,ctx);
        //z3.toExpr(result);
        expr k(ctx, result); 
        opt.add(k);

        auto&& res = opt.check();
        switch (res) {
            case z3::sat: std::cout << "Sat" << std::endl;break;
            case z3::unsat: std::cout << "Unsat" << std::endl;break;
            case z3::unknown: std::cout << "Unknown" << std::endl;break;
        }
        vector<expr> bfs_queue;
        walk(0,k, bfs_queue);
        //visit(k);
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
            json_node["data"]["application"] = it->first.decl().name().str();
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