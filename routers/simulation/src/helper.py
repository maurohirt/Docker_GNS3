import networkx as nx
import sqlite3
import random
from random import randint
import numpy as np
from pyvis.network import Network
import os
from statistics import mean
import math
import pulp
import re
import itertools


class simulation:
    def __init__(self):
        self.conn = []
        self.domains = []
        self.G = nx.MultiDiGraph(directed=True)
        self.flows = []

    def make_db(self, db_parent_folder, db_folder, db_name):
        db_pos = "./"+db_parent_folder+"/"+db_folder+"/"+db_name
        try:
            os.stat(db_parent_folder+"/"+db_folder)
        except:
            os.mkdir(db_parent_folder+"/"+db_folder)

        self.conn = sqlite3.connect(db_pos)
        cur = self.conn.cursor()
        cur.execute("""
                CREATE TABLE IF NOT EXISTS "links" (
            "id"    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            "source"    TEXT,
            "source_int"    TEXT,
            "dest"    TEXT,
            "dest_int"    TEXT,
            "capacity"    INTEGER DEFAULT 1,
            "utilization"    INTEGER DEFAULT 0,
            "weight"    INTEGER DEFAULT 1,
            "delay"    INTEGER DEFAULT 2
        )
                """)
        cur.execute("""
                CREATE TABLE IF NOT EXISTS "flows" (
            "id"    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            "src"    TEXT,
            "dst"    TEXT,
            "rate"    INTEGER,
            "loss"    INTEGER DEFAULT 0,
            "path"    TEXT,
            "path_ids"    TEXT,
            "delay"    INTEGER DEFAULT 0
        )
                """)
        cur.execute("""
                CREATE TABLE IF NOT EXISTS "nodes" (
            "id"    INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
            "name"    TEXT UNIQUE,
            "type"    TEXT
        )
                """)
        self.conn.commit()
    # deletes flows

    def delete_flows_table(self):
        # delete_previous_flows(self)
        cur = self.conn.cursor()
        sqlite_query = """DELETE FROM flows"""
        cur.execute(sqlite_query)
        sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"flows\" """
        cur.execute(sqlite_query)
        self.conn.commit()
    # deletes nodes

    def _delete_nodes_table(self):
        cur = self.conn.cursor()
        sqlite_query = """DELETE FROM nodes"""
        cur.execute(sqlite_query)
        sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"nodes\" """
        cur.execute(sqlite_query)
        self.conn.commit()
    # deletes links

    def _delete_links_table(self):
        cur = self.conn.cursor()
        sqlite_query = """DELETE FROM links"""
        cur.execute(sqlite_query)
        sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"links\" """
        cur.execute(sqlite_query)
        self.conn.commit()
    # re-initialize db

    def db_init(self):
        self.delete_flows_table()
        self._delete_links_table()
        self._delete_nodes_table()
        self.domains = []
    # generates a single flow

    def single_dest_flow_gen(self, src, dst, rate, how_many_times=1):
        # single_dest_flow_gen(self,src,dst,rate,how_many_times=1)
        cur = self.conn.cursor()
        for i in range(0, how_many_times):
            sqlite_query = "INSERT INTO flows (src,dst,rate) VALUES(\"" + \
                src+"\",\""+dst+"\","+rate+")"
            cur.execute(sqlite_query)
        self.conn.commit()
    # generates random flows

    def flow_gen(self, src_domain, dst_domain, n_flows=1, mode="random_dest", rate_set=[], rate_prob_dstr=[], seed=0):
        random.seed(seed)
        if rate_prob_dstr:
            num = 100
            prob_set = []
            i = 1
            for p in rate_prob_dstr:
                prob_set += int(num*p)*str(i)
                i += 1
        for i in range(0, len(self.domains)):
            if self.domains[i][0] == src_domain:
                n_src_nodes = self.domains[i][1]
            if self.domains[i][0] == dst_domain:
                n_dst_nodes = self.domains[i][1]

        for i in range(1, n_flows+1):

            src = src_domain+str(randint(1, n_src_nodes))
            dst = src
            while not(dst != src):
                dst = dst_domain+str(randint(1, n_dst_nodes))
            if mode == "random_dest":
                rate = str(0.1*randint(1, 10))
                self.single_dest_flow_gen(src, dst, rate)
            elif mode == "rate_set":
                rate = str(rate_set[int(random.random()*len(rate_set))])
                self.single_dest_flow_gen(src, dst, rate)
            elif mode == "rate_set_prob":
                index = int(prob_set[int(random.random()*num)])-1
                rate = str(rate_set[index])
                self.single_dest_flow_gen(src, dst, rate)

    def _flows_update(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM flows")
        self.flows = cur.fetchall()

    def _gen_nodes(self, node_name, n_nodes, pce_node):
        cur = self.conn.cursor()
        self.domains.append([node_name, n_nodes])
        # insert new nodes
        for i in range(1, n_nodes+1):
            if i == pce_node:
                sqlite_query = """INSERT INTO nodes (name,type) VALUES(\""""+str(
                    node_name)+str(i)+"""\",\"PCE\")"""
            else:
                sqlite_query = """INSERT INTO nodes (name,type) VALUES(\""""+str(
                    node_name)+str(i)+"""\",\"PCC\")"""
            cur.execute(sqlite_query)
        self.conn.commit()

    def db_population_with_names(self, node_name, n_nodes, pce_node, max_int, max_links_out, seed=0):
        self.domains.append([node_name, n_nodes])
        random.seed(seed)
        cur = self.conn.cursor()
        # insert new nodes
        for i in range(1, n_nodes+1):
            if i == pce_node:
                sqlite_query = """INSERT INTO nodes (name,type) VALUES(\""""+str(
                    node_name)+str(i)+"""\",\"PCE\")"""
            else:
                sqlite_query = """INSERT INTO nodes (name,type) VALUES(\""""+str(
                    node_name)+str(i)+"""\",\"PCC\")"""
            cur.execute(sqlite_query)
        # insert new links
        links_per_router = []
        interfaces = []
        for i in range(0, n_nodes):
            num_links = randint(1, max_links_out)
            links_per_router.append([i+1, num_links, max_int])
            interfaces.append(0)
        # print(links_per_router)
        link_vector = []
        # print(links_per_router[0])
        for link in links_per_router:
            for i in range(0, link[1]):
                source = link[0]
                dest = source
                """while (not(dest != source) and (links_per_router[source-1][2]>0) and (links_per_router[dest-1][2]>0)):
                    dest=randint(1,n_nodes)
                #print(source,dest)"""
                while source == dest:
                    dest = randint(1, n_nodes)
                if links_per_router[dest-1][2] > 0 and (links_per_router[dest-1][2] > 0):
                    links_per_router[source -
                                     1][2] = links_per_router[source-1][2]-1
                    links_per_router[dest-1][2] = links_per_router[dest-1][2]-1
                    source_int = "eth"+str(interfaces[source-1])
                    dest_int = "eth"+str(interfaces[dest-1])
                    if source == dest:
                        print("self-connection to node")
                    link_vector.append([source, source_int, dest, dest_int])
                    link_vector.append([dest, dest_int, source, source_int])
                    interfaces[source-1] = interfaces[source-1]+1
                    interfaces[dest-1] = interfaces[dest-1]+1
        # links_per_router[1][2]=3
        for l in link_vector:
            sqlite_query = """INSERT INTO links (source,source_int,dest,dest_int) VALUES(\""""+str(node_name)+str(
                l[0])+"""\",\""""+str(l[1])+"""\",\""""+str(node_name)+str(l[2])+"""\",\""""+str(l[3])+"""\")"""
            cur.execute(sqlite_query)
        self.conn.commit()
        self._graph_topo_update()

    def small_world_topo_gen(self, node_name, n_nodes, pce_node, k, beta, seed=0):
        cur = self.conn.cursor()
        self._gen_nodes(node_name, n_nodes, pce_node)
        random.seed(seed)
        k_2 = int(k/2)
        nodes_set = np.arange(1, n_nodes+1, 1)
        neighbors = np.arange(-k_2, k_2+1, 1)
        neighbors = np.delete(neighbors, k_2)
        print(nodes_set, neighbors)
        for i in range(1, n_nodes+1):
            source = node_name+str(i)
            for j in neighbors:
                dest = node_name+str(nodes_set[(i+j-1) % n_nodes])
                self.uni_link_to_db(source, dest, cap=1)
        for i in range(1, n_nodes+1):
            if random.random() > beta:
                source = node_name+str(i)
                dest = source
                while dest == source:
                    dest = node_name+str(random.randint(1, n_nodes))
                self.bi_link_to_db(source, dest, cap=1)
        self._graph_topo_update()

    def tree_topo_gen(self, node_name, n_nodes, pce_node=1, distribution=[], int_n=[4, 4, 4], seed=0):
        cur = self.conn.cursor()
        self._gen_nodes(node_name, n_nodes, pce_node)
        random.seed(seed)
        n_nodes_type = []
        for d in distribution:
            n_nodes_type.append(int(n_nodes*d))
        tot_nodes = sum(n_nodes_type)
        if tot_nodes < n_nodes:
            n_nodes_type[-1] += n_nodes-tot_nodes

        cap_set = [[1000, 10000], [150, 1000], [40, 150]]
        cap_dstr = [[0.6, 0.4], [0.6, 0.4], [0.4, 0.6]]

        n_links_out = [2, 3, 4]
        n_links_out_between = [2, 3]

        temp = 0
        for i in range(0, len(n_nodes_type)):
            num = 100
            j = 1
            prob_set = []
            for p in cap_dstr[i]:
                prob_set += int(num*p)*str(j)
                j += 1
            for k in range(temp, temp+n_nodes_type[i]):
                int_src = k+1
                n_links = random.randint(1, n_links_out[i])
                for m in range(0, n_links):
                    int_dst = int_src
                    while int_dst == int_src:
                        int_dst = temp+randint(1, n_nodes_type[i])
                    src = node_name+str(int_src)
                    dst = node_name+str(int_dst)
                    temp_rand = int(random.randint(1, num)-1)
                    cap = cap_set[i][int(prob_set[temp_rand])-1]
                    self.bi_link_to_db(src, dst, cap)
            temp += n_nodes_type[i]
            # print(temp)
        temp1 = 0
        temp2 = 0
        for i in range(0, len(n_nodes_type)-1):
            temp2 += n_nodes_type[i]
            # print("----",temp1+1,temp1+n_nodes_type[i])
            # print("-----",temp2+1,temp2+n_nodes_type[i+1])
            num = 100
            j = 1
            prob_set = []
            for p in cap_dstr[i]:
                prob_set += int(num*p)*str(j)
                j += 1
            for k in range(temp1, temp1+n_nodes_type[i]):
                int_src = k+1
                n_links = random.randint(1, n_links_out_between[i])
                for n in range(0, n_links):
                    int_dst = random.randint(temp2+1, temp2+n_nodes_type[i+1])
                    src = node_name+str(int_src)
                    dst = node_name+str(int_dst)
                    temp_rand = int(random.randint(1, num)-1)
                    cap = cap_set[i][int(prob_set[temp_rand])-1]
                    self.bi_link_to_db(src, dst, cap)
                # print(int_src,int_dst)
            temp1 += n_nodes_type[i]

    def backbone_topo_gen(self, node_name, n_nodes, pce_node=1, distribution=[], seed=0, parallel_prob=0):
        cur = self.conn.cursor()
        self._gen_nodes(node_name, n_nodes, pce_node)
        random.seed(seed)
        n_nodes_type = []
        for d in distribution:
            n_nodes_type.append(int(n_nodes*d))
        tot_nodes = sum(n_nodes_type)
        if tot_nodes < n_nodes:
            n_nodes_type[-1] += n_nodes-tot_nodes

        cap_set = [[1000, 10000], [150, 1000]]
        cap_dstr = [[0.9, 0.1], [0.7, 0.3]]

        n_links_out = [2, 3]
        n_links_out_between = [2]

        temp = 0
        for i in range(0, len(n_nodes_type)):
            num = 100
            j = 1
            prob_set = []
            for p in cap_dstr[i]:
                prob_set += int(num*p)*str(j)
                j += 1
            for k in range(temp, temp+n_nodes_type[i]):
                int_src = k+1
                n_links = random.randint(1, n_links_out[i])
                for m in range(0, n_links):
                    int_dst = int_src
                    while int_dst == int_src:
                        int_dst = temp+randint(1, n_nodes_type[i])
                    src = node_name+str(int_src)
                    dst = node_name+str(int_dst)
                    temp_rand = int(random.randint(1, num)-1)
                    cap = cap_set[i][int(prob_set[temp_rand])-1]
                    self.bi_link_to_db(src, dst, cap)
                    while random.random() < parallel_prob:
                        #print("parallel link")
                        self.bi_link_to_db(src, dst, cap)
            temp += n_nodes_type[i]
            # print(temp)
        temp1 = 0
        temp2 = 0
        for i in range(0, len(n_nodes_type)-1):
            temp2 += n_nodes_type[i]
            # print("----",temp1+1,temp1+n_nodes_type[i])
            # print("-----",temp2+1,temp2+n_nodes_type[i+1])
            num = 100
            j = 1
            prob_set = []
            for p in cap_dstr[i]:
                prob_set += int(num*p)*str(j)
                j += 1
            for k in range(temp1, temp1+n_nodes_type[i]):
                int_src = k+1
                n_links = random.randint(1, n_links_out_between[i])
                for n in range(0, n_links):
                    int_dst = random.randint(temp2+1, temp2+n_nodes_type[i+1])
                    src = node_name+str(int_src)
                    dst = node_name+str(int_dst)
                    temp_rand = int(random.randint(1, num)-1)
                    cap = cap_set[i][int(prob_set[temp_rand])-1]
                    self.bi_link_to_db(src, dst, cap)
                    while random.random() < parallel_prob:
                        #print("parallel link")
                        self.bi_link_to_db(src, dst, cap)
                # print(int_src,int_dst)
            temp1 += n_nodes_type[i]

        """temp=n_nodes_type[0]
        for i in range(0,len(n_nodes_type)-1):
            print(temp)
            num=100
            j=1
            prob_set=[]
            for p in cap_dstr[i]:
                prob_set+=int(num*p)*str(j)
                j+=1
            
            for k in range(temp,temp+n_nodes_type[i]):
                int_src=k+1
                n_links=random.randint(1,n_links_out[i])
                for m in range(0,n_links):
                    print(temp,temp+n_nodes_type[i+1])
                    int_dst=randint(temp+1, temp+n_nodes_type[i+1])
                    src=node_name+str(int_src)
                    dst=node_name+str(int_dst)
                    temp_rand=int(random.randint(1,num)-1)
                    cap=cap_set[i][int(prob_set[temp_rand])-1]
                    print(src,dst)
                    self.bi_link_to_db(src, dst, cap)
                
            temp+=n_nodes_type[i+1]"""
        self._graph_topo_update()

    def access_topo_gen(self, node_name, n_nodes, pce_node=1, distribution=[], seed=0):
        cur = self.conn.cursor()
        self._gen_nodes(node_name, n_nodes, pce_node)
        random.seed(seed)
        n_nodes_type = []
        for d in distribution:
            n_nodes_type.append(int(n_nodes*d))
        tot_nodes = sum(n_nodes_type)
        if tot_nodes < n_nodes:
            n_nodes_type[-1] += n_nodes-tot_nodes

        cap_set = [[150, 1000], [40, 150]]
        cap_dstr = [[0.6, 0.4], [0.6, 0.4]]

        n_links_out = [2, 3]
        n_links_out_between = [3]

        temp = 0
        for i in range(0, len(n_nodes_type)):
            num = 100
            j = 1
            prob_set = []
            for p in cap_dstr[i]:
                prob_set += int(num*p)*str(j)
                j += 1
            for k in range(temp, temp+n_nodes_type[i]):
                int_src = k+1
                n_links = random.randint(1, n_links_out[i])
                for m in range(0, n_links):
                    int_dst = int_src
                    while int_dst == int_src:
                        int_dst = temp+randint(1, n_nodes_type[i])
                    src = node_name+str(int_src)
                    dst = node_name+str(int_dst)
                    temp_rand = int(random.randint(1, num)-1)
                    cap = cap_set[i][int(prob_set[temp_rand])-1]
                    self.bi_link_to_db(src, dst, cap)
            temp += n_nodes_type[i]
            # print(temp)
        temp1 = 0
        temp2 = 0
        for i in range(0, len(n_nodes_type)-1):
            temp2 += n_nodes_type[i]
            # print("----",temp1+1,temp1+n_nodes_type[i])
            # print("-----",temp2+1,temp2+n_nodes_type[i+1])
            num = 100
            j = 1
            prob_set = []
            for p in cap_dstr[i]:
                prob_set += int(num*p)*str(j)
                j += 1
            for k in range(temp1, temp1+n_nodes_type[i]):
                int_src = k+1
                n_links = random.randint(1, n_links_out_between[i])
                for n in range(0, n_links):
                    int_dst = random.randint(temp2+1, temp2+n_nodes_type[i+1])
                    src = node_name+str(int_src)
                    dst = node_name+str(int_dst)
                    temp_rand = int(random.randint(1, num)-1)
                    cap = cap_set[i][int(prob_set[temp_rand])-1]
                    self.bi_link_to_db(src, dst, cap)
                # print(int_src,int_dst)
            temp1 += n_nodes_type[i]

        """temp=n_nodes_type[0]
        for i in range(0,len(n_nodes_type)-1):
            print(temp)
            num=100
            j=1
            prob_set=[]
            for p in cap_dstr[i]:
                prob_set+=int(num*p)*str(j)
                j+=1
            
            for k in range(temp,temp+n_nodes_type[i]):
                int_src=k+1
                n_links=random.randint(1,n_links_out[i])
                for m in range(0,n_links):
                    print(temp,temp+n_nodes_type[i+1])
                    int_dst=randint(temp+1, temp+n_nodes_type[i+1])
                    src=node_name+str(int_src)
                    dst=node_name+str(int_dst)
                    temp_rand=int(random.randint(1,num)-1)
                    cap=cap_set[i][int(prob_set[temp_rand])-1]
                    print(src,dst)
                    self.bi_link_to_db(src, dst, cap)
                
            temp+=n_nodes_type[i+1]"""
        self._graph_topo_update()

    def _graph_topo_update(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM nodes")
        nodes = cur.fetchall()
        cur.execute("SELECT * FROM links")
        links = cur.fetchall()
        self.G = nx.MultiDiGraph(directed=True)
        for node in nodes:
            self.G.add_node(node[1])
        for link in links:
            self.G.add_edge(link[1], link[3], weight=link[7], src_int=link[2], dst_int=link[4],
                            id=link[0], capacity=link[5], utilization=link[6], delay=link[8], group=[0])
        self.update_group()

    def bi_link_to_db(self, src, dst, cap):
        cur = self.conn.cursor()
        weight = 1/cap
        sqlite_query = """INSERT INTO links (source,dest,capacity,weight) VALUES(\""""+str(
            src)+"""\",\""""+str(dst)+"""\",\""""+str(cap)+"""\",\""""+str(weight)+"""\")"""
        cur.execute(sqlite_query)
        temp = src
        src = dst
        dst = temp
        sqlite_query = """INSERT INTO links (source,dest,capacity,weight) VALUES(\""""+str(
            src)+"""\",\""""+str(dst)+"""\",\""""+str(cap)+"""\",\""""+str(weight)+"""\")"""
        cur.execute(sqlite_query)
        self.conn.commit()

    def uni_link_to_db(self, src, dst, cap):
        cur = self.conn.cursor()
        weight = 1/cap
        sqlite_query = """INSERT INTO links (source,dest,capacity,weight) VALUES(\""""+str(
            src)+"""\",\""""+str(dst)+"""\",\""""+str(cap)+"""\",\""""+str(weight)+"""\")"""
        cur.execute(sqlite_query)
        self.conn.commit()

    def connect_domains(self, src_domain, dst_domain, cap=1, n_connections=1):
        for d in self.domains:
            if src_domain == d[0]:
                n_src = d[1]
            if dst_domain == d[0]:
                n_dst = d[1]
        for i in range(0, n_connections):
            src = src_domain+str(random.randint(1, n_src))
            dst = dst_domain+str(random.randint(1, n_dst))
            self.bi_link_to_db(src, dst, cap)
        self._graph_topo_update()

    def traffic_effect_update(self, clean_path, flow):
        cur = self.conn.cursor()
        flow_actual_rate = flow[3]
        loss_on_path = []
        for edge in clean_path:
            """print(G.edges[edge])
            print(edge)
            print("flow_actual_rate->",flow_actual_rate)"""
            u = self.G.edges[edge]["utilization"]
            c = self.G.edges[edge]["capacity"]
            loss_on_edge = 0
            c_avl = c-u
            if flow_actual_rate >= c_avl:
                loss_on_edge = flow_actual_rate-c_avl
                flow_actual_rate = c_avl
                new_u = c
                new_weight = 1000000
            else:
                new_u = u + flow_actual_rate
                c_avl = c-new_u
                if c_avl == 0:
                    new_weight = 1000000
                else:
                    new_weight = 1/c_avl
            ID = self.G.edges[edge]["id"]
            # print(flow_actual_rate,c_avl,loss_on_edge)
            sql_query = "UPDATE links SET utilization = " + \
                str(new_u)+" WHERE id = "+str(ID)
            cur.execute(sql_query)
            sql_query = "UPDATE links SET weight = " + \
                str(new_weight)+" WHERE id = "+str(ID)
            cur.execute(sql_query)
            self.G.edges[edge]["utilization"] = new_u
            self.G.edges[edge]["weight"] = new_weight
            loss_on_path.append(loss_on_edge)
        tot_loss_on_path = sum(loss_on_path)
        # print(loss_on_path,tot_loss_on_path)
        sql_query = "UPDATE flows SET loss = " + \
            str(tot_loss_on_path)+" WHERE id = "+str(flow[0])
        cur.execute(sql_query)
        path_ids = []
        for edge in clean_path:
            path_ids.append(self.G.edges[edge]["id"])
        sql_query = "UPDATE flows SET path = \"" + \
            str(clean_path)+"\", path_ids = \"" + \
            str(path_ids)+"\" WHERE id = "+str(flow[0])
        cur.execute(sql_query)
        self.conn.commit()
        return loss_on_path

    def traffic_effect_update_threshold(self, clean_path, flow, threshold, mult_fact):
        cur = self.conn.cursor()
        flow_actual_rate = flow[3]
        loss_on_path = []
        for edge in clean_path:
            """print(G.edges[edge])
            print(edge)
            print("flow_actual_rate->",flow_actual_rate)"""
            u = self.G.edges[edge]["utilization"]
            c = self.G.edges[edge]["capacity"]
            new_u = u
            loss_on_edge = 0
            c_avl = c-u
            norm_u = u/c
            if flow_actual_rate >= c_avl:
                loss_on_edge = flow_actual_rate-c_avl
                flow_actual_rate = c_avl
                new_u = c
                new_weight = mult_fact[-1]/c
            else:
                new_u = u + flow_actual_rate
                norm_u = new_u/c
                c_avl = c-new_u
                mult = mult_fact[int(norm_u*len(threshold))]
                mult = 1
                for i in range(0, len(mult_fact)):
                    if norm_u > threshold[i]:
                        mult = mult_fact[i]
                new_weight = mult/c
            ID = self.G.edges[edge]["id"]
            # print(flow_actual_rate,c_avl,loss_on_edge)
            sql_query = "UPDATE links SET utilization = " + \
                str(new_u)+" WHERE id = "+str(ID)
            cur.execute(sql_query)
            sql_query = "UPDATE links SET weight = " + \
                str(new_weight)+" WHERE id = "+str(ID)
            cur.execute(sql_query)
            self.G.edges[edge]["utilization"] = new_u
            self.G.edges[edge]["weight"] = new_weight
            loss_on_path.append(loss_on_edge)
        tot_loss_on_path = sum(loss_on_path)
        # print(loss_on_path,tot_loss_on_path)
        sql_query = "UPDATE flows SET loss = " + \
            str(tot_loss_on_path)+" WHERE id = "+str(flow[0])
        cur.execute(sql_query)
        path_ids = []
        for edge in clean_path:
            path_ids.append(self.G.edges[edge]["id"])
        sql_query = "UPDATE flows SET path = \"" + \
            str(clean_path)+"\", path_ids = \"" + \
            str(path_ids)+"\" WHERE id = "+str(flow[0])
        cur.execute(sql_query)
        self.conn.commit()
        return loss_on_path

    def traffic_effect_update_ospf(self, clean_path, flow):
        cur = self.conn.cursor()
        flow_actual_rate = flow[3]
        loss_on_path = []
        for edge in clean_path:
            u = self.G.edges[edge]["utilization"]
            c = self.G.edges[edge]["capacity"]
            loss_on_edge = 0
            c_avl = c-u
            if flow_actual_rate >= c_avl:
                loss_on_edge = flow_actual_rate-c_avl
                flow_actual_rate = c_avl
                new_u = c
            else:
                new_u = u + flow_actual_rate
                c_avl = c-new_u
            # print(flow_actual_rate,c_avl,loss_on_edge)
            sql_query = "UPDATE links SET utilization = " + \
                str(new_u)+" WHERE id = "+str(self.G.edges[edge]["id"])
            cur.execute(sql_query)
            self.G.edges[edge]["utilization"] = new_u
            loss_on_path.append(loss_on_edge)
        tot_loss_on_path = sum(loss_on_path)
        # print(loss_on_path,tot_loss_on_path)
        sql_query = "UPDATE flows SET loss = " + \
            str(tot_loss_on_path)+" WHERE id = "+str(flow[0])
        cur.execute(sql_query)
        sql_query = "UPDATE flows SET path = \"" + \
            str(clean_path)+"\" WHERE id = "+str(flow[0])
        cur.execute(sql_query)
        path_ids = []
        for edge in clean_path:
            path_ids.append(self.G.edges[edge]["id"])
        sql_query = "UPDATE flows SET path = \"" + \
            str(clean_path)+"\", path_ids = \"" + \
            str(path_ids)+"\" WHERE id = "+str(flow[0])
        cur.execute(sql_query)
        self.conn.commit()
        return loss_on_path

    def calculate_path_weighted(self, flow, criterion="weight"):
        sp = nx.shortest_path(self.G, flow[1], flow[2], criterion)
        # print("calculate_path_weighted.sp--->",sp)
        path = nx.path_graph(sp)
        path_edges = []
        for pe in path.edges:
            for e in self.G.edges:
                if [e[0], e[1]] == [pe[0], pe[1]]:
                    path_edges.append(e)
        path_group = []
        i = 0
        while i < (len(path_edges)):
            j = 0
            temp = []
            temp.append(path_edges[i])
            while ((i+j < (len(path_edges)-1) and path_edges[i+j][0] == path_edges[i+j+1][0])):
                j += 1
                temp.append(path_edges[i+j])
            i = i+j+1
            path_group.append(temp)
        clean_path = []
        for pg in path_group:
            edge_id = []
            for edge in pg:
                edge_id.append(self.G.edges[edge][criterion])
            index_edge_id_min = min(
                range(len(edge_id)), key=edge_id.__getitem__)
            clean_path.append(pg[index_edge_id_min])
        return clean_path

    def flows_effect(self, alg, params=[]):
        if alg == 'ospf':
            self.flows_effect_ospf()
        elif alg == 'weighted':
            self.flows_effect_weighted()
        elif alg == 'threshold':
            self.flows_effect_threshold(params[0], params[1])
        elif alg == 'mcnf':
            self.flows_effect_mcnf()
        elif alg == 'search':
            self.flows_effect_search()
        else:
            raise 'Invalid algorithm'

    def flows_effect_weighted(self):
        paths = []
        total_loss = 0
        self._flows_update()

        for flow in self.flows:
            clean_path = self.calculate_path_weighted(flow)
            loss_on_path = self.traffic_effect_update(clean_path, flow)
            paths.append(clean_path)
            total_loss += sum(loss_on_path)

        print(paths, total_loss)

    def flows_effect_ospf(self):
        paths = []
        total_loss = 0

        self._flows_update()

        for flow in self.flows:
            clean_path = self.calculate_path_weighted(flow)
            loss_on_path = self.traffic_effect_update_ospf(clean_path, flow)
            paths.append(clean_path)
            total_loss += sum(loss_on_path)

        print(paths, total_loss)

    def flows_effect_threshold(self, threshold=[0, 0.5, 1], mult_fact=[1, 5, 1000]):
        paths = []
        total_loss = 0
        self._flows_update()

        for flow in self.flows:
            clean_path = self.calculate_path_weighted(flow)
            loss_on_path = self.traffic_effect_update_threshold(
                clean_path, flow, threshold, mult_fact)
            paths.append(clean_path)
            total_loss += sum(loss_on_path)

        print(paths, total_loss)

    def flows_effect_mcnf(self):
        self._flows_update()
        calculate_paths_mcnf(self.G, self.flows, self.conn.cursor())
        # self.conn.commit()
        # loss_on_path=self.traffic_effect_update_ospf(clean_path,flow)

    def flows_effect_search(self):
        self._flows_update()
        calculate_paths_search(self.G, self.flows, self.conn.cursor())
        # self.conn.commit()
        # loss_on_path=self.traffic_effect_update_ospf(clean_path,flow)

    def change_link_capacity(self, mode="random", cap_set=[], prob_cap_set=[]):
        cur = self.conn.cursor()
        if mode == "random":
            i = 1
            cap_set = [1, 3, 10, 100, 1000]
            for edge in self.G.edges:
                ID = self.G.edges[edge]["id"]
                cap = cap_set[random.randint(0, len(cap_set)-1)]
                wei = 1/cap
                sql_query = "UPDATE links SET capacity = " + \
                    str(cap)+", weight = "+str(wei)+" WHERE id = "+str(i)
                cur.execute(sql_query)
                i += 1
        elif mode == "cust_cap_set":
            i = 1
            for edge in self.G.edges:
                ID = self.G.edges[edge]["id"]
                cap = cap_set[random.randint(0, len(cap_set)-1)]
                wei = 1/cap
                sql_query = "UPDATE links SET capacity = " + \
                    str(cap)+", weight = "+str(wei)+" WHERE id = "+str(i)
                cur.execute(sql_query)
                i += 1
        elif mode == "prob_cap_set":
            num = len(prob_cap_set)
            i = 1
            for edge in self.G.edges:
                if(i % 2 == 1):
                    ID = self.G.edges[edge]["id"]
                    sel = random.uniform(0, 1)
                    not_found = True
                    index = int(random.random()*num)
                    if index == num:
                        index = num-1
                    cap = cap_set[index]
                    wei = 1/cap
                    sql_query = "UPDATE links SET capacity = " + \
                        str(cap)+", weight = "+str(wei)+" WHERE id = "+str(i)
                    cur.execute(sql_query)
                    sql_query = "UPDATE links SET capacity = " + \
                        str(cap)+", weight = "+str(wei)+" WHERE id = "+str(i+1)
                    cur.execute(sql_query)
                i += 1
        self.conn.commit()
        self._graph_topo_update()

    def flow_delay_calc(self):
        self._flows_update()
        cur = self.conn.cursor()
        for flow in self.flows:
            temp = flow[5]
            temp = temp[0:-2]
            sp = temp.split("),")
            path = []
            for i in sp:
                edge = i+")"
                #print("edge ",edge)
                path.append(edge)
            edge = []
            for link in path:
                temp = link[0:-1]
                #print("temp ",temp)
                sp = temp.split(",")
                a = sp[0][3:-1]
                b = sp[1][2:-1]
                c = int(sp[2])
                # print("--->",a,b,c)
                edge.append((a, b, c))
            delay_on_links = []
            for e in edge:
                delay_on_links.append(self.G.edges[e]["delay"])
            tot_delay = sum(delay_on_links)
            sql_query = "UPDATE flows SET delay = " + \
                str(tot_delay)+" WHERE id = "+str(flow[0])
            cur.execute(sql_query)
        self.conn.commit()

    def update_group(self):
        for edge in self.G.edges:
            self.G.edges[edge]["group"] = self.G.edges[edge]["capacity"]


class visualization():
    def __init__(self, G):
        self.G = G
        self.net = []
        self.color_map = ("#007900", "00ff00", "ffff00",
                          "feb300", "ff7f00", "ff4600", "ff0000")
        self.flows = []

    def _view_nodes(self):
        net = Network("800px", "800px", notebook=True, directed=True)
        for node in self.G.nodes:
            if (node[0] == "N"):
                col = "lightblue"
            elif (node[0] == "A"):
                col = "lightgreen"
            elif (node[0] == "B"):
                col = "orange"
            elif (node[0] == "C"):
                col = "lightpink"
            else:
                col = "lightblue"
            net.add_node(node, x=0, y=0, shape="circle", color=col)
        return net

    def view_utilization(self, filename="example.html"):
        # net.from_nx(G)
        xx = 0
        yy = 0
        net = self._view_nodes()
        for edge in self.G.edges:
            u = self.G.edges[edge]["utilization"]
            cap = self.G.edges[edge]["capacity"]
            color = self.color_map[int(u/cap*7-0.0001)]
            net.add_edge(edge[0], edge[1], color=color,
                         label=self.G.edges[edge]["id"])
        net.set_edge_smooth("dynamic")
        net.show(filename)
        self.net = net

    def view_link_type(self, filename="example.html"):
        groups = []
        for edge in self.G.edges:
            groups.append(self.G.edges[edge]["capacity"])
        set_groups = sorted(set(groups))
        net = self._view_nodes()
        for edge in self.G.edges:
            group = self.G.edges[edge]["capacity"]
            found = False
            i = 0
            while not(found):
                if group == set_groups[i]:
                    col_index = i
                    found = True
                i += 1
            color = self.color_map[col_index]
            net.add_edge(edge[0], edge[1], color=color,
                         label=self.G.edges[edge]["id"])
        net.set_edge_smooth("dynamic")
        net.show(filename)

    def view_flow(self, filename, db_pos, flow_id):
        self._get_flows(db_pos)
        path_ids = self._get_flow_path_from_id(flow_id)
        for ID in path_ids:
            # print(ID)
            for i in range(0, len(self.net.edges)):
                if self.net.edges[i]["label"] == ID:
                    self.net.edges[i]["color"] = "blue"
        self.net.show(filename)

    def view_circle_graph(self, distribution=[0.2, 0.5, 0.3], filename="circle.html", view_mode="capacity"):
        net = Network("800px", "800px", notebook=True, directed=True)
        n_nodes = len(self.G.nodes)

        n_nodes_type = []
        for d in distribution:
            n_nodes_type.append(int(n_nodes*d))
        tot_nodes = sum(n_nodes_type)
        if tot_nodes < n_nodes:
            n_nodes_type[-1] += n_nodes-tot_nodes
        print(n_nodes_type)
        j = 0
        i = 0
        r_set = [100, 300, 500]
        temp = 0
        for node in self.G.nodes:
            if i < n_nodes_type[0]:
                j = 0
            elif i < n_nodes_type[0]+n_nodes_type[1]:
                j = 1
            else:
                j = 2
            r = r_set[j]
            xx = r*math.cos(2*math.pi*i/n_nodes_type[j])
            yy = r*math.sin(2*math.pi*i/n_nodes_type[j])
            net.add_node(node, x=xx, y=yy, shape="circle", size=50)
            i += 1
        if view_mode == "capacity":
            groups = []
            for edge in self.G.edges:
                groups.append(self.G.edges[edge]["capacity"])
            set_groups = sorted(set(groups))
            for edge in self.G.edges:
                group = self.G.edges[edge]["capacity"]
                found = False
                i = 0
                while not(found):
                    if group == set_groups[i]:
                        col_index = i
                        found = True
                    i += 1
                color = self.color_map[col_index]
                net.add_edge(edge[0], edge[1], color=color,
                             label=self.G.edges[edge]["id"])
        elif view_mode == "utilization":
            for edge in self.G.edges:
                norm_u = self.G.edges[edge]["utilization"] / \
                    self.G.edges[edge]["capacity"]
                if norm_u == 1:
                    color = self.color_map[-1]
                else:
                    color = self.color_map[int(norm_u*len(self.color_map))]
                net.add_edge(edge[0], edge[1], color=color,
                             label=self.G.edges[edge]["id"])
        net.toggle_physics(False)
        net.show_buttons()
        net.show(filename)

    def view_backbone_graph(self, distribution=[0.5, 0.5], filename="circle.html", view_mode="capacity"):
        net = Network("800px", "800px", notebook=True, directed=True)
        n_nodes = len(self.G.nodes)

        n_nodes_type = []
        for d in distribution:
            n_nodes_type.append(int(n_nodes*d))
        tot_nodes = sum(n_nodes_type)
        if tot_nodes < n_nodes:
            n_nodes_type[-1] += n_nodes-tot_nodes
        print(n_nodes_type)
        j = 0
        i = 0
        r_set = [200, 400]
        temp = 0
        for node in self.G.nodes:
            if i < n_nodes_type[0]:
                j = 0
            else:
                j = 1
            r = r_set[j]
            xx = r*math.cos(2*math.pi*i/n_nodes_type[j])
            yy = r*math.sin(2*math.pi*i/n_nodes_type[j])
            net.add_node(node, x=xx, y=yy, shape="circle", size=50)
            i += 1
        if view_mode == "capacity":
            groups = []
            for edge in self.G.edges:
                groups.append(self.G.edges[edge]["capacity"])
            set_groups = sorted(set(groups))
            for edge in self.G.edges:
                group = self.G.edges[edge]["capacity"]
                found = False
                i = 0
                while not(found):
                    if group == set_groups[i]:
                        col_index = i
                        found = True
                    i += 1
                color = self.color_map[col_index]
                net.add_edge(edge[0], edge[1], color=color,
                             label=self.G.edges[edge]["id"])
        elif view_mode == "utilization":
            for edge in self.G.edges:
                norm_u = self.G.edges[edge]["utilization"] / \
                    self.G.edges[edge]["capacity"]
                if norm_u == 1:
                    color = self.color_map[-1]
                else:
                    color = self.color_map[int(norm_u*len(self.color_map))]
                net.add_edge(edge[0], edge[1], color=color,
                             label=self.G.edges[edge]["id"])
        net.toggle_physics(False)
        net.show_buttons()
        net.show(filename)

    def _get_flows(self, db_pos):
        conn = sqlite3.connect(db_pos)
        cur = conn.cursor()

        cur.execute("SELECT * FROM flows")
        self.flows = cur.fetchall()

    def _get_flow_path_from_id(self, flow_id):
        for flow in self.flows:
            if flow[0] == flow_id:

                vector = np.fromstring(flow[6][1:-1], dtype=int, sep=",")
        return vector

    def _graph_population(self, db_pos):
        conn = sqlite3.connect(db_pos)
        cur = conn.cursor()
        cur.execute("SELECT * FROM nodes")
        nodes = cur.fetchall()

        cur.execute("SELECT * FROM links")
        links = cur.fetchall()

        G = nx.MultiDiGraph(directed=True)
        for node in nodes:
            G.add_node(node[1])
        for link in links:
            G.add_edge(link[1], link[3], weight=link[7], src_int=link[2], dst_int=link[4],
                       id=link[0], capacity=link[5], utilization=link[6], delay=link[8], group=[0])
        return G


def random_flows_gen(conn, n_nodes, n_flows=25):
    cur = conn.cursor()
    sqlite_query = """DELETE FROM flows"""
    cur.execute(sqlite_query)
    sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"flows\" """
    cur.execute(sqlite_query)
    for i in range(1, n_flows+1):
        rate = str(0.1*randint(1, 10))
        src = "N"+str(randint(1, n_nodes))
        dst = src
        while not(dst != src):
            dst = "N"+str(randint(1, n_nodes))
        sqlite_query = "INSERT INTO flows (src,dst,rate) VALUES(\"" + \
            src+"\",\""+dst+"\","+rate+")"
        cur.execute(sqlite_query)
    conn.commit()
    return


def delete_previous_flows(conn):
    cur = conn.cursor()
    sqlite_query = """DELETE FROM flows"""
    cur.execute(sqlite_query)
    sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"flows\" """
    cur.execute(sqlite_query)
    conn.commit()


def single_dest_flow_gen(conn, src, dst, rate, how_many):
    delete_previous_flows(conn)
    cur = conn.cursor()
    for i in range(0, how_many):
        sqlite_query = "INSERT INTO flows (src,dst,rate) VALUES(\"" + \
            src+"\",\""+dst+"\","+rate+")"
        cur.execute(sqlite_query)
    conn.commit()
    return


def calculate_path(G, flow, criterion="id"):
    sp = nx.shortest_path(G, flow[1], flow[2])
    path = nx.path_graph(sp)
    path_edges = []
    for pe in path.edges:
        for e in G.edges:
            if [e[0], e[1]] == [pe[0], pe[1]]:
                path_edges.append(e)
    path_group = []
    i = 0
    while i < (len(path_edges)-1):
        j = 0
        temp = []
        temp.append(path_edges[i])
        while ((path_edges[i+j][0] == path_edges[i+j+1][0]) and (i+j < len(path_edges)-1)):
            j += 1
            temp.append(path_edges[i+j])

        i = i+j+1
        path_group.append(temp)
    path_group.append([path_edges[-1]])
    clean_path = []
    for pg in path_group:
        edge_id = []
        for edge in pg:
            edge_id.append(G.edges[edge][criterion])
        index_edge_id_min = min(range(len(edge_id)), key=edge_id.__getitem__)
        clean_path.append(pg[index_edge_id_min])
    return clean_path


def calculate_path_weighted_old(G, flow, criterion="id"):
    sp = nx.shortest_path(G, flow[1], flow[2], "weight")
    print(sp)
    path = nx.path_graph(sp)
    path_edges = []
    for pe in path.edges:
        for e in G.edges:
            if [e[0], e[1]] == [pe[0], pe[1]]:
                path_edges.append(e)
    path_group = []
    i = 0
    print(path_edges)
    while i < (len(path_edges)-1):
        j = 0
        temp = []
        temp.append(path_edges[i])
        print(i, j)
        while ((path_edges[i+j][0] == path_edges[i+j+1][0]) and (i+j < (len(path_edges)-1))):
            j += 1
            temp.append(path_edges[i+j])
            print(i, j, len(path_edges)-1)

        i = i+j+1
        path_group.append(temp)
    path_group.append([path_edges[-1]])
    clean_path = []
    for pg in path_group:
        edge_id = []
        for edge in pg:
            edge_id.append(G.edges[edge][criterion])
        index_edge_id_min = min(range(len(edge_id)), key=edge_id.__getitem__)
        clean_path.append(pg[index_edge_id_min])
    return clean_path


def calculate_path_weighted(G, flow, criterion="id"):
    sp = nx.shortest_path(G, flow[1], flow[2], "weight")
    print("calculate_path_weighted.sp--->", sp)
    path = nx.path_graph(sp)
    path_edges = []
    for pe in path.edges:
        for e in G.edges:
            if [e[0], e[1]] == [pe[0], pe[1]]:
                path_edges.append(e)
    path_group = []
    i = 0
    while i < (len(path_edges)):
        j = 0
        temp = []
        temp.append(path_edges[i])
        while ((i+j < (len(path_edges)-1) and path_edges[i+j][0] == path_edges[i+j+1][0])):
            j += 1
            temp.append(path_edges[i+j])
        i = i+j+1
        path_group.append(temp)
    clean_path = []
    for pg in path_group:
        edge_id = []
        for edge in pg:
            edge_id.append(G.edges[edge][criterion])
        index_edge_id_min = min(range(len(edge_id)), key=edge_id.__getitem__)
        clean_path.append(pg[index_edge_id_min])
    return clean_path

# calculate_path_ospf evaluates the spf based on weightselects randomly the local minimum links


def calculate_path_ospf(G, flow, criterion="weight"):
    sp = nx.shortest_path(G, flow[1], flow[2], criterion)
    print("calculate_path_weighted.sp--->", sp)
    path = nx.path_graph(sp)
    path_edges = []
    for pe in path.edges:
        for e in G.edges:
            if [e[0], e[1]] == [pe[0], pe[1]]:
                path_edges.append(e)
    path_group = []
    i = 0
    while i < (len(path_edges)):
        j = 0
        temp = []
        temp.append(path_edges[i])
        while ((i+j < (len(path_edges)-1) and path_edges[i+j][0] == path_edges[i+j+1][0])):
            j += 1
            temp.append(path_edges[i+j])
        i = i+j+1
        path_group.append(temp)
    clean_path = []
    i = 0
    for pg in path_group:
        edge_id = []
        for edge in pg:
            edge_id.append(G.edges[edge][criterion])
        min_cost = min(edge_id)
        min_pos_v = []
        for i in range(0, len(edge_id)):
            if edge_id[i] == min_cost:
                min_pos_v.append(i)
        rand_min_index = randint(0, len(min_pos_v)-1)
        min_pos = min_pos_v[rand_min_index]
        clean_path.append(pg[min_pos])
    return clean_path


def calculate_paths_mcnf(G, flows, cur):

    # define link utilization r for MCNF
    r = pulp.LpVariable("r", cat='Continuous', lowBound=0, upBound=None)
    paths = []
    no_paths = []
    putil = []
    coef = []
    dot_coef_putil = []

    lp_prob = pulp.LpProblem("Minmax Problem", pulp.LpMinimize)
    lp_prob += pulp.lpSum([r]), "Minimize_the_maximum"

    for i, flow in enumerate(flows):

        # print ("Flow number %d" % i)

        # derive simple paths in O(V+E) time
        simple_paths = list(nx.all_simple_edge_paths(G, flow[1], flow[2]))
        paths.append(simple_paths)

        no_paths.append(len(simple_paths))

        # print("Paths: ")
        # for index, path in enumerate(paths[i]):
        #    print(index, path)

        # path utilization vector
        putil.append(pulp.LpVariable.dicts("putil%d" %
                     i, range(no_paths[i]), 0, None))

        # consider all child flows between flow[1],flow[2] == flow[3]
        label = "Sum_of_paths_%d" % i
        condition = pulp.lpSum([putil[i][j]
                               for j in range(no_paths[i])]) == flow[3]
        lp_prob += condition, label

        for index, path in enumerate(paths[i]):
            for pe in path:
                for ge in G.edges:
                    if [ge[0], ge[1], ge[2]] == [pe[0], pe[1], pe[2]]:
                        G.edges[ge]["flow%d:path%d" % (i, index)] = 1

    # print("Path coefficients: ")

    for index, edge in enumerate(G.edges):
        # print("Edge index %d" % index)

        coef = []

        for i, flow in enumerate(flows):
            # print ("Path Flow Index %d" % i)

            coef.append(np.zeros(no_paths[i]))

            for pindex, path in enumerate(paths[i]):
                # print ("Path Index %d" % i)
                coef[i][pindex] = G.edges[edge].get(
                    "flow%d:path%d" % (i, pindex), 0)

        w = G.edges[edge].get("capacity", 0)
        label = "Max_constraint_%d" % index

        dot_coef_putil.append(pulp.lpSum([]))

        for i, flow in enumerate(flows):
            dot_coef_putil[index] += pulp.lpSum([coef[i][j] * putil[i][j]
                                                for j in range(no_paths[i])])

        condition = (pulp.lpSum([r*w - dot_coef_putil[index]]) >= 0)
        lp_prob += condition, label

    lp_prob.writeLP("MinmaxProblem.lp")  # optional
    lp_prob.solve()
    print("Status:", pulp.LpStatus[lp_prob.status])

    for v in lp_prob.variables():
        # print (v.name, "=", v.varValue)
        ret = list(map(int, re.findall('\d*\d', v.name)))
        # print(ret)

        if(ret != []):
            # print(paths[ret[0]][ret[1]])
            #
            # temp = []
            # for e in paths[ret[0]][ret[1]]:
            #    temp.append(G.edges[e]["id"])
            # print(temp)

            if(v.varValue != 0):
                traffic_effect_update_mcnf(
                    G, paths[ret[0]][ret[1]], v.varValue, flows[ret[0]][0], cur)

    print("Total Cost =", pulp.value(lp_prob.objective))


def calculate_paths_search(G, flows, cur):
    paths = []
    minimal_loss = float('inf')
    best_element = []

    for i, flow in enumerate(flows):

        # derive simple paths in O(V+E) time
        simple_paths = list(nx.all_simple_edge_paths(G, flow[1], flow[2]))
        paths.append(simple_paths)

    p = itertools.product(*paths)

    for i, element in enumerate(p):
        for ge in G.edges:
            G.edges[ge]["utilization"] = 0

        iteration_loss = 0

        for j, flow_path in enumerate(element):
            iteration_loss += sum(traffic_effect_update_search(G,
                                  flow_path, flows[j][3], flows[j][0], cur))
            # print(i, j, flow_path, flows[j][0], flows[j][1], flows[j][2], flows[j][3])
            # print (iteration_loss)

        if iteration_loss < minimal_loss:
            minimal_loss = iteration_loss
            best_element = element
        elif iteration_loss == minimal_loss:
            path_costs_element = 0
            path_costs_best_element = 0

            for j, flow_path in enumerate(element):
                path_costs_element += path_cost(G, element[j])
                path_costs_best_element += path_cost(G, best_element[j])

            if path_costs_element < path_costs_best_element:
                best_element = element

    print(best_element, minimal_loss)


def path_cost(G, path):
    return sum([1.0/G.edges[path[i]]["capacity"] for i in range(len(path))])


def traffic_effect_update_old(G, clean_path, flow, cur):
    flow_init_rate = flow[3]
    actual_rate = flow_init_rate
    loss_vect = []
    for link in clean_path:
        loss = 0
        capacity = G.edges[link]["capacity"]
        u = G.edges[link]["utilization"]
        capacity_avl = capacity-capacity*u
        print(capacity_avl)

        new_u = u+actual_rate/capacity
        if actual_rate > capacity_avl:
            loss = abs(capacity_avl-actual_rate)
            new_u = 1
        loss_vect.append(loss)
        if capacity_avl == 0:
            new_weight = 100000000
        else:
            new_weight = 1/(capacity_avl)

        sql_query = "UPDATE links SET utilization = " + \
            str(new_u)+" WHERE id = "+str(G.edges[link]["id"])
        cur.execute(sql_query)
        sql_query = "UPDATE links SET weight = " + \
            str(new_weight)+" WHERE id = "+str(G.edges[link]["id"])
        cur.execute(sql_query)
        # update utilization in GRAPH
        G.edges[link]["utilization"] = new_u
        G.edges[link]["weight"] = new_weight
        G.edges[link]["group"].append(flow[0])
    sql_query = "UPDATE flows SET loss = " + \
        str(loss)+" WHERE id = "+str(flow[0])
    cur.execute(sql_query)
    return loss_vect


def traffic_effect_update(G, clean_path, flow, cur):
    flow_actual_rate = flow[3]
    loss_on_path = []
    for edge in clean_path:
        """print(G.edges[edge])
        print(edge)
        print("flow_actual_rate->",flow_actual_rate)"""
        u = G.edges[edge]["utilization"]
        c = G.edges[edge]["capacity"]
        loss_on_edge = 0
        c_avl = c-u
        if flow_actual_rate >= c_avl:
            loss_on_edge = flow_actual_rate-c_avl
            flow_actual_rate = c_avl
            new_u = c
            new_weight = 1000000
        else:
            new_u = u + flow_actual_rate
            c_avl = c-new_u
            if c_avl == 0:
                new_weight = 1000000
            else:
                new_weight = 1/c_avl
        # print(flow_actual_rate,c_avl,loss_on_edge)
        sql_query = "UPDATE links SET utilization = " + \
            str(new_u)+" WHERE id = "+str(G.edges[edge]["id"])
        cur.execute(sql_query)
        sql_query = "UPDATE links SET weight = " + \
            str(new_weight)+" WHERE id = "+str(G.edges[edge]["id"])
        cur.execute(sql_query)
        G.edges[edge]["utilization"] = new_u
        G.edges[edge]["weight"] = new_weight
        loss_on_path.append(loss_on_edge)
    tot_loss_on_path = sum(loss_on_path)
    # print(loss_on_path,tot_loss_on_path)
    sql_query = "UPDATE flows SET loss = " + \
        str(tot_loss_on_path)+" WHERE id = "+str(flow[0])
    cur.execute(sql_query)
    return loss_on_path


def traffic_effect_update_ospf(G, clean_path, flow, cur):
    flow_actual_rate = flow[3]
    loss_on_path = []
    for edge in clean_path:
        """print(G.edges[edge])
        print(edge)
        print("flow_actual_rate->",flow_actual_rate)"""
        u = G.edges[edge]["utilization"]
        c = G.edges[edge]["capacity"]
        loss_on_edge = 0
        c_avl = c-u
        if flow_actual_rate >= c_avl:
            loss_on_edge = flow_actual_rate-c_avl
            flow_actual_rate = c_avl
            new_u = c
        else:
            new_u = u + flow_actual_rate
            c_avl = c-new_u
        # print(flow_actual_rate,c_avl,loss_on_edge)
        sql_query = "UPDATE links SET utilization = " + \
            str(new_u)+" WHERE id = "+str(G.edges[edge]["id"])
        cur.execute(sql_query)
        G.edges[edge]["utilization"] = new_u
        loss_on_path.append(loss_on_edge)
    tot_loss_on_path = sum(loss_on_path)
    # print(loss_on_path,tot_loss_on_path)
    sql_query = "UPDATE flows SET loss = " + \
        str(tot_loss_on_path)+" WHERE id = "+str(flow[0])
    cur.execute(sql_query)
    return loss_on_path


def traffic_effect_update_mcnf(G, clean_path, flow_rate, flow_id, cur):
    flow_actual_rate = flow_rate
    loss_on_path = []
    for edge in clean_path:
        """print(G.edges[edge])
        print(edge)
        print("flow_actual_rate->",flow_actual_rate)"""
        u = G.edges[edge]["utilization"]
        c = G.edges[edge]["capacity"]
        loss_on_edge = 0
        c_avl = c-u
        if flow_actual_rate >= c_avl:
            loss_on_edge = flow_actual_rate-c_avl
            flow_actual_rate = c_avl
            new_u = c
        else:
            new_u = u + flow_actual_rate
            c_avl = c-new_u
        # print(flow_actual_rate,c_avl,loss_on_edge)
        sql_query = "UPDATE links SET utilization = " + \
            str(new_u)+" WHERE id = "+str(G.edges[edge]["id"])
        print(sql_query)
        cur.execute(sql_query)
        G.edges[edge]["utilization"] = new_u
        loss_on_path.append(loss_on_edge)
    tot_loss_on_path = sum(loss_on_path)
    # print(loss_on_path,tot_loss_on_path)
    sql_query = "UPDATE flows SET loss = loss + " + \
        str(tot_loss_on_path)+" WHERE id = "+str(flow_id)
    print(sql_query)
    cur.execute(sql_query)
    return loss_on_path


def traffic_effect_update_search(G, clean_path, flow_rate, flow_id, cur):
    flow_actual_rate = flow_rate
    loss_on_path = []
    for edge in clean_path:
        """print(G.edges[edge])
        print(edge)
        print("flow_actual_rate->",flow_actual_rate)"""
        u = G.edges[edge]["utilization"]
        c = G.edges[edge]["capacity"]
        loss_on_edge = 0
        c_avl = c-u
        if flow_actual_rate >= c_avl:
            loss_on_edge = flow_actual_rate-c_avl
            flow_actual_rate = c_avl
            new_u = c
        else:
            new_u = u + flow_actual_rate
            c_avl = c-new_u
        # print(flow_actual_rate,c_avl,loss_on_edge)
        sql_query = "UPDATE links SET utilization = " + \
            str(new_u)+" WHERE id = "+str(G.edges[edge]["id"])
        # print(sql_query)
        # cur.execute(sql_query)
        G.edges[edge]["utilization"] = new_u
        loss_on_path.append(loss_on_edge)
    tot_loss_on_path = sum(loss_on_path)
    # print(loss_on_path,tot_loss_on_path)
    # sql_query="UPDATE flows SET loss = loss + "+str(tot_loss_on_path)+" WHERE id = "+str(flow_id)
    # print(sql_query)
    # cur.execute(sql_query)
    return loss_on_path


def visualize_graph(G, filename="example.html"):
    net = Network("800px", "800px", notebook=True, directed=True)
    # net.from_nx(G)
    xx = 0
    yy = 0
    for node in G.nodes:
        if (node[0] == "N"):
            col = "lightblue"
        elif (node[0] == "A"):
            col = "lightgreen"
        elif (node[0] == "B"):
            col = "orange"
        elif (node[0] == "C"):
            col = "lightpink"
        net.add_node(node, x=0, y=0, shape="circle", color=col)
        # net.add_node(node)
        # xx+=1

    color_map = ("#007900", "00ff00", "ffff00",
                 "feb300", "ff7f00", "ff4600", "ff0000")
    for edge in G.edges:
        u = G.edges[edge]["utilization"]
        cap = G.edges[edge]["capacity"]
        color = color_map[int(u/cap*7-0.0001)]
        net.add_edge(edge[0], edge[1], color=color, label=G.edges[edge]["id"])
    net.set_edge_smooth("dynamic")
    net.show(filename)
    return


def get_flows_stats(db_folder, db_name):
    #conn = sqlite3.connect("nodes_db")
    conn = sqlite3.connect(db_folder+"/"+db_name)
    cur = conn.cursor()
    cur.execute("SELECT * FROM links")
    links = cur.fetchall()
    net_use = []
    net_cap = []
    for link in links:
        net_use.append(link[6])
        net_cap.append(link[5])
    cur.execute("SELECT * FROM flows")
    flows = cur.fetchall()
    flows_req = []
    flows_loss = []
    for flow in flows:
        flows_req.append(flow[3])
        flows_loss.append(flow[4])
    return [net_use, net_cap, flows_req, flows_loss]


def visualize_graph_links_color(G, filename="link_color.html"):
    net = Network("800px", "800px", notebook=True, directed=True)
    # net.from_nx(G)
    xx = 0
    yy = 0
    for node in G.nodes:
        net.add_node(node, x=0, y=0, shape="circle")
        # net.add_node(node)
        # xx+=1
    capacity_set = []
    cap = []
    for edge in G.edges:
        cap.append(G.edges[edge]["capacity"])
    capacity_set = set(cap)
    capacity_set = sorted(capacity_set)
    print(capacity_set)
    color_map = ("#007900", "00ff00", "ffff00",
                 "feb300", "ff7f00", "ff4600", "ff0000")
    for edge in G.edges:
        u = G.edges[edge]["utilization"]
        cap = G.edges[edge]["capacity"]
        color = color_map[int(u/cap*7-0.0001)]
        net.add_edge(edge[0], edge[1], color=color, label=G.edges[edge]["id"])
    net.set_edge_smooth("dynamic")
    net.show(filename)
    return


def db_population(n_nodes, pce_node, max_int, max_links_out, n_flows, seed=0, db_folder="dbs", db_name="test_db"):
    random.seed(seed)
    db_pos = "./"+db_folder+"/"+db_name
    conn = sqlite3.connect(db_pos)
    cur = conn.cursor()
    # delete nodes TABLE
    sqlite_query = """DELETE FROM nodes"""
    cur.execute(sqlite_query)
    sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"nodes\" """
    cur.execute(sqlite_query)
    # insert new nodes
    for i in range(1, n_nodes+1):
        if i == pce_node:
            sqlite_query = """INSERT INTO nodes (name,type) VALUES(\"N""" + \
                str(i)+"""\",\"PCE\")"""
        else:
            sqlite_query = """INSERT INTO nodes (name,type) VALUES(\"N""" + \
                str(i)+"""\",\"PCC\")"""
        cur.execute(sqlite_query)
    rows = cur.fetchall()
    # delete links TABLE
    sqlite_query = """DELETE FROM links"""
    cur.execute(sqlite_query)
    sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"links\" """
    cur.execute(sqlite_query)
    # insert new links
    links_per_router = []
    interfaces = []
    for i in range(0, n_nodes):
        num_links = randint(1, max_links_out)
        links_per_router.append([i+1, num_links, max_int])
        interfaces.append(0)
    # print(links_per_router)
    link_vector = []
    # print(links_per_router[0])
    for link in links_per_router:
        for i in range(0, link[1]):
            source = link[0]
            dest = source
            while (not(dest != source) and (links_per_router[source-1][2] > 0) and (links_per_router[dest-1][2] > 0)):
                dest = randint(1, n_nodes)
            # print(source,dest)

            links_per_router[source-1][2] = links_per_router[source-1][2]-1
            links_per_router[dest-1][2] = links_per_router[dest-1][2]-1
            source_int = "eth"+str(interfaces[source-1])
            dest_int = "eth"+str(interfaces[dest-1])
            link_vector.append([source, source_int, dest, dest_int])
            link_vector.append([dest, dest_int, source, source_int])
            interfaces[source-1] = interfaces[source-1]+1
            interfaces[dest-1] = interfaces[dest-1]+1
    # links_per_router[1][2]=3
    for l in link_vector:
        sqlite_query = """INSERT INTO links (source,source_int,dest,dest_int) VALUES(\"N"""+str(
            l[0])+"""\",\""""+str(l[1])+"""\",\"N"""+str(l[2])+"""\",\""""+str(l[3])+"""\")"""
        cur.execute(sqlite_query)
    random_flows_gen(conn, n_nodes, n_flows)
    conn.commit()
    return


def db_population_with_names(node_name, n_nodes, pce_node, max_int, max_links_out, n_flows, seed=0, db_folder="dbs", db_name="test_db"):
    random.seed(seed)
    db_pos = "./"+db_folder+"/"+db_name
    conn = sqlite3.connect(db_pos)
    cur = conn.cursor()
    # delete nodes TABLE
    # insert new nodes
    for i in range(1, n_nodes+1):
        if i == pce_node:
            sqlite_query = """INSERT INTO nodes (name,type) VALUES(\""""+str(
                node_name)+str(i)+"""\",\"PCE\")"""
        else:
            sqlite_query = """INSERT INTO nodes (name,type) VALUES(\""""+str(
                node_name)+str(i)+"""\",\"PCC\")"""
        cur.execute(sqlite_query)
    #rows = cur.fetchall()

    # insert new links
    links_per_router = []
    interfaces = []
    for i in range(0, n_nodes):
        num_links = randint(1, max_links_out)
        links_per_router.append([i+1, num_links, max_int])
        interfaces.append(0)
    # print(links_per_router)
    link_vector = []
    # print(links_per_router[0])
    for link in links_per_router:
        for i in range(0, link[1]):
            source = link[0]
            dest = source
            while (not(dest != source) and (links_per_router[source-1][2] > 0) and (links_per_router[dest-1][2] > 0)):
                dest = randint(1, n_nodes)
            # print(source,dest)

            links_per_router[source-1][2] = links_per_router[source-1][2]-1
            links_per_router[dest-1][2] = links_per_router[dest-1][2]-1
            source_int = "eth"+str(interfaces[source-1])
            dest_int = "eth"+str(interfaces[dest-1])
            link_vector.append([source, source_int, dest, dest_int])
            link_vector.append([dest, dest_int, source, source_int])
            interfaces[source-1] = interfaces[source-1]+1
            interfaces[dest-1] = interfaces[dest-1]+1
    # links_per_router[1][2]=3
    for l in link_vector:
        sqlite_query = """INSERT INTO links (source,source_int,dest,dest_int) VALUES(\""""+str(node_name)+str(
            l[0])+"""\",\""""+str(l[1])+"""\",\""""+str(node_name)+str(l[2])+"""\",\""""+str(l[3])+"""\")"""
        cur.execute(sqlite_query)
    # random_flows_gen(conn,n_nodes,n_flows)
    conn.commit()
    return


def graph_population(db_folder, db_name):
    db_pos = "./"+db_folder+"/"+db_name
    conn = sqlite3.connect(db_pos)
    cur = conn.cursor()
    cur.execute("SELECT * FROM nodes")
    nodes = cur.fetchall()

    cur.execute("SELECT * FROM links")
    links = cur.fetchall()

    cur.execute("SELECT * FROM flows")
    flows = cur.fetchall()

    G = nx.MultiDiGraph(directed=True)
    for node in nodes:
        G.add_node(node[1])
    for link in links:
        G.add_edge(link[1], link[3], weight=link[7], src_int=link[2], dst_int=link[4],
                   id=link[0], capacity=link[5], utilization=link[6], delay=link[8], group=[0])
    return G


def delete_nodes_table(conn):
    cur = conn.cursor()
    sqlite_query = """DELETE FROM nodes"""
    cur.execute(sqlite_query)
    sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"nodes\" """
    cur.execute(sqlite_query)
    conn.commit()


def delete_links_table(conn):
    # delete links TABLE
    cur = conn.cursor()
    sqlite_query = """DELETE FROM links"""
    cur.execute(sqlite_query)
    sqlite_query = """DELETE FROM sqlite_sequence WHERE name=\"links\" """
    cur.execute(sqlite_query)
    conn.commit()


def single_link_to_db(conn, src, dst, cap, weight):
    print(src, dst)
    cur = conn.cursor()
    sqlite_query = """INSERT INTO links (source,dest,capacity,weight) VALUES(\""""+str(
        src)+"""\",\""""+str(dst)+"""\",\""""+str(cap)+"""\",\""""+str(weight)+"""\")"""
    cur.execute(sqlite_query)
    temp = src
    src = dst
    dst = temp
    sqlite_query = """INSERT INTO links (source,dest,capacity,weight) VALUES(\""""+str(
        src)+"""\",\""""+str(dst)+"""\",\""""+str(cap)+"""\",\""""+str(weight)+"""\")"""
    cur.execute(sqlite_query)
    conn.commit()


class flow_struct():
    def __init__(self):
        self.ID = []
        self.src = []
        self.dst = []
        self.rate = []
        self.loss = []
        self.path = []
        self.path_ids = []
        self.delay = []


class link_struct():
    def __init__(self):
        self.ID = []
        self.src = []
        self.dst = []
        self.capacity = []
        self.utilization = []
        self.weight = []
        self.delay = []


class get_results():
    def __init__(self, db_pos):
        self.conn = sqlite3.connect(db_pos)
        self.cur = self.conn.cursor()
        self.flows = self._flows_from_db()
        self.avg_flows_rate, self.tot_flows_rate = self._avg_tot_flows_rate()
        self.avg_flows_loss, self.tot_flows_loss = self._avg_tot_flows_loss()
        self.links = self._links_from_db()
        self.avg_links_cap, self.tot_links_cap = self._avg_tot_links_cap()
        self.avg_links_util, self.tot_links_util = self._avg_tot_links_util()

    def _flows_from_db(self):
        sql_query = "SELECT * FROM flows"
        self.cur.execute(sql_query)
        flows = self.cur.fetchall()
        flows_collection = []
        for f in flows:
            flow = flow_struct()
            flow.ID = f[0]
            flow.src = f[1]
            flow.dst = f[2]
            flow.rate = f[3]
            flow.loss = f[4]
            flow.path = f[5]
            flow.path_ids = f[6]
            flow.delay = f[7]
            flows_collection.append(flow)
        return flows_collection

    def _avg_tot_flows_rate(self):
        rates = []
        for f in self.flows:
            rates.append(f.rate)
        return mean(rates), sum(rates)

    def _avg_tot_flows_loss(self):
        losses = []
        for f in self.flows:
            losses.append(f.loss)
        return mean(losses), sum(losses)

    def _links_from_db(self):
        sql_query = "SELECT * FROM links"
        self.cur.execute(sql_query)
        links = self.cur.fetchall()
        links_collection = []
        for l in links:
            link = link_struct()
            link.ID = l[0]
            link.src = l[1]
            link.dst = l[3]
            link.capacity = l[5]
            link.utilization = l[6]
            link.weight = l[7]
            link.delay = l[8]
            links_collection.append(link)
        return links_collection

    def _avg_tot_links_cap(self):
        cap = []
        for l in self.links:
            cap.append(l.capacity)
        return mean(cap), sum(cap)

    def _avg_tot_links_util(self):
        util = []
        for l in self.links:
            util.append(l.utilization)
        return mean(util), sum(util)
