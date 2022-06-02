import networkx as nx
from helper import visualization,simulation

s=simulation()
s.make_db("dbs","","try_db")
s.db_init()
s.small_world_topo_gen("A",10,2,4,0.1)
view=visualization(s.G)
view.view_link_type()