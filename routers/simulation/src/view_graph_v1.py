from helper import get_results,visualization,graph_population




num="205"

parent_folder="dbs"
folder="10_nodes_net"
db_name="10_nodes_"+num+"_db"
G=graph_population("./"+parent_folder+"/"+folder,db_name)

db_pos="./"+parent_folder+"/"+folder+"/"+db_name
print(db_pos)
vis=visualization(G)
vis.view_utilization(db_pos+".html")
vis=visualization(G)
vis.view_link_type(db_pos+"_link_type.html")