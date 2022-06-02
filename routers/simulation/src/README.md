python_sim_v5

main.py --> main file

helper.py --> contains the class sim (to be updated and modified) and other useful routines

stats_read.py --> generates a html files with the visualization of the network and its usage for all the databases present in the folder "dbs"

dbs folder -->  contains the dbs in which the simulated networks are stored

dbs and folders are generated from main.py and simulation class

**Additions with respect to v2**
-new function in simulation class that updates the capacity values for the links in the network: 
	change_link_capacity (3 different modes):
		random: selects a random link capacity value in the set [1,3,10,100]
		cust_cap_set: similar to random but receives a custom capacity set as input
		prob_cap_set: similar to cust_cap_set but probability distribution are arbitrary

link_hist.py--> visualizes an histogram of the utilization of links normalized to their capacity 

**Additions with respect to v3**

-new functions in simulation class: 

	flows_effect_threshold evaluates the impact of the generated flow on the network using the threshold strategy that multiplies the weight provided by ospf (or cost) by a factor which scales with link usage, making the link less attractive as its usage grows. threshold values and multiplication factors can be set.
	
	random_flow_gen is NOW flow_gen. flow_gen keeps the same function of the old function, but allows also to generate random destination flows with custom (to be set) data-rates following a non-uniform distribution
	
	update_group explots the "group" field in the graph G. It sets the group with the same value of the link capacity.
	
-new class visualization, receives as input the network graph:

	view_utilization generates a network graph in an html file which shows links utilization (in percent)
	
	view_link_type reads the "group" field of the links in graph G and show different colors for the different link types
	
	small_world_topo_gen generates a watts-strogatz small world network with parameters k  and beta
	
[BUG FIXES]

	simulation._graph_topo_update was duplicating links when used
	
	simulation.db_population_with_names generation of links has been revised
	
**Additions with respect to v4**,


-new class get_results collects data from database regarding links and flows
	
	
- s.change_link_capacity(mode="prob_cap_set") now generates consistent bidirectional links
	









	
