##########################################################################
# Copyright (C) 2022 HARMONIA Project
#
# Main function for HARMONIA's Tiny SRv6 Controller
#
##########################################################################

from utils.net_monitor import filter_valid
import asyncio
import json
from threading import Thread, Lock
import networkx as nx
from flask import Flask, request, jsonify
from time import sleep
import logging
from config import \
    CONTROLLER_ADDRESS, COLLECTOR_PORT, INTERVAL, CAPACITY, THRESHOLD
from classes.frr_ospfv3 import FrrOspfv3
from utils.ip import compare_prefixes
from sr_handler import add_sr_encap, add_sr_decap
from topology_extractor import update_topology, map_interfaces
from humanfriendly import format_size, parse_size
from networkx.drawing.nx_pydot import write_dot
from sys import argv

app = Flask(__name__)

G = nx.MultiDiGraph()
worsened_links = set([])
segments_inserted = []
provisioned_flows = []


def define_weigth(utilization, threshold):
    if utilization <= threshold:
        return 1
    return 1 / (CAPACITY-utilization)


def estimate_bw(previous_counter, current_counter):
    # ToDo: INTERVAL should be replaced by the timestamp difference between the counters
    up = (current_counter['bytes_sent'] -
          previous_counter['bytes_sent'])/INTERVAL
    down = (current_counter['bytes_recv'] -
            previous_counter['bytes_recv'])/INTERVAL
    return {
        'up': up,
        'down': down
    }


def get_router_ipv6(router_id):
    base = 'fcff:{id}::1'
    numeric_id = router_id.split('.')[3]
    return base.format(id=numeric_id)


def map_interface_load_information(graph, source, interfaces):
    for edge in graph.edges:
        r1, r2, _ = edge
        for interface in interfaces:
            if 'neighbor' in interfaces[interface] and r1 == source and r2 == interfaces[interface]['neighbor']:
                if 'bytes_sent' in graph.edges[edge]:
                    bw = estimate_bw(graph.edges[edge], interfaces[interface])
                    graph.edges[edge]['bw_up'] = f'{format_size(float(bw["up"]))}/s'
                    graph.edges[edge]['bw_down'] = f'{format_size(float(bw["down"]))}/s'
                graph.edges[edge]['intf_name'] = interface
                graph.edges[edge]['bytes_sent'] = interfaces[interface]['bytes_sent']
                graph.edges[edge]['bytes_recv'] = interfaces[interface]['bytes_recv']


def get_congested_links(graph, threshold):
    paths = []
    for edge in graph.edges:
        if 'bw_down' in graph.edges[edge]:
            edge_data = graph.edges[edge]
            if parse_size(edge_data['bw_down']) > threshold or parse_size(edge_data['bw_up']) > threshold:
                paths.append(edge)
    return paths


def get_router_id(router_ip):
    _id = None
    id_parts = router_ip.split(':')
    if id_parts[0].startswith('fcff'):
        _id = int(id_parts[1])
    if id_parts[0].startswith('fd00'):
        _id = int(id_parts[2])
    return f'10.0.0.{_id}'


def set_congested(new_graph, link, congested_links):
    default_weight = 1
    congested_weight = 10
    src, dst, link_id = link
    new_graph[src][dst][link_id]['weight'] = congested_weight if link in congested_links else default_weight


def delete_segments(source, destination, ingress_intf):
    '''
        Delete existing segments that have the similar attributes as the paramers, 
            avoiding InterfaceExists exception
    '''
    for index in range(len(segments_inserted)):
        if (source, destination, ingress_intf) in segments_inserted[index]['encap']:
            try:
                del_sr_encap(source, destination, ingress_intf)
                del(segments_inserted[index]['encap']
                    [(source, destination, ingress_intf)])
                return (source, destination, ingress_intf)
            except Exception as e:
                logging.warning(
                    f'Error deleting segments {source} {destination} {ingress_intf}: {e}')
    return None


def insert_segments(source, destination, sid_list, ingress_intf, egress_intf):
    '''
        Install SID on ingress and egress routers. 
        Also, it maps all added segments and paths into the global segments_inserted
    '''
    sr_setup = None
    try:
        add_sr_encap(source, destination, ingress_intf, sid_list[1:])
        add_sr_decap(sid_list[-1], egress_intf)
        sr_setup = {
            'encap': {(source, destination, ingress_intf): sid_list[1:]},
            'decap': (sid_list[-1], egress_intf)
        }
        segments_inserted.append(sr_setup)
    except Exception as e:
        logging.warning(
            f"Error inserting segments: {source}->{destination} {sid_list[1:]} : {e}")
    return sr_setup


def flow_is_provisioned(flow):
    return flow in [{'src': f['src'], 'dst': f['dst']} for f in provisioned_flows]


def relocate_flows(graph, flows, congested_links):
    '''
        Make a copy of the network graph and worse congested links.
        Then relocate all flows using SRv6
    '''
    if len(congested_links) == 0 or set(congested_links).issubset(worsened_links):
        return
    new_graph = graph.to_directed()
    for link in new_graph.edges:
        set_congested(new_graph, link, congested_links)
    worsened_links.update(congested_links)

    for flow in flows:
        if flow_is_provisioned(flow):
            continue
        src = get_router_id(flow['src'])
        dst = get_router_id(flow['dst'])
        wsp = nx.shortest_path(new_graph, src, dst, 'weight')
        logging.info(f'New path: {wsp}')
        # derive SRv6 SID list from wsp
        sid_list = [get_router_ipv6(r) for r in wsp]
        ingress = sid_list[0]
        # Random value; should be replaced by the correct one
        ingress_intf = egress_intf = 'eth1'

        del_sr = delete_segments(ingress, flow['dst'], ingress_intf)
        logging.info(f'Segments deleted: {del_sr}')

        sr_list = insert_segments(ingress, flow['dst'],
                                  sid_list, ingress_intf, egress_intf)
        logging.info(f'Segments inserted: {segments_inserted}')
        provisioned_flows.append({
            'src': src,
            'dst': dst,
            'segments': sr_list
        })


def debug_graph(info='bw_up'):
    logging.debug('Current nodes: ')
    for k in G.nodes:
        logging.debug(f'{k} -> {G.nodes[k]}')

    logging.debug('Current edges: ')
    for k in G.edges:
        if info == 'all':
            logging.debug(k)
        elif info in G.edges[k]:
            logging.debug(k)
            logging.debug(
                f'\tup: {G.edges[k]["bw_up"]}, down: {G.edges[k]["bw_down"]}, intf: {G.edges[k]["intf_name"]}')


@app.route("/collector", methods=['POST'])
def flow_get():
    request_data = request.get_json()
    source = request_data.get('source')
    timestamp = request_data.get('timestamp')
    interfaces = request_data.get('interfaces')
    current_flows = request_data.get('flows')

    map_interface_load_information(G, source['id'], interfaces)

    threshold_bw = parse_size('1 MB/s')
    congested = get_congested_links(G, threshold_bw)
    if len(current_flows) >= 1:
        lock = Lock()
        lock.acquire()
        relocate_flows(G, current_flows, congested)
        lock.release()

    k = debug_graph()
    logging.info(f'Congested links: {congested}')

    # ToDo: meaningful server reply
    reply = {'status': 'ok'}
    return jsonify(reply)


def topology_update():
    # Create a graph from LSDB
    topo = FrrOspfv3()
    lsdb = asyncio.run(topo.get_lsdb())
    interfaces = asyncio.run(topo.get_interfaces())
    status = asyncio.run(topo.get_status())
    update_topology(G, lsdb)
    map_interfaces(G, interfaces, status['routerId'])

    debug_graph(info='all')

    write_dot(G, 'topo_debug.dot')


if __name__ == '__main__':
    # Adjust log level
    log_level = logging.INFO
    if len(argv) > 1 and argv[1] == '--debug':
        log_level = logging.DEBUG
    logging.root.setLevel(log_level)

    # Start Collector's server (background)
    Thread(
        target=lambda:
            app.run(host=CONTROLLER_ADDRESS, port=COLLECTOR_PORT, debug=False)
    ).start()

    while True:
        topology_update()
        sleep(10)
