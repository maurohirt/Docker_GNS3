#!/usr/bin/python3

##########################################################################
# Copyright (C) 2022 HARMONIA Project
#
# Controller's topology extractor
#
# @author Rafael Hengen Ribeiro <ribeiro@ifi.izh.ch>
#
##########################################################################

import networkx as nx


def add_node(graph, areaId, node, **details):
    if node in graph:
        if areaId not in graph.nodes[node]['areas']:
            graph.nodes[node]['areas'].append(areaId)
    else:
        graph.add_node(node, areas=[areaId], **details)


def add_edge(graph, r1, r2, interface_id):
    _interface_id = get_interface_id(interface_id)
    if r2 in graph[r1]:
        for edge in graph[r1][r2]:
            if 'intf' in graph[r1][r2][edge] and graph[r1][r2][edge]['intf'] == _interface_id:
                return
    graph.add_edge(r1, r2, intf=_interface_id)


def update_topology(graph, lsdb):
    '''
    Update the NetworkX topology according to the LSDB
    '''
    for area in lsdb['areaScopedLinkStateDb']:
        if len(area['lsa']) == 0:
            raise Exception('Topology not available!')
        routers = filter(
            lambda device: device["type"] == "Router", area['lsa'])
        for router in routers:
            add_node(graph, area['areaId'],
                     router['advertisingRouter'], type='Router')

            for neighbor in router['lsaDescription']:
                if neighbor['neighborRouterId'] != router['advertisingRouter']:
                    add_node(graph, area['areaId'],
                             neighbor['neighborRouterId'], type='Router')
                    add_edge(graph, router['advertisingRouter'],
                             neighbor['neighborRouterId'], neighbor['interfaceId'])
                    add_edge(graph, neighbor['neighborRouterId'],
                             router['advertisingRouter'], neighbor['neighborInterfaceId'])


def get_interface_id(interface_id):
    try:
        return int(interface_id.split('.')[3])
    except Exception:
        return interface_id


def map_interfaces(graph, interfaces, router_id):
    '''
    Map interface IDs to names
    '''
    del(interfaces['lo'])
    for intf_name, intf_data in interfaces.items():
        for neighbor in graph[router_id]:
            for link in graph[router_id][neighbor]:
                conn = graph[router_id][neighbor][link]
                if conn['intf'] == intf_data['interfaceId']:
                    graph[router_id][neighbor][link]['if_name'] = intf_name


def attribute_routes(graph, routes):
    '''
    Attribute routes into the current graph
    '''
    loopback = 'fcff:'
    for route in routes:
        if route.startswith(loopback):
            adv_router = routes[route]['lsAdvertisingRouter']
            graph.nodes[adv_router]['prefix'] = route
