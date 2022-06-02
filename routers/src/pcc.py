#!/usr/bin/python

##########################################################################
# Copyright (C) 2022 HARMONIA Project
#
# Reporting current flows to the controller
#
# @author Rafael Hengen Ribeiro <ribeiro@ifi.izh.ch>
#
##########################################################################

import requests
import asyncio
from time import sleep
from datetime import datetime
from utils import net_monitor
from config import COLLECTOR_URL, INTERVAL
from classes.flow import Flow
from classes.frr_ospfv3 import FrrOspfv3

flow = Flow(INTERVAL)

while True:
    connections = []
    try:
        flows = net_monitor.get_conntrack_flows()
        conn_list = flows['conntrack']['flow']
    except:
        print('Error getting conntrack flows')
        sleep(INTERVAL)
        continue

    for conn_object in conn_list:
        for direction in conn_object['meta']:
            connection = net_monitor.get_formatted_flow(direction)
            if connection:
                connections.append(connection)

    timestamp = str(datetime.now())
    current_flows = connections
    flow.update_flows(current_flows, timestamp)
    significant_flows = list(flow.filter(net_monitor.filter_valid))
    total_bytes = sum([int(x['bytes']) for x in flow.flows])

    topo = FrrOspfv3()
    lsdb_intf = asyncio.run(topo.get_interfaces())
    interfaces = net_monitor.get_io_counters()
    status = asyncio.run(topo.get_status())

    for intf in lsdb_intf:
        interfaces[intf]['intf_id'] = lsdb_intf[intf]["interfaceId"]
        if 'bdr' in lsdb_intf[intf]:
            interfaces[intf]['neighbor'] = lsdb_intf[intf]["bdr"]

    del(interfaces['lo'])

    data = {
        'source': {
            'ipv6': None,
            'id': status['routerId']
        },
        'flows': significant_flows,
        'total': total_bytes,
        'interfaces': interfaces,
        'timestamp': str(datetime.now())
    }

    try:
        print(f'Reporting {len(significant_flows)} flows to the controller')
        requests.post(COLLECTOR_URL, json=data)
    except (ConnectionError, OSError) as e:
        print(f'Error connecting to {COLLECTOR_URL}')

    sleep(INTERVAL)
