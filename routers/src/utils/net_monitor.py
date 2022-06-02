#!/usr/bin/python3

##########################################################################
# Copyright (C) 2021 HARMONIA Project
#
# Basic structure for flow monitoring using Conntrack
#
# @author Rafael Hengen Ribeiro <ribeiro@ifi.izh.ch>
#
##########################################################################

import subprocess
import psutil
import xmltodict
from humanfriendly import parse_size

conntrack_map = {
    'src': {
        'main': 'layer3',
        'child': 'src'
    },
    'dst': {
        'main': 'layer3',
        'child': 'dst'
    },
    'protoL4': {
        'main': 'layer4',
        'child': '@protoname'
    },
    'sport': {
        'main': 'layer4',
        'child': 'sport'
    },
    'dport': {
        'main': 'layer4',
        'child': 'dport'
    },
    'packets': {
        'main': 'counters',
        'child': 'packets'
    },
    'bytes': {
        'main': 'counters',
        'child': 'bytes'
    }
}


def get_interfaces():
    return psutil.net_if_addrs().keys() \
        if psutil.net_if_addrs() is not None else None


def get_io_counters():
    intf_counters = {}
    counters = psutil.net_io_counters(pernic=True)
    for intf in counters:
        intf_counters[intf] = counters[intf]._asdict()
    return intf_counters


def filter_valid(flow):
    loopback = 'fcff:'
    border = 'fd00'
    threshold_bw = parse_size('1 MB/s')
    return (
        (flow['src'].startswith(loopback) or
            flow['src'].startswith(border)) and
        (flow['dst'].startswith(loopback) or
            flow['dst'].startswith(border)) and
        parse_size(flow['bandwidth']) > threshold_bw
    )


def get_formatted_flow(conntrack_object):
    connection = {}
    for k, mapping in conntrack_map.items():
        main_object = conntrack_object.get(mapping['main'])
        if main_object:
            connection[k] = main_object.get(mapping['child'])
        else:
            continue
    return connection


def get_conntrack_flows():
    output_format = 'xml'
    conntrack_output = read_conntrack_output(output_format)
    conntrack_flows = read_xml_flows(conntrack_output)
    return conntrack_flows


def read_conntrack_output(output_format):
    options = "-o {output}".format(
        output=output_format) if output_format else ""
    conntrack = subprocess.Popen("conntrack -L {options}".format(
        options=options),
        shell=True,
        stdout=subprocess.PIPE)
    return conntrack.stdout.read()


def read_xml_flows(conntrack_output):
    return xmltodict.parse(conntrack_output)
