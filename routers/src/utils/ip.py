#!/usr/bin/python3

##########################################################################
# Copyright (C) 2021 HARMONIA Project
#
# Utility functions to deal with IPv6 addresses
#
# @author Rafael Hengen Ribeiro <ribeiro@ifi.izh.ch>
#
##########################################################################

import ipaddress


def compare_prefixes(ip_addr, prefix):
    try:
        router_addr = ipaddress.IPv6Address(ip_addr)
        router_prefix = ipaddress.IPv6Network(prefix)
    except Exception:
        return False
    return router_addr in router_prefix
