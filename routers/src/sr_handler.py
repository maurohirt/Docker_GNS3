#!/usr/bin/python3

##########################################################################
# Copyright (C) 2021 HARMONIA Project
#
# SR methods for SRv6 encap/decap operations based on ROSE SRv6 libraries
#
# @author Rafael Hengen Ribeiro <ribeiro@ifi.izh.ch>
#
##########################################################################

# Proto dependencies
import srv6_manager_pb2

# Controller dependencies
from controller.srv6_utils import handle_srv6_path
from controller.srv6_utils import handle_srv6_behavior

from config import GRPC_PORT, BASE_PATH


def add_sr_encap(source, destination, intf, segments):
    return handle_srv6_path(
        operation='add',
        grpc_address=source,
        grpc_port=GRPC_PORT,
        destination=destination,
        segments=segments,
        device=intf,
        metric=99
    )


def add_sr_decap(target, intf):
    return handle_srv6_behavior(
        operation='add',
        grpc_address=target,
        grpc_port=GRPC_PORT,
        segment=target,
        action='End.DT6',
        lookup_table=254,
        device=intf,
        metric=99
    )


def del_sr_encap(source, destination, intf):
    return handle_srv6_path(
        operation='del',
        grpc_address=source,
        grpc_port=GRPC_PORT,
        destination=destination,
        device=intf
    )
