#!/usr/bin/python

##########################################################################
# Copyright (C) 2021 HARMONIA Project
#
# Configuration parameters
#
# @author Rafael Hengen Ribeiro <ribeiro@ifi.izh.ch>
#
##########################################################################

import os

INTERVAL = 10
CAPACITY = 10_000_000
THRESHOLD = 0.7

CONTROLLER_ADDRESS = 'fcff:5::1'
COLLECTOR_PORT = 8081
COLLECTOR_URL = 'http://[{addr}]:{port}/collector'.format(
    addr=CONTROLLER_ADDRESS, port=COLLECTOR_PORT)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# Port of the gRPC server
GRPC_PORT = 12345
