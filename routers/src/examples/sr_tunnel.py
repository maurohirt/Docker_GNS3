#!/usr/bin/python3

##########################################################################
# Copyright (C) 2021 HARMONIA Project
#
# Adding a Encap SRv6 path from R7 to R8 passing through R6
#
# @author Rafael Hengen Ribeiro <ribeiro@ifi.izh.ch>
#
##########################################################################

from sr_handler import add_sr_encap, add_sr_decap

if __name__ == '__main__':
    # Routers
    r6 = 'fd00:0:06::1'
    r7 = 'fd00:0:07::1'
    r8 = 'fd00:0:08::1'
    h2 = 'fd00:0:08::2/64'

    # Encap route R7 -> R6 -> R8 on R7
    add_sr_encap(r7, h2, 'eth1', [r6, r8])

    # Decap route on R8
    add_sr_decap(r8, 'eth0')
