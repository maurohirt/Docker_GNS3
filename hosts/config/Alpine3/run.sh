#!/bin/sh

IF_NAME=eth0
IP_ADDR=fd00:0:10::2/64
GW_ADDR=fd00:0:10::1

ip -6 addr add $IP_ADDR dev $IF_NAME
ip -6 route add default via $GW_ADDR dev $IF_NAME
