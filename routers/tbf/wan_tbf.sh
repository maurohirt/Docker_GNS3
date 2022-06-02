#!/bin/bash

if [ ! -e "/sys/class/net/${1}" ]; then
	echo ${1}: not found
	exit -1
fi

if [ $# == 1 ]; then
	if [[ `/sbin/tc qdisc show dev ${1}` =~ tbf ]]; then
		RET=`/sbin/tc qdisc show dev ${1}|cut -d" " -f 8`
		echo ${1}: $RET
	else
		echo ${1}: 0
	fi
fi

if [ $# == 2 ]; then
	/sbin/tc qdisc del dev ${1} root > /dev/null 2>/dev/null
        /sbin/tc qdisc add dev ${1} root tbf rate ${2}mbit latency 50ms burst 1540
fi

