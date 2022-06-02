#!/bin/sh

# Enable IPv6 forwarding
sysctl -w net.ipv6.conf.all.forwarding=1

# Enable SRv6
for dev in $(ip -o -6 a | awk '{ print $2 }' | grep -v "lo"); do
  sysctl -w net.ipv6.conf."$dev".seg6_enabled=1
done
sysctl net.ipv6.conf.all.seg6_enabled=1

# Enable Conntrack accounting
echo "1" >/proc/sys/net/netfilter/nf_conntrack_acct
ip6tables -A INPUT -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

# Update configurations - The router hostname should follow the pattern Router1, Router2, ..., Routern
cat $(cat /etc/hostname)/frr.conf >/etc/frr/frr.conf

# Routine to write routers' hostsnames on /etc/hosts
HOSTS_LOCK=hosts.lock
if [ ! -f "$HOSTS_LOCK" ]; then
  touch hosts.lock
  for n in $(seq 6 11); do
    echo "fd00:0:$n::1     router$n" >>/etc/hosts
  done
fi

network=/sys/class/net
for t in ${network}/*; do
  interface=$(basename $t)
  if [[ ${interface} =~ ^eth[0-9]+ ]]; then
    speed=$(cat ${network}/${interface}/speed)
    echo ${interface}: ${speed} Mbit/s
    tc qdisc del dev ${interface} root >/dev/null 2>/dev/null
    tc qdisc add dev ${interface} root tbf rate ${speed}mbit latency 50ms burst 1540
  fi
done

node_manager &

service frr start &

/etc/init.d/tbf start

# Start PCC
/bin/sh -c "python3 /src/pcc.py &"

while :; do
  bash
  # Infinite console loop only when run from GNS3
  [ "$GNS3_VOLUMES" ] || exit $?
  echo "Restarting console..."
done
