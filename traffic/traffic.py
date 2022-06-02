#!/usr/bin/python3

import docker
import re
import time

hostname = "/bin/hostname"
cmd_lo = '/bin/sh -c "/sbin/ip -6 addr show dev lo"'
cmd_eth0 = '/bin/sh -c "/sbin/ip -6 addr show dev eth0"'
iperf_server = '/bin/sh -c "/usr/bin/iperf3 -s'
iperf_client = '/bin/sh -c "/usr/bin/iperf3 -c'


ids = {}
rev_ids = {}
routers = {}
hosts = {}
ip = {}

flows = [ ["Alpine1", "Alpine5", 1000000 ], ["Alpine5", "Alpine1", 1000000 ] ]

cli = docker.DockerClient()
containers = cli.containers.list()

for cont in containers:
   host_id = cont.short_id
   res = cont.exec_run(hostname, stream=True, demux=True)
   host = next(res.output)[0].decode('ascii').replace('\n', '')

   ids[host] = host_id
   rev_ids[host_id] = host

   res=cont.exec_run(cmd_lo, stream=True, demux=True)
   cmd_lo_txt = next(res.output)[0].decode('ascii')
   lo_desc = cmd_lo_txt.split('\n')

   for t in lo_desc:
      inet6_line = re.search('inet6 (.*)', str(lo_desc))
      inet6_addr = str(inet6_line.group(1)).split(" ")[0];
      if inet6_addr != "::1/128":
          routers[host] = inet6_addr

   res=cont.exec_run(cmd_eth0, stream=True, demux=True)
   cmd_eth0_txt = next(res.output)[0].decode('ascii')
   eth0_desc = cmd_eth0_txt.split('\n')

   for t in eth0_desc:
      inet6_line = re.search('inet6 (.*)', str(eth0_desc))
      inet6_addr = str(inet6_line.group(1)).split(" ")[0];
      if "fd00:0:" in inet6_addr:
          hosts[host] = inet6_addr

print ("Docker Containers: ")
for key, value in ids.items():
  print(key, value)   
print ("Router Devices: ")
for key, value in routers.items():
    print(key, value)
print ("Host Devices: ")
for key, value in hosts.items():
    print(key, value)
print ("Requested Traffic: ")
for flow in flows:
    print(flow)

   
port = 5201
for flow in flows:
   server_ip = ''

   # server 

   for cont in containers:
      host_id = cont.short_id
      
      if ids[flow[0]] ==  host_id:
          print("Found %s" % host_id)
          server_command = "%s %s %d %s" % (iperf_server, '-p ', port, '"')
          print (server_command)
          cont.exec_run(server_command , detach=True, stream=False)
          server_ip = hosts[rev_ids[host_id]]

   time.sleep(1)
   # client 

   for cont in containers:
      host_id = cont.short_id
      
      if ids[flow[1]] ==  host_id:
          print("Found %s" % host_id)
          client_command = "%s %s %s %d %s %d %s %d %s" % (iperf_client, server_ip.split("/")[0], " -p ", port, ' -b ', flow[2], ' -t ', 0, '"')
          print(client_command)
          cont.exec_run(client_command , detach=True, stream=False)
   
   port+=1



