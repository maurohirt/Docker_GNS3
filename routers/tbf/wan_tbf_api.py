#!/usr/bin/python3

# REST API for HARMONIA WAN emulation for GNS3 Docker-based routers

import re
import subprocess
import flask
import socket

path = "/tbf"

def check_interface(name):
  found = False 
  interfaces = socket.if_nameindex()
  for interface in interfaces:
    if (interface[1] == name):
       found = True

  return found 

def check_auth_token():
  if flask.request.form.get("token") != "cb6b048bd225e3340e0ba889d33ae6fa":
    flask.abort(401)

def bvm_cmd(cmd, catch=True):
  try:
    out = subprocess.check_output(cmd)
  except Exception:
    if not catch:
      raise
    return ""
  return out.decode()

def get_wan_tbf(wanport):
  if (check_interface(wanport) == False):
    flask.abort(400)
  return bvm_cmd(["%s/wan_tbf.sh" % path, str(wanport)]).strip()

def set_wan_tbf(wanport, speed):
  if (check_interface(wanport) == False):
    flask.abort(400)
  if(isinstance(speed, int) == False):
    flask.abort(400)
  bvm_cmd(["%s/wan_tbf.sh" % path, str(wanport), str(speed)])

app = flask.Flask(__name__)

@app.route("/v1/htb/<string:wanport>", methods=["GET"])
def get_tbf(wanport):
  result = get_wan_tbf(wanport)

  return flask.jsonify(result)


@app.route("/v1/htb/<string:wanport>", methods=["POST"])
def set_tbf(wanport):
  data=flask.request.get_json()
    
  set_wan_tbf(wanport, data["capacity"])
  result = get_wan_tbf(wanport)

  return flask.jsonify(result)

if __name__ == "__main__":
  app.run(host="::", port=8080)

