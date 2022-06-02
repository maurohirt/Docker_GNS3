#!/usr/bin/python3

##########################################################################
# Copyright (C) 2021 HARMONIA Project
#
# HARMONIA's Tiny SRv6 Controller - Get link state information from FRR's
#  OSPFv3
#
# @author Martin Buck <martin.buck@ruag.ch>
# - Modified by Rafael Hengen Ribeiro <ribeiro@ifi.uzh.ch>
#  -> Multi-instance parameters were removed
#  -> Code formatting (PEP8)
#  -> new method get_network()
#
##########################################################################

import asyncio
import json


class FrrOspfv3:
    def __init__(self, mock=None, dump=None):
        self._mock = mock
        self._dump = dump
        self._prev_interfaces_set = set()
        self._prev_neighbors_set = set()
        self._prev_routes_set = set()
        self._prev_lsdb_set = set()

    async def get_status(self):
        status = await self._vtysh(f"show ipv6 ospf6 json",
                                   do_json=True)
        return status

    async def get_interfaces(self):
        interfaces = await self._vtysh(f"show ipv6 ospf6 interface json",
                                       do_json=True)
        return interfaces

    async def get_neighbors(self):
        neighbors = await self._vtysh(f"show ipv6 ospf6 neighbor json",
                                      do_json=True)
        return neighbors["neighbors"]

    async def get_network(self):
        status = await self._vtysh(f"show ipv6 ospf6 database network json",
                                   do_json=True)
        return status

    async def get_routes(self):
        routes = await self._vtysh(f"show ipv6 ospf6 route detail json",
                                   do_json=True)
        return routes["routes"]

    async def get_lsdb(self):
        lsdb = await self._vtysh(f"show ipv6 ospf6 database detail json",
                                 do_json=True)
        return lsdb

    async def get_lsdb_flat(self):
        lsdb = await self.get_lsdb()
        lsdb_flat = self._flatten(
            lsdb["areaScopedLinkStateDb"],
            dict(scope="area"), ("lsa", "lsaDescription")) \
            + self._flatten(lsdb["asScopedLinkStateDb"],
                            dict(scope="as"), ("lsa",)) \
            + self._flatten(lsdb["interfaceScopedLinkStateDb"],
                            dict(scope="interface"), ("lsa",))
        return lsdb_flat

    async def diff_neighbors(self):
        neighbors = [self._hashable(d) for d in await self.get_neighbors()]
        neighbors_set = set(neighbors)
        added = list(neighbors_set - self._prev_neighbors_set)
        deleted = list(self._prev_neighbors_set - neighbors_set)
        self._prev_neighbors_set = neighbors_set
        return added, deleted, neighbors

    async def diff_interfaces(self):
        interfaces = self._hashable(await self.get_interfaces())
        interfaces_set = set(interfaces)
        added = list(interfaces_set - self._prev_interfaces_set)
        deleted = list(self._prev_interfaces_set - interfaces_set)
        self._prev_interfaces_set = interfaces_set
        return added, deleted, interfaces

    async def diff_routes(self):
        routes = [self._hashable(d) for d in await self.get_routes()]
        routes_set = set(routes)
        added = list(routes_set - self._prev_routes_set)
        deleted = list(self._prev_routes_set - routes_set)
        self._prev_routes_set = routes_set
        return added, deleted, routes

    async def diff_lsdb(self):
        lsdb = [self._hashable(d) for d in await self.get_lsdb_flat()]
        lsdb_set = set(lsdb)
        added = list(lsdb_set - self._prev_lsdb_set)
        deleted = list(self._prev_lsdb_set - lsdb_set)
        self._prev_lsdb_set = lsdb_set
        return added, deleted, lsdb

    @staticmethod
    def _flatten(items, base, iter_keys):
        """Return list of flat dicts created from list of nested dict lists

        Use list of keys in iter_keys to descend down into inner lists and
        merge the innermost dicts up to create the flat list. When merging up,
        first append the current base dict without merged inner dicts, then
        append inner dict by inner dict, each merged with base dict.

        Example:
        > i = [dict(a=1, l1=[dict(b=1, l2=[dict(c=1),dict(c=2)]), \
            dict(b=2)]), dict(a=2)]
        > print(json.dumps(i, indent=2))
        [
          {
            "a": 1,
            "l1": [
              {
                "b": 1,
                "l2": [
                  {
                    "c": 1
                  },
                  {
                    "c": 2
                  }
                ]
              },
              {
                "b": 2
              }
            ]
          },
          {
            "a": 2
          }
        ]
        > print(json.dumps(mtsc_pce_mgr.FrrOspfv3._flatten(i, dict(base=123), \
            ["l1", "l2"]), indent=2))
        [
          {
            "base": 123,
            "a": 1
          },
          {
            "base": 123,
            "a": 1,
            "b": 1
          },
          {
            "base": 123,
            "a": 1,
            "b": 1,
            "c": 1
          },
          {
            "base": 123,
            "a": 1,
            "b": 1,
            "c": 2
          },
          {
            "base": 123,
            "a": 1,
            "b": 2
          },
          {
            "base": 123,
            "a": 2
          }
        ]
        """
        ret = []
        for item in items:
            newbase = base.copy()
            newbase.update(item)
            if iter_keys and iter_keys[0] in item:
                del newbase[iter_keys[0]]
                ret.append(newbase)
                ret.extend(FrrOspfv3._flatten(
                    item[iter_keys[0]], newbase, iter_keys[1:]))
            else:
                ret.append(newbase)
        return ret

    @staticmethod
    def _hashable(d):
        """Return hashable equivalent of dict/list d

        Dicts are converted to frozenset containing key/value tuples from dict.
        Lists are converted to tuples. Other types are returned unchanged.
        Conversion is recursive, i.e. dict/list values are converted as well.
        """
        if isinstance(d, dict):
            return frozenset([(k, FrrOspfv3._hashable(v))
                              for k, v in d.items()])
        elif isinstance(d, list):
            return tuple([FrrOspfv3._hashable(v) for v in d])
        else:
            return d

    async def _vtysh(self, cmd, do_json=False):
        if self._mock:
            return self._mock[cmd]
        else:
            p = await asyncio.create_subprocess_exec(
                "vtysh", "-c", cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            (stdout_raw, stderr_raw) = await p.communicate()
            if p.returncode:
                raise RuntimeError(f"vtysh returned error code {p.returncode}")
            stdout = stdout_raw.decode()
            ret = json.loads(stdout) if do_json else stdout
            if self._dump is not None:
                self._dump[cmd] = ret
            return ret
