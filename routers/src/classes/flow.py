##########################################################################
# Copyright (C) 2021 HARMONIA Project
#
# Class Flow - Functions for flow entries
#
# @author Rafael Hengen Ribeiro <ribeiro@ifi.izh.ch>
#
##########################################################################
from humanfriendly import format_size

class Flow:
    def __init__(self, interval, flows=[]):
        self.flows = flows
        self.interval = interval

    def update_flows(self, current_flows, timestamp):
        for flow in current_flows:
            flow_entry = self.search_entry(flow)
            if flow_entry:
                self.update_entry(timestamp, flow, flow_entry)
            else:
                flow_entry = Flow.create_entry(timestamp, flow)
                self.flows.append(flow_entry)

    def search_entry(self, flow):
        for item in self.flows:
            if Flow.compare_flows(item, flow):
                return item
        return None

    def compare_flows(flow_a, flow_b):
        keys = ['src', 'dst', 'protoL4', 'sport', 'dport']
        for key in keys:
            if flow_a[key] != flow_b[key]:
                return False
        return True

    def create_entry(timestamp, flow):
        flow_entry = flow.copy()
        flow_entry['last_update'] = timestamp
        flow_entry['bandwidth'] = format_size(0)
        return flow_entry

    def update_entry(self, timestamp, flow, flow_entry):
        flow_entry['last_update'] = timestamp
        # Todo: Replace interval with the timestamp diff
        flow_entry['bandwidth'] = format_size(max(
            ((int(flow.get('bytes', 0)) - int(flow_entry.get('bytes', 0)))
             / self.interval), 0)) + '/s'
        flow_entry['bytes'] = flow.get('bytes', 0)

    def filter(self, func):
        return filter(func, self.flows)
