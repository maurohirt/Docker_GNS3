FROM debian:buster-slim
LABEL maintainer="ribeiro@ifi.uzh.ch"

# Make apt noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# Install all dependencies
RUN apt-get update && \
    apt-get -y install net-tools iproute2 iputils-ping traceroute && \
    apt-get -y install conntrack iptables && \
    apt-get -y install git python3-setuptools python3-pip && \
    apt-get -y install curl iperf3 vim && \
    apt-get -y install autoconf && \
    apt-get -y install ethtool && \
    apt-get -y install procps && \
    apt-get -y install systemd && \
    apt-get -y clean all

# Install FRR
RUN curl -s https://deb.frrouting.org/frr/keys.asc | apt-key add -
RUN FRRVER="frr-8" && \
    echo deb https://deb.frrouting.org/frr buster $FRRVER | tee -a /etc/apt/sources.list.d/frr.list
RUN apt-get -y update && apt-get -y install frr frr-pythontools

# Clone ROSE controller repository and install dependencies
RUN git clone https://github.com/netgroup/rose-srv6-control-plane && \
    cd /rose-srv6-control-plane/control_plane/node-manager && \
    python3 setup.py install && \
    cd /rose-srv6-control-plane/control_plane/controller && \
    python3 setup.py install

RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Install pip dependencies
RUN pip install flask==2.0.1 --no-cache-dir && \
    pip install requests==2.25.1 --no-cache-dir && \
    pip install networkx --no-cache-dir && \
    pip install psutil --no-cache-dir && \
    pip install xmltodict --no-cache-dir && \
    pip install humanfriendly --no-cache-dir && \
    pip install pydot --no-cache-dir

# Install library dependencies for ROSE controller
RUN cd /rose-srv6-control-plane/control_plane/protos && \
    python3 setup.py install && \
    cd /rose-srv6-control-plane/control_plane/nb_protos && \
    python3 setup.py install && \
    cd /rose-srv6-control-plane/db_update && \
    python3 setup.py install

# Copy router configs and scripts
COPY daemons /etc/frr/daemons
COPY run.sh /run.sh
COPY config/ /

# Copy sources
COPY simulation /simulation
#RUN pip install -r simulation/requirements.txt --no-cache-dir
COPY src /src

RUN mkdir /tbf

COPY tbf/wan_tbf_api.py /tbf/
COPY tbf/wan_tbf.sh /tbf/
COPY tbf/tbf /etc/init.d/

CMD ["/bin/bash", "/run.sh"]
