FROM alpine:latest

RUN apk add --no-cache iperf3 net-tools

COPY config/ /
COPY run.sh /run.sh

CMD ["/bin/sh", "/run.sh"]
