#!/bin/sh

# Update configurations - The host hostname should follow the pattern Alpine1, Alpine2, ..., Alpinen
sh $(cat /etc/hostname)/run.sh

while :; do
  sh
  [ "$GNS3_VOLUMES" ] || exit $?
  echo "Restarting console..."
done
