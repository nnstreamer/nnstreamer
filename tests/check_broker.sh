#!/usr/bin/env bash
mosquitto_pub -r -t check/broker/running -m "mosquitto_broker_is_running"
check=`mosquitto_sub  -t check/broker/running --remove-retained -W 1 | grep mosquitto_broker_is_running`

if [ "$check" = "mosquitto_broker_is_running" ]; then
    echo 1
fi
