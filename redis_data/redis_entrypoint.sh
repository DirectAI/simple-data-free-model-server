#!/bin/bash
#Define cleanup procedure
cleanup() {
    redis-cli -a "default_password" SAVE
}

#Trap SIGTERM
trap 'cleanup' SIGTERM

#Execute a command in the background
redis-server --requirepass "default_password" &

#Save the PID of the background process
REDIS_PID=$!

#Wait for the redis-server process to exit
wait $REDIS_PID