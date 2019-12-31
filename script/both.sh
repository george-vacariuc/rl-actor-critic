#!/bin/bash

#$RL -> the location of the git repo
IMAGE=gv/tf:rl
# scripts home dir
SCRIPT=/tf/rl-actor-critic/script
REDIS_HOST='localhost'
docker stop $(docker ps -a -q)
#docker run -p 6379:6379 -d redis
docker run -v ~/redis.conf:/usr/local/etc/redis/redis.conf -p 6379:6379 -d redis

screen -S explorer -dm docker run -u $(id -u ${USER}):$(id -g ${USER})  --rm -v $RL:/tf/rl-actor-critic -e REDIS_HOST=$REDIS_HOST --net="host" -it --entrypoint $SCRIPT/explore.sh $IMAGE
screen -S learner -m docker run -u $(id -u ${USER}):$(id -g ${USER}) --runtime=nvidia --rm -v $RL:/tf/rl-actor-critic -e REDIS_HOST=$REDIS_HOST --net="host" -it --entrypoint $SCRIPT/learn.sh $IMAGE
#screen -S learner -m docker run -u $(id -u ${USER}):$(id -g ${USER}) --runtime=nvidia --rm -v $RL:/tf/rl-actor-critic -e REDIS_HOST=$REDIS_HOST --net="host" -it --entrypoint bash $IMAGE
#screen -S explorer -m docker run -u $(id -u ${USER}):$(id -g ${USER}) --runtime=nvidia --rm -v $RL:/tf/rl-actor-critic -e REDIS_HOST=$REDIS_HOST --net="host" -it --entrypoint bash $IMAGE

#docker stop $(docker ps -a -q)
#screen -S learner -m docker run -u $(id -u ${USER}):$(id -g ${USER}) --runtime=nvidia --rm -v $RL:/tf/rl-actor-critic -e REDIS_HOST=$REDIS_HOST -it --entrypoint bash $IMAGE
