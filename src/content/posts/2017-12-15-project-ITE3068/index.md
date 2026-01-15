---
title: "ITE3068: Software Studio"
description: "ITE3068: Software Studio; Naver open source project"
date: 2017-12-15
permalink: /ITE3068
tags: [curriculum]
heroImage:
heroImageAlt:
published: true
---

Project: Software Studio @ cs.hanynag

## Project Requirements

- [x] Use [Docker](https://www.docker.com/) (10pts)
- [x] Performance Comparison (using [arcus](http://naver.github.io/arcus/)) (20pts)
- [x] Use [nBase-ARC](https://github.com/naver/nbase-arc) (20pts)
- [x] Use multi-node (10pts)
- [x] Use [Hubblemon](https://github.com/naver/hubblemon) (10pts)
- [x] Use [nGrinder](http://naver.github.io/ngrinder/) (10pts)
- [x] [Naver Open Source](https://github.com/naver) Contribution
    - Oppend Issues
        1. [arcus-python-client: `Pypi's arcus package seems to have expired`](https://github.com/naver/arcus-python-client/issues/11)
        2. [arcus-python-client: `Not working in macOS`](https://github.com/naver/arcus-python-client/issues/12)
        3. [arcus: `Support docker`](https://github.com/naver/arcus/issues/35)
    - Pull Requests
        1. [hubblemon: `update dependency`](https://github.com/naver/hubblemon/pull/22)
        2. [arcus-python-client: `Update Poller for darwin(macOS)`](https://github.com/naver/arcus-python-client/pull/13)
        3. [arcus: `Update dockerfile`](https://github.com/naver/arcus/pull/36)

## Project

Using [Docker](https://www.docker.com/) with [arcus](https://hub.docker.com/r/ruo91/arcus/), [MySQL](https://hub.docker.com/_/mysql/), [nBase-ARC](https://hub.docker.com/r/hyeongseok05/nbase-arc/).

API app is simple flask app for performance comparison.

All settings are stored in `settings.json`. you can change enviroments for you own service.


### Docker

Using dockerized images and dockerfiles.

Check `app.py` and `settings.json` for docker run.

![Docker ps -a](https://github.com/jiunbae/ITE3068/blob/master/results/docker%20process.png?raw=true)

### API App (for performance comparison)
Powered by flask, support simple RestfulAPI for performance comparison.

API Lists

- `GET`: `/init`
    Initialize database, create testset table, and insert some records.
- `GET`: `/mysql` 
    Select some integer from mysql (range 0 - testsize)
- `GET`: `/arcus`
    Select some integer from arcus if missed, select from mysql and add to arcus
- `GET`: `/nbase`
    Select some integer from nbase if missed, select from mysql and add to nbase

### MySQL

Pulling from public mysql dockerfile(version `5.7`, but it can also `latest`).
There are some enviroments for mysql db.

- *MYSQL_ROOT_PASSWORD*: password
- *MYSQL_USER*: maybe
- *MYSQL_PASSWORD*: password
- *MYSQL_DATABASE*: ite3068

### Arcus

Pulling from [ruo91/arcus](https://hub.docker.com/r/ruo91/arcus/) and some appendix scripts for memcached server.
See `arcus/install.sh`. It provide arcus to memcached server list and set up zookeeper and memcached.
It automatically run after docker container started.

Arcus is memory cache cloud based on [memcached](https://memcached.org/) and [zookeeper](https://zookeeper.apache.org/). So Arcus is distributed cache cloud. Therefore Arcus can configure multiple nodes to improve performance.

In this project, use 3 memcached-server on arcus. You can see how many nodes are used in settings.json.

### nBase-ARC

Pulling from [hyeongseok05/nbase-arc](https://hub.docker.com/r/hyeongseok05/nbase-arc/).
Dockerfile prepare all for start nbase-arc, so just start docker container is enough.

nBase-ARC is distributed storage platform using [redis](https://redis.io/) and clustering.
A cluster consists of several gateways and replication groups, and each replication group has its own storage unit called redis.
There is an advantage that the redis API can be used as it is.

In this project use 4-cluster node server.

### Hubblemon

It's difficult to compose all of docker container in a single `docker-compose`. 
*Because, arcus and memcached require settins after container started.*

So, hubblemon run each `mysql`, `arcus`, `nbase`.
To do this, after each container started, process `hubblemon/install.sh`.

Script contains below

- Install depedency (It takes time depending on the internet(or repo server) speed)
- Clone hubblemon repository
- Copy each setting
- Install python dependency
- Run server

Hubblemon initially started to monitor Arcus. So it is not very difficult and kind to use with other clients while currently support other clients,

Monitoring is based on a django, web and information about the listener is delivered when the client is connected, and the client executes the information collected by the server to be monitored.

Since hubblemon runs in the container, it can take a very long time to start(In my case, about 5-6m).

In settings.json, you can see `HUBBLEMON` environments. This port is the port where HubbleMon runs in each container. And `install.sh` script on hubblemon dir is for run hubblemon in each container.

- MySQL: 4584
- Arcus: 4585 
- nBase-ARC:  4586

![Running hubblemon on arcus](https://github.com/jiunbae/ITE3068/blob/master/results/hubblemon.png?raw=true)

### nGrinder Test

nGrinder supports writing a script that sends an HTTP request and sends it to the agent for testing. 
The web server receives the HTTP request and can communicate with the mysql server or the arcus server. 
Now you can actually write those scripts through nGrinder to compare performance differences between using the mysql server directly and the arcus server.

You can test the performance in nGrinder by calling the api created in the flask app above.

Check my results in `results` directory.

## Review

The performance test of `mysql`, `arcus` and `nbase`.

![mysql](https://github.com/jiunbae/ITE3068/blob/master/results/mysql.png?raw=true)

*mysql ngrinder results*

![arcus](https://github.com/jiunbae/ITE3068/blob/master/results/arcus.png?raw=true)

*arcus ngrinder result*

![nbase](https://github.com/jiunbae/ITE3068/blob/master/results/nbase.png?raw=true)

*nbase ngrinder result*

nBase-ARC is fastest because memory cache is faster than access raw db. Over 10 times faster than mysql. 
Now that mysql is running too slow, arcus is slowing down.
If you increase the experiment time or if mysql is fast enough, the acus will be even faster.

## Usage 

Just run application, it uses docker service.

```
pip3 install -r api/requirements.txt

python3 app.py start

// To setup arcus memcached server
docker exec -it arcus-admin /bin/bash -c /opt/install.sh

// To setup hubblemon, each container
docker exec -it arcus-memcached-1 /bin/bash -c /opt/install.sh
docker exec -it mysql /bin/bash -c /opt/install.sh
docker exec -it nbase-arc /bin/bash -c /opt/install.sh

// You can run this command in other shell after above execution print `Docker started!`
// Waiting hubblemon takes too long, You do not have to wait for hubblemon to start the api server.
python3 api/app.py
```

for stop service,

```
python3 app.py stop
```
