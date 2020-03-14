# FROM node
FROM node:alpine

MAINTAINER IascCHEN

# 更新Alpine的软件源为国内（清华大学）的站点 TUNA
RUN echo "https://mirror.tuna.tsinghua.edu.cn/alpine/v3.11/main/" > /etc/apk/repositories

RUN apk update \
    && apk add --no-cache ca-certificates \
    && update-ca-certificates \
    && apk add --no-cache --virtual .gyp bash bash-doc bash-completion vim wget python make g++

ARG NPM_REGISTRY="https://registry.npm.taobao.org"

RUN mkdir -p /opt/app/node

EXPOSE 3000
CMD ["yarn", "start"]

# use changes to package.json to force Docker not to use the cache
# when we change our application's nodejs dependencies:

RUN yarn config set registry ${NPM_REGISTRY}
RUN yarn config get registry

WORKDIR /opt/app/node
COPY node/package.json /opt/app/node/package.json
COPY node/yarn.lock /opt/app/node/yarn.lock
RUN yarn

WORKDIR /opt/app
COPY package.json /opt/app/package.json
COPY yarn.lock /opt/app/yarn.lock
RUN yarn

COPY . /opt/app
