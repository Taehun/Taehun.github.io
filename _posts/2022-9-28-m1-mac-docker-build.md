---
layout: post
title: Docker buildx로 멀티 플랫폼 이미지 빌드하기
categories: general
tags: [docker, buildx, mac]
toc: true
comments: true
excerpt_separator: <!--more-->
---

## 개요
노트북이 노후되어 최근 MacBook을 새로 하나 장만했습니다. M2 칩이 탑재된 MacBook Air 모델로 구매하였는대요. 지금까지 상당히 만족하면서 사용하고 있습니다.

M1이나 M2와 같은 애플 칩이 탑재된 MacBook 사용시 ARM64 아키텍처 호환 이슈를 가끔 겪게 되는대요. Docker 사용시에도 마찬가지로 아키텍처 호환 이슈를 신경써야 합니다.

<!--more-->

제가 겪었던 이슈를 한마디로 정리하면 _Docker 이미지 빌드 환경과 배포 환경간 아키텍처 미호환 이슈_ 입니다. Docker 이미지 빌드 환경은 제 노트북 (M2 Mac, arm64)이었고, 이미지를 배포하여 실행하는 환경은 클라우드 (Intel VM, x86_64)가 되면서 문제가 발생했습니다. M1/M2 MacBook을 사용하면서 기존 Intel MacBook과 차이점을 느끼지 못할정도로 프로그램들이 스무스하게 잘 돌아가서 내가 사용 중인 노트북이 ARM64 머신이라는 것을 종종 잊어먹습니다. ARM 머신에 각종 소프트웨어 포팅을 하던 임베디드 개발자 출신으로서 감회가 새롭네요. (~~라떼는 말이야...~~)

문제의 원인을 파악 하였으면 해결 방법은 간단합니다. `docker buildx`로 컨테이너 빌드시 멀티 아키텍처 지원 가능한 이미지를 빌드하여 배포하시면 됩니다.

## Docker Buildx

Docker Buildx은 멀티 플랫폼 빌드 기능을 지원하는 docker CLI 확장 플러그인 입니다. Docker 19.03 이후 버전부터 사용할 수 있습니다.

> 이후 내용은 모두 M2 MacBook Air에서 Docker Desktop을 설치하여 실행 한 것 입니다. (ver. 4.12.0)


`docker buildx ls`로 빌더 목록과 정보를 확인 합니다. 현재 사용 중인 빌더에 `*` 표시가 되어 있습니다.

```shell
$ docker buildx ls
NAME/NODE       DRIVER/ENDPOINT STATUS  BUILDKIT PLATFORMS
default *       docker
  default       default         running 20.10.17 linux/arm64, linux/amd64, linux/riscv64, linux/ppc64le, linux/s390x, linux/386, linux/arm/v7, linux/arm/v6
desktop-linux   docker
  desktop-linux desktop-linux   running 20.10.17 linux/arm64, linux/amd64, linux/riscv64, linux/ppc64le, linux/s390x, linux/386, linux/arm/v7, linux/arm/v6
```

`docker-container` 드라이버를 사용하는 새로운 빌더를 생성 합니다. (Why? default 빌더의 `docker` 드라이버는 멀티 플랫폼 기능이 지원되지 않습니다.)
```shell
$ docker buildx create --name multi-arch-builder --driver docker-container --bootstrap --use
[+] Building 3.3s (1/1) FINISHED
 => [internal] booting buildkit                                                         3.3s
 => => pulling image moby/buildkit:buildx-stable-1                                      2.7s
 => => creating container buildx_buildkit_multi-arch-builder0                           0.5s
multi-arch-builder
```
- `--driver`: 사용할 드라이버 설정
- `--bootstrap`: 빌더 생성이 끝나면 자동 초기화 합니다.
- `--use`: 새로 생성한 빌더를 사용하도록 자동 설정 합니다.

샘플 `Dockerfile`
```
FROM alpine:3.16
RUN apk add curl
```

M1/M2 MacBook과 Intel VM을 두 플랫폼을 지원하는 멀티 플랫폼 이미지를 빌드 합니다.
```shell
$ docker buildx build --platform linux/arm64,linux/amd64 -t buildx-test .
[+] Building 8.8s (8/8) FINISHED

  (...)

  => CACHED [linux/amd64 2/2] RUN apk add curl
  => CACHED [linux/arm64 2/2] RUN apk add curl
```

빌드후 `docker images`로 이미지 목록을 확인해보면, 빌드된 이미지가 없습니다. 왜냐하면 빌드된 플랫폼별 이미지는 캐싱되어 있어서 그렇습니다.
이미지를 어디로 내보낼지 설정해야 합니다.

**로컬로 가져오기 (테스트용)**

`--load` 옵션을 사용하여 이미지를 로컬로 가져 올 수 있습니다. 현재 로컬 플랫폼만 지원하도록 빌드가 되어서 `docker build`와 별반 차이가 없습니다. `docker buildx build`로 빌드가 되는지 여부와 빌드된 컨테이너 이미지를 로컬에서 테스트하는 목적으로 사용합니다.

```shell
$ docker buildx build --load -t buildx-test .
[+] Building 1.2s (7/7) FINISHED
  (...)

$  docker images
REPOSITORY      TAG               IMAGE ID       CREATED          SIZE
buildx-test     latest            ff1af71a52f1   19 minutes ago   9.83MB
```

**컨테이너 레지스트리에 올리기**

`--push` 옵션을 사용하여 멀티 플랫폼 이미지 빌드와 동시에 컨테이너 레지스트리 (_Docker Hub, ECR, GCR, ACR_)에 push 합니다. 이때는 `-t` 옵션에 올바른 컨테이너 레지스트리 URI를 사용해야 합니다. `docker login`으로 사용할 컨테이너 레지스트리에 로그인이 되어 있어야 합니다.

```shell
$ docker buildx build --platform linux/arm64,linux/amd64 -t <Container Registry URI>/buildx-test --push .
```

이렇게 컨테이너 레지스트리에 멀티 플랫폼 이미지를 올려두면, `docker pull`로 컨테이너 이미지를 가져올때 플랫폼에 해당하는 이미지만 가져옵니다. 즉, M1/M2 MacBook에서 `docker pull`을 하면 ARM64 컨테이너 이미지를 가져오고, Intel VM에서 `docker pull`을 하면 x86_64 컨테이너 이미지를 가져 옵니다.

**On M2 MacBook**

```shell
$ docker pull briankim/buildx-test
Using default tag: latest
latest: Pulling from briankim/buildx-test
9b18e9b68314: Already exists
14acd24d0487: Pull complete
Digest: sha256:31f4e41656c9c3ac009e90628244f71068b7572d7303187cf41c0c62b722837a
Status: Downloaded newer image for briankim/buildx-test:latest
docker.io/briankim/buildx-test:latest

$ docker run --rm -it briankim/buildx-test
/ # arch
aarch64
```

**On Intel VM**

```shell
$ docker pull briankim/buildx-test
Using default tag: latest
latest: Pulling from briankim/buildx-test
9b18e9b68314: Already exists
14acd24d0487: Pull complete
Digest: sha256:31f4e41656c9c3ac009e90628244f71068b7572d7303187cf41c0c62b722837a
Status: Downloaded newer image for briankim/buildx-test:latest
docker.io/briankim/buildx-test:latest

$ docker run --rm -it briankim/buildx-test
/ # arch
x86_64
```

## 요약

- 팀 중에 M1/M2 MacBook 사용자가 있으면 `docker buildx`로 멀티 플랫폼을 지원하도록 빌드 합시다.