---
title: 멀티코어 프로그래밍
date: 2017-12-26 18:40:00
description: "ITE4065: Parallel Programming @ Hanyang Univ."
tag: Hanyang, Parallel, Concurrency, Multicore
author: MaybeS
category: tech
---

# 멀티코어 프로그래밍

한양대학교에는 전설로만 내려오는 악명 높은 수업이 하나 존재한다. 바로 [정형수](https://sites.google.com/site/hyungsoojung/) 교수님의 멀티코어 프로그래밍이 그것이다. 첫 강의 개설시에는 A+가 하나도 없었다는 둥, 너무 어려워서 중도 포기자가 속출한다는 소문만 무성했던 수업이었다.

드디어 3학년이 되어 외부 활동도 어느정도 마무리가 되었고, 학점도 얼마 남지 않아 여유로운 마음을 가지고 수강 신청을 하게 되었다.

## Class

사실 이 수업이 어려운 이유는 수업 자체의 디펜던시가 강하기 때문이다. 특히 정형수 교수님의 데이터 베이스`ITE2038`나 운영체제`ELE3021`을  **수강하지 않은** 학생은 이해하기 난해한 부분이 많다. 물론 교수님이 이제 학생들의 실력을 어느정도 이해하시고 대부분의 (*이미 배웠지만*) 모르는 부분을 다시 설명하시고 넘어가신다.

정형수 교수님은 매우 힘이 넘치는 강의를 진행하신다. 원래는 2시간의 이론 수업과 2시간의 실습 수업으로 이루어진 3학점 강의이지만, 대부분의 이론 수업시간은 3시간에 육박하는 수업을 풀타임으로 진행하신다.

## Assignments

몇 가지 과제와 4개의 프로젝트를 진행 했다. 목록은 아래와 같다. 각 프로젝트는 영문으로 위키가 작성되어 있어 [GitHub](https://github.com/MaybeS/ITE4065)에서 확인 할 수 있다.

### Project

- [x] Project1: [Signal: Concurrent String Search](https://github.com/MaybeS/ITE4065/wiki/project1-multi)
- [x] Project2: [Simple Two-phase locking with Read-Write Lock](https://github.com/MaybeS/ITE4065/wiki/Project2)
- [x] Project3: [Wait-Free Snapshot](https://github.com/MaybeS/ITE4065/wiki/Project3)
- [x] Project4: [Scalable Lock Manager](https://github.com/MaybeS/ITE4065/wiki/Project4) (with mariaDB)

### Lab

- [x] Lab01: pthread practice
- [x] Lab02: mutex practice
- [x] Lab03: get prime with mutex and prime
- [x] Lab04: make simple thread pool
- [x] Lab05: try `vargrind`
- [x] Lab06: `boost::asio`
- [x] Lab07: thread pool using `boost::thread_group`
- [x] Lab08: mariaDB Install and initialize
- [x] Lab09: Cscope and sysbench
- [x] Lab10: Spinlock
- [x] Lab11: Conccurent Linked List
- [x] Lab12: jemalloc
- [x] Lab13: Concurrent Queue


## Review

너무 악명이 높아서인지 기대했던 것 만큼이나 수업이나 과제가 어렵진 않았다. 물론 과제의 난이도가 처음 개설되었을 때 보다 쉬워진 이유도 있지만, 작년 과제에 비해 어려웠다는걸 생각해보면 딱히 그렇지도 않은 것 같다. 과제(*특히 마지막인 MariaDB를 뜯어 고치는 과제*)가 어렵다고 느껴지는 이유중 가장 큰 이유는 **병렬적인 설계**와는 별계로 거대한 프로젝트에 익숙하지 않아서 인것 같다. 물론 **병렬적 설계**는 매우 어려우며 과제 전반에 거쳐 수업에서 배운 중요한 부분들을 녹여낼 수 있어야 한다.

