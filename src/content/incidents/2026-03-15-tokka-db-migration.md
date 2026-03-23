---
title: "Tokka DB Migration"
date: 2026-03-15T09:00:00+09:00
resolvedDate: 2026-03-15T11:30:00+09:00
severity: maintenance
status: resolved
affectedServices:
  - Tokka
published: true
timeline:
  - time: 2026-03-15T09:00:00+09:00
    status: scheduled
    message: "데이터베이스 마이그레이션 작업 시작."
  - time: 2026-03-15T09:30:00+09:00
    status: monitoring
    message: "마이그레이션 진행 중. 서비스 일시 중단."
  - time: 2026-03-15T11:30:00+09:00
    status: resolved
    message: "마이그레이션 완료. 정상 운영 재개."
---

## 사고 개요

Tokka 서비스의 데이터베이스 스키마 업데이트를 위한 정기 점검입니다. 약 2시간 30분간 서비스가 일시 중단되었습니다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | Tokka (웹 분석) |
| 작업 시간 | ~2시간 30분 (09:00 ~ 11:30 KST) |
| 영향받은 사용자 | Tokka 웹 서비스 사용자 (일시 중단) |
| 데이터 손실 | 없음 |

## 작업 내용

- 분석 결과 테이블 스키마 최적화
- 인덱스 재구성으로 쿼리 성능 개선
