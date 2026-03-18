---
title: "Example: Tokka DB Migration"
date: 2026-03-15T09:00:00+09:00
resolvedDate: 2026-03-15T11:30:00+09:00
severity: maintenance
status: resolved
affectedServices:
  - Tokka
published: false
timeline:
  - time: 2026-03-15T09:00:00+09:00
    status: scheduled
    message: "Scheduled maintenance for database migration."
  - time: 2026-03-15T09:30:00+09:00
    status: monitoring
    message: "Migration in progress. Service temporarily unavailable."
  - time: 2026-03-15T11:30:00+09:00
    status: resolved
    message: "Migration completed successfully. All systems operational."
---

Tokka 서비스의 데이터베이스 마이그레이션 작업이 예정되어 있습니다.

## 영향 범위

- Tokka 웹 서비스 일시 중단
- 기존 분석 데이터는 보존됩니다

## 작업 내용

- PostgreSQL 스키마 업데이트
- 인덱스 재구성
