---
layout: ../../../layouts/IgemoyaDoc.astro
title: "이게모야 개인정보 처리방침"
description: "iOS 앱 이게모야가 어떤 정보를 어디에서 어떻게 다루는지 설명합니다."
updated: "시행일 2026-09-04 · 기준 앱 버전 0.10(26)"
---

- 시행일: 2026-09-04 (초안 작성 2026-09-02, 기준 앱 버전 0.10(26))
- 운영자(개인정보 처리자): 배지운 (Jiun Bae)
- 문의: jiunbae.623@gmail.com

이 문서는 iOS 앱 **이게모야**(번들 ID `ai.jiun.dogam`, 이하 “앱”)가 어떤 정보를 어디에서 어떻게 다루는지 설명합니다. 앱이 실제로 하는 일만 적었고, 하지 않는 일은 하지 않는다고 적었습니다. 영어 요약은 마지막에 있습니다.

## 1. 한눈에 보기

- 회원가입, 비밀번호, 이메일 수집이 없습니다. 기기의 iCloud 계정으로 동기화합니다.
- 원본 사진은 저장하거나 전송하지 않습니다. 물건만 오려 낸 파생 카드 이미지 한 장(긴 변 최대 960px, 480KiB 이하)만 남습니다.
- 얼굴이 보이는 사진은 카드가 되지 않고 버려집니다. 사람·동물로 분류된 물체도 카드가 되지 않습니다.
- 앱 서버는 Apple iCloud(CloudKit)뿐입니다. 광고, 분석 도구, 추적, 제3자 SDK가 없습니다.
- 카드를 ‘친구’ 또는 ‘전체’로 공개할 때만 카드 이미지와 카드 내용, 별명, 프로필 아이콘이 다른 사람에게 전달됩니다.
- 내 카드는 앱에서 언제든 기기와 iCloud에서 지울 수 있습니다.

## 2. 수집하지 않는 정보

앱은 다음 정보를 수집·저장·전송하지 않습니다.

- 원본 카메라 사진, EXIF, 촬영 위치
- 위치 정보, 연락처, 사진 보관함(앱은 사진 보관함을 읽지 않습니다)
- 실명, 이메일, 전화번호, 생년월일, 결제 정보
- 사진 속 문자(OCR), 얼굴 특징값
- 광고 식별자(IDFA), 기기 고유 식별자, 사용 통계, 크래시 리포트를 위한 제3자 SDK
- 웹 탐색 기록, 검색어(앱 내 검색은 기기 안에서만 처리)

## 3. 기기 안에서만 처리되는 정보

촬영 버튼을 누르면 카메라 프레임 한 장이 메모리에서만 다음 순서로 처리됩니다. 이 단계의 데이터는 어디에도 저장·전송되지 않습니다.

1. **얼굴 검사.** Apple Vision으로 프레임 전체에서 얼굴을 찾습니다. 얼굴이 하나라도 있으면 즉시 중단하고 “사람이 보여요. 사물만 발견할 수 있어요.”를 표시합니다. 검사를 실행할 수 없으면 카드를 만들지 않습니다.
2. **피사체 분리.** 가운데 가이드가 가리키는 전경 물체 하나만 오려 냅니다. 물체를 찾지 못하거나 배경 전체가 잡히면 아무것도 저장하지 않습니다.
3. **분류.** 오려 낸 물체를 Apple Vision 분류기로 보고 물품 종류·종족·재질·색상·형태를 앱의 고정 분류표에 맞춥니다. 분류 결과에 사람·동물 계열 라벨이 있으면 “사람이나 동물은 도감에 오르지 않아요.”를 표시하고 중단합니다.
4. **설명 생성.** Apple Intelligence가 켜진 기기에서는 Apple Foundation Models(온디바이스)가 분류표와 관찰 태그만 받아 설명 문장을 씁니다. 이미지는 모델에 전달되지 않습니다. Apple Intelligence가 없거나 꺼진 기기에서는 앱에 내장된 문장 조합으로 설명을 채웁니다. 종 이름은 앱이 정한 후보 목록에서 고릅니다.
5. **카드 합성.** 오려 낸 물체에 흰 외곽선을 입히고 흐린 배경 위에 얹어 카드 이미지를 만듭니다. 원본 프레임과 중간 결과물은 이 단계가 끝나면 버려집니다.

## 4. 기기에 저장되는 정보

사용자가 “내 도감에 채집”을 누른 카드만 저장됩니다.

| 항목 | 내용 | 저장 위치 |
|---|---|---|
| 카드 이미지 | 파생 이미지 1장(JPEG 또는 투명 PNG, 긴 변 최대 960px, 480KiB 이하) | 앱 데이터 |
| 카드 내용 | 종 이름, 설명, 물품 종류, 종족·재질·색상·형태, 희귀도, 수집 가치, 채집 날짜, 편집 여부 | `collection.json` |
| 시각 지문 | 같은 물건을 다시 찍었을 때 알아보기 위한 16×16 색상 요약과 9×8 해시. 원본 복원이 불가능하며 ‘나만’ 사본에만 남고 친구·전체 공개본에서는 제거됩니다 | `collection.json` |
| 프로필 | 별명(앱이 무작위로 정한 12개 한국어 별칭 중 하나를 기본값으로, 사용자가 바꿀 수 있음), 프로필 아이콘(기본 아이콘 4종 또는 내 카드 하나의 128px 파생물), 새 카드의 기본 공개 범위, 덱 카드 순서 | `social.json` |
| 숨김·차단 목록 | 내가 숨긴 카드의 레코드 이름과 차단한 작성자의 CloudKit 사용자 레코드 이름 | `social.json` (기기 전용, 동기화 안 함) |
| 동기화 제어 정보 | 아직 iCloud에 올리지 못한 변경 목록, 마지막 동기화 시각, 설치 단위 무작위 ID | `social.json` |
| 온보딩 완료 여부 | 첫 실행 안내를 봤는지 여부 | UserDefaults |
| 모모 카드 사진 | 연습 상대 모모의 연필 사진(생성 또는 번들) | Application Support 캐시 |

`collection.json`과 `social.json`은 iOS 파일 보호가 적용되고 iCloud/iTunes 백업에서 제외됩니다. 앱을 삭제하면 이 파일들도 함께 지워집니다.

## 5. iCloud(CloudKit)로 전송되는 정보

앱은 Apple CloudKit 컨테이너 `iCloud.ai.jiun.dogam` 하나만 사용합니다. 어떤 정보가 어디까지 가는지는 카드의 **공개 범위**가 정합니다.

### 5.1 ‘나만’ — 내 iCloud 개인 데이터베이스

- 저장된 카드(이미지, 내용, 시각 지문), 프로필, 삭제 기록이 내 iCloud 개인 데이터베이스에 동기화됩니다.
- 이 데이터는 iCloud 계정 소유자만 읽을 수 있습니다. 운영자(개발자)는 개인 데이터베이스를 열람할 수 없습니다.
- iCloud에 로그인하지 않았거나 오프라인이면 카드는 기기에만 있고, 연결되면 올라갑니다.

### 5.2 ‘친구’ — 초대를 수락한 친구와의 공유 영역

- 프로필에서 만든 초대 링크를 상대가 수락하면 내 iCloud 개인 데이터베이스 안의 공유 영역에 상대가 접근합니다. 연락처는 사용하지 않고 링크만 오갑니다.
- 공유되는 항목: 카드 이미지, 카드 내용(시각 지문 제거), 카드 소유자의 CloudKit 사용자 레코드 이름, 별명, 프로필 아이콘 파생물(최대 128px), 공개 시각, 희귀도 등 메타데이터.
- 초대를 수락한 친구는 내 ‘친구’ 공개 카드를 읽고, 같은 공유 영역에 교환 요청을 쓸 수 있습니다.
- 운영자는 이 공유 영역을 열람할 수 없습니다.

### 5.3 ‘전체’ — 공개 데이터베이스

- 카드가 CloudKit 공개 데이터베이스에 올라가고, 이게모야 사용자 누구나 갤러리에서 볼 수 있습니다.
- 공개되는 항목은 5.2와 같습니다. 카드 소유자의 CloudKit 사용자 레코드 이름은 차단 기능과 소유자 확인에 쓰이며 앱은 이를 이름·이메일로 변환하지 않습니다.
- 운영자는 CloudKit Console에서 공개 데이터베이스의 카드와 신고를 열람·삭제할 수 있습니다. 이는 커뮤니티 규칙 위반 대응에만 사용합니다.

### 5.4 교환 요청

- 친구 카드에 교환을 요청하면 요청 양식이 친구 공유 영역에 기록됩니다. 양식에는 보낸 사람과 받는 사람의 CloudKit 사용자 레코드 이름과 별명, 카드 ID와 이름, 상태만 있고 이미지는 없습니다.
- 교환이 성사되면 상대는 내 카드의 **복제본**(새 ID, 시각 지문 없음)을 받습니다. 복제본은 상대의 카드가 되어 내가 원본을 지워도 남습니다.
- 모모(앱 내장 연습 상대)와의 교환은 기기 안에서만 처리됩니다.

### 5.5 신고

- 갤러리 카드를 신고하면 공개 데이터베이스에 신고 레코드가 생성됩니다. 내용: 대상 카드의 레코드 이름, 대상 소유자의 CloudKit 사용자 레코드 이름, 선택한 사유, 시각. CloudKit은 생성자(신고자)의 사용자 레코드 이름을 자동으로 기록합니다.
- 신고 레코드는 일반 사용자가 읽을 수 없고 운영자만 확인합니다. 신고자는 자신의 신고를 앱에서 취소·삭제할 수 없습니다.

### 5.6 Apple의 역할

CloudKit 데이터는 Apple의 iCloud 약관과 개인정보 처리방침에 따라 Apple 서버에 저장됩니다. 운영자는 공개 데이터베이스만 접근할 수 있습니다.

## 6. Apple Intelligence와 Image Playground

- **카드 설명**: Apple Intelligence가 켜진 기기에서 Apple Foundation Models가 기기 안에서 실행됩니다. 앱은 이미지를 보내지 않고 분류표·관찰 태그·서술 방식만 전달합니다. 모델 사용 여부는 iOS 설정의 Apple Intelligence 상태를 따릅니다.
- **모모의 사진**: 연습 상대 모모의 연필 카드 사진은 iOS 26.4 이상 실기기에서 Apple Image Playground(ImageCreator)의 기기 내(on-device) 스타일(일러스트·애니메이션·스케치)로만 생성을 시도합니다. ChatGPT 확장 등 외부 제공자 스타일은 코드에서 제외되어 있으며 프롬프트가 기기 밖으로 나가지 않습니다. 프롬프트는 앱에 고정된 영어 문장이며 사용자 사진·이름·데이터를 포함하지 않습니다. 사용자가 iOS 설정에서 외부 제공자(예: ChatGPT)를 허용한 경우 iOS가 그 고정 프롬프트를 해당 제공자로 보낼 수 있습니다. 생성이 불가능하면 앱에 포함된 사진을 씁니다. 사용자의 카드는 Image Playground를 거치지 않습니다.

## 7. 권한

| 권한 | 용도 | 없으면 |
|---|---|---|
| 카메라 | 물건 촬영. 앱 내 카메라만 있고 사진 보관함 선택은 없습니다 | 카드를 만들 수 없음 |
| 사진 추가 | 공유 메뉴에서 카드 이미지를 사진 앱에 저장할 때만 iOS가 요청합니다. 앱은 사진 보관함을 읽지 않습니다 | 저장만 불가 |
| iCloud | 동기화와 친구·전체 공개 | 기기 전용으로 동작 |

## 8. 보관 기간과 삭제

- **카드 삭제**: 카드 옵션 메뉴의 “기기와 iCloud에서 지우기”를 확인하면 기기, 개인 데이터베이스, 친구 공유 영역, 공개 데이터베이스의 사본이 삭제됩니다. 오프라인이면 다음 연결 때 완료됩니다. 삭제 기록(카드 ID와 시각)은 다른 기기가 카드를 되살리지 않도록 최대 90일 뒤 정리됩니다.
- **교환 복제본**: 교환으로 상대가 받은 복제본은 상대의 도감에 남습니다.
- **신고 레코드**: 운영자가 처리한 뒤 삭제하며, 처리 근거 확인을 위해 최대 90일까지만 보관합니다.
- **앱 삭제**: 기기의 카드·프로필·숨김 목록이 지워집니다. iCloud 개인 데이터베이스의 사본은 iOS 설정 › Apple 계정 › iCloud › 저장 공간 관리 › 이게모야에서 삭제할 수 있습니다. **공개 갤러리에 올린 카드는 앱을 지우기 전에 앱 안에서 삭제하거나, 운영자에게 삭제를 요청하세요.**
- **숨김·차단 목록**: 기기에만 있으며 현재 앱 안에서 해제하는 화면이 없습니다. 해제가 필요하면 문의해 주세요.

## 9. 아동

앱은 아동을 대상으로 하지 않으며 나이를 묻지 않습니다. 사람 사진을 자동 거부하고 별도 개인정보를 수집하지 않습니다. 예상 연령 등급은 4+입니다.

## 10. 정보 주체의 권리

- 앱 안에서 카드 열람·수정·삭제, 공개 범위 변경, 별명·아이콘 변경이 가능합니다.
- 공개 갤러리에 남은 내 카드나 신고 관련 문의, 차단 해제 요청, 그 밖의 개인정보 관련 요청은 위 문의처로 보내 주세요. 처리 시 iCloud 계정 확인을 위해 카드 이름이나 채집 날짜 같은 정보를 요청할 수 있습니다.
- 한국 거주자는 개인정보 보호법에 따른 열람·정정·삭제·처리정지 요구권을 가지며, 위 문의처로 행사할 수 있습니다.

## 11. 변경

이 방침을 바꾸면 앱 지원 페이지와 이 문서의 시행일을 갱신합니다. 수집 항목이 늘어나는 변경은 앱 업데이트 설명에도 적습니다.

## English summary

이게모야 ("What's this?") turns a photo of an everyday object into a fictional collectible card. There is no sign-up; the app syncs through the device's iCloud account via Apple CloudKit and talks to no other server. The original photo is never stored or uploaded: a face check runs first and discards any frame with a person, objects classified as people or animals are refused, and only a derived card image (≤960 px, ≤480 KiB) plus the card's text and taxonomy are kept. Descriptions are written on device by Apple Foundation Models from tags only (the image never reaches the model), with built-in templates on devices without Apple Intelligence. Data leaves the user's private iCloud database only when the user sets a card's scope to 친구 (friends who accepted an invitation link) or 전체 (public gallery); what is shared is the derived image, card text, a pseudonymous nickname, a 128 px avatar derivative, and the opaque CloudKit user record name. The developer can see and delete public-database records only, uses them solely for moderation, and cannot read private or friend data. No analytics, ads, tracking, or third-party SDKs are used. Users can delete cards (removing all cloud copies), change scope, report, hide, and block in the app; cards copied to a friend through a trade remain the friend's. Contact: jiunbae.623@gmail.com.
