---
title: "Bubbles: 비흡연자를 위한 담타를 만들었다"
description: "담타(damta.world)를 보고 떠올렸다 — 흡연자에게 담배 타임이 있다면, 비흡연자에게는 비눗방울이 있어야 하지 않을까? GLSL 박막 간섭 셰이더, 결정론적 물리, Redis 멀티팟 아키텍처로 만든 멀티플레이어 비눗방울 게임 빌드로그."
date: 2026-03-25
permalink: /bubbles-multiplayer-bubble-game
tags: [dev, WebGL, GLSL, ThreeJS, WebSocket, Redis, SideProject]
published: true
---

> 목표: **담배 안 피는 사람도 잠깐 나가서 쉴 구실**이 있으면 좋겠다.
> 결과: 링크 하나로 같이 비눗방울을 불 수 있는 웹앱을 만들었다. 엑셀로 위장도 된다.

## 담타를 보고

[담타(damta.world)](https://www.damta.world/)를 처음 봤을 때 생각이 꽤 오래 머물렀다.

한국 직장인이라면 누구나 아는 그 장면이 있다. 흡연자 동료들이 "담배 한 대만" 하고 나가면, 비흡연자는 자리에 남아서 전화를 받는다. 담타는 그 흡연 시간을 효율적으로 만들어주는 서비스인데, 나는 반대쪽에서 생각했다.

**흡연자에게 담배 타임이 있다면, 비흡연자에게도 뭔가 있어야 하지 않을까?**

커피는 너무 흔하고, 산책은 너무 길다. 비눗방울은 어떨까. 불면 3초 만에 끝나고, 같이 불면 묘하게 웃긴다. 아무 의미 없는데 그게 좋다. 경쟁 없이, 규칙 없이, 같은 공간에서 같은 걸 하는 것. 그래서 만들었다.

```
Frontend: React 19 + Three.js (R3F) + Zustand
Backend:  Hono + Bun + WebSocket
State:    Redis (Pub/Sub + Hash Map)
Database: MongoDB
Infra:    Kubernetes + ArgoCD + Nginx
```

가입 없이 링크 하나로 들어와서 바로 같이 불 수 있어야 한다는 게 처음부터 끝까지의 원칙이었다. 담타처럼 — 앱 열고, 바로 쓰고, 돌아오는 것.

## 비눗방울답게 보이게 하기

비눗방울이 비눗방울처럼 보이려면 **박막 간섭(thin-film interference)**을 구현해야 한다. 빛이 비누막 앞뒤면에서 반사되면서 파장별로 간섭을 일으켜 무지갯빛이 생기는 그 현상이다. 동그란 반투명 구체를 띄우는 건 쉽지만, 그건 유리구슬이지 비눗방울이 아니다.

Fragment shader 핵심:

```glsl
// 광경로차(OPD) — Snell's law, 굴절률 n=1.33
float cosI = abs(dot(normal, viewDir));
float sinT = (1.0 / n) * sqrt(1.0 - cosI * cosI);
float cosT = sqrt(1.0 - sinT * sinT);
float opd = 2.0 * n * thickness * cosT;

// RGB 파장별 간섭
interference.r = 0.5 + 0.5 * cos(2.0 * PI * opd / 650.0); // 빨강 650nm
interference.g = 0.5 + 0.5 * cos(2.0 * PI * opd / 510.0); // 초록 510nm
interference.b = 0.5 + 0.5 * cos(2.0 * PI * opd / 475.0); // 파랑 475nm
```

`thickness`는 `vUv.y`에 따라 변한다. 실제 비누막이 중력으로 아래쪽이 두꺼워지는 것과 같은 원리다. 여기에 3옥타브 FBM 노이즈로 표면 물결을 만들고, Fresnel 효과로 가장자리 반사를 강화하고, 듀얼 스펙큘러로 하이라이트를 얹었다.

환경맵 반사는 의도적으로 껐다. 80개 버블이 동시에 떠있을 때 스프라이트 아티팩트가 보여서 — 한 개일 때는 예뻤는데 여러 개가 모이니까 깨졌다. `envMapIntensity=0`으로 밀어버리고 셰이더 자체의 간섭 패턴에만 의존하는 게 오히려 자연스러웠다.

## 80개를 한 번에 그리기

처음에는 버블마다 개별 `Mesh`를 만들었다. 80개 = 80 draw calls + 80 `useFrame`. 당연히 느렸다.

**InstancedMesh**로 바꿨다. 하나의 geometry + material로 80개 인스턴스를 한 번에 그린다.

```typescript
<instancedMesh ref={meshRef} args={[geometry, material, MAX_BUBBLES]} />
```

문제는 버블마다 투명도가 달라야 한다는 것. `instanceColor`는 Three.js가 기본 지원하지만 `instanceOpacity`는 없다. `onBeforeCompile`로 셰이더에 직접 attribute를 주입했다:

```typescript
shader.vertexShader = shader.vertexShader.replace(
  'void main() {',
  `attribute float instanceOpacity;
   varying float vInstanceOpacity;
   void main() {
     vInstanceOpacity = instanceOpacity;`
);
```

슬롯 관리는 LIFO 스택. 버블이 터지면 슬롯 번호를 push, 새로 생기면 pop. 배열 재정렬 없이 O(1).

## 같은 버블을 같이 보기

멀티플레이어에서 가장 까다로운 부분이었다. **모든 클라이언트가 같은 버블 궤적을 봐야 한다.**

서버가 매 프레임 위치를 브로드캐스트하면? 80개 × 60fps × N명 = 대역폭 폭발. 대신 **결정론적 물리**를 썼다. 서버는 버블 생성 시 `seed`만 전달하고, 각 클라이언트가 동일한 seed로 동일한 궤적을 독립적으로 계산한다.

PRNG는 Mulberry32를 가져왔다:

```typescript
export function seededRandom(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
```

바람은 3D Simplex Noise를 **방 시드**로 생성한다. 같은 방에 접속한 모든 클라이언트가 동일한 바람장을 계산하므로 버블이 같은 방향으로 흘러간다. 물리는 Semi-implicit Euler로 부력, 항력, 3D 바람, 횡진동, 소프트 바운더리를 적용했다.

버블이 터지는 것도 결정론적이다. 수명의 70% 이후부터 프레임 단위로 확률 체크를 하는데, seed 기반이라 모든 클라이언트에서 같은 프레임에 터진다:

```typescript
const tick = Math.floor(age * 60); // 프레임 레이트 독립적
const rng = seededRandom(seed * 1337 + tick);
const popChance = ((progress - 0.7) / 0.3) * 0.02;
return rng() < popChance;
```

## 크로스팟 동기화

Kubernetes에서 여러 팟이 뜨면, 유저 A는 Pod-1에, 유저 B는 Pod-2에 연결될 수 있다. 같은 방인데 서로의 버블이 안 보이면 멀티플레이어가 아니다.

```
Nginx Ingress → Pod-1 (Hono/Bun) ─┐
              → Pod-2 (Hono/Bun) ─┼→ Redis Pub/Sub
              → Pod-3 (Hono/Bun) ─┘
```

각 팟은 `room:{placeId}` 채널을 구독한다. 로컬 유저의 메시지는 로컬 브로드캐스트 + Redis publish, 다른 팟에서 온 메시지는 `originPodId` 체크 후 로컬에만 전달한다. 유저 목록은 `room:{placeId}:members` Hash Map에 저장해서 `HLEN`으로 정확한 크로스팟 접속자 수를 O(1)로 조회한다.

브로드캐스트할 때 `JSON.stringify`를 한 번만 하고 로컬 전송과 Redis publish에 같은 문자열을 재사용하는 것도 의외로 체감 차이가 있었다.

### 롤링 업데이트와 WebSocket

Kubernetes에서 팟이 교체될 때 WebSocket이 끊기면 유저 입장에서는 그냥 "끊겼다"가 된다. close code **1012**("Service Restart")를 보내고, 클라이언트는 이걸 감지하면 exponential backoff 없이 즉시 재연결하도록 했다. thundering herd를 막기 위해 200~1500ms 랜덤 지터만 넣었다.

```typescript
if (closeCode === 1012) {
  const jitter = 200 + Math.random() * 1300;
  setTimeout(() => this.doConnect(), jitter);
}
```

## 스텔스 모드

담타에서 영감을 받은 만큼, **회사에서 쓸 수 있어야 한다**는 생각이 자연스럽게 따라왔다.

`Ctrl+Shift+M`을 누르면 3D 버블 씬이 "Q1 운영.xlsx"라는 이름의 업무 관리 스프레드시트로 바뀐다. 게임 상태가 스프레드시트로 1:1 매핑된다:

| 게임 | 스프레드시트 |
|------|------------|
| 버블 불기 | 새 태스크 행 추가 (NEW) |
| 버블 터뜨리기 | 태스크 완료 (DONE) |
| 유저 입장/퇴장 | ACTIVE / ON HOLD |
| 크기 S/M/L | 우선순위 Low/Medium/High |
| 색상 8종 | 카테고리 (Marketing, Engineering, ...) |

수식 바에는 `=SUM(B2:B14)`, `=VLOOKUP(D5, ...)` 같은 가짜 수식이 뜬다. 행+열 해시로 결정론적이라 같은 셀을 클릭하면 항상 같은 수식이 보인다. 시트 탭, 리본 메뉴, 상태 바까지 만들어서 스크린샷으로는 진짜 엑셀과 구별이 어렵다.

~~흡연자들이 담배 피러 나가면 스텔스 모드로 비눗방울을 불면 된다.~~

## 버블 브레이크

담타의 핵심이 "잠깐 나갔다 오는 것"이라면, Bubbles에도 그 리듬이 있어야 한다고 생각했다.

화면 왼쪽 하단에 커피 아이콘이 있다. 누르면 1분, 3분, 5분 중 하나를 고를 수 있고, 프로그레스 링이 돌면서 남은 시간을 보여준다. 타이머가 끝나면 "휴식 완료!" 토스트가 뜨고, 오늘 몇 번 쉬었는지 `localStorage`에 기록된다.

기능적으로는 단순한 타이머인데, **"비눗방울 불면서 3분만 쉬자"**라는 구실을 만들어주는 게 포인트다. 담배 타임에 시간 제한이 있듯이.

## 시간에 따라 변하는 하늘

하루 종일 같은 하늘이면 심심하다. 실제 시간에 따라 새벽에는 분홍빛, 낮에는 파랑, 석양에는 주황, 밤에는 남색으로 그라데이션이 바뀌도록 했다. 60초마다 체크해서 부드럽게 전환된다. 퇴근 무렵에 접속하면 석양 아래서 비눗방울을 불 수 있다.

## 인증

WebSocket 연결 시 JWT를 URL 쿼리에 넣으면 서버 로그에 토큰이 남는다. **One-Time Ticket** 패턴으로 해결했다:

```
1. POST /api/auth/ws-ticket (JWT in header)
2. 서버: 30초 유효 ticket 발급
3. ws://host/ws?ticket=abc123
4. 서버: ticket 검증 후 즉시 삭제
```

로그에 ticket이 남아도 이미 소멸됐으니 재사용 불가.

## 삽질들

**React Error #300** — `BubbleSpawner`에서 조건부 `return null` 뒤에 `useRef`를 선언했더니 모드 전환 시 Hook 개수가 달라져서 React가 전체 트리를 파괴했다. WebGL 컨텍스트까지 같이 날아간다. 모든 Hook을 조건부 반환 위로 올려서 해결.

**InstancedMesh 클릭** — 80개 중 어떤 버블을 클릭했는지 O(1)로 알아내야 한다. `Raycaster`가 `event.instanceId`를 주니까, 슬롯→버블 ID 역방향 맵을 유지했다.

**크로스팟 유저 유령** — Pod-2에서만 유저가 안 보이는 버그. Redis MONITOR로 메시지 흐름을 추적해보니 `originPodId` 체크 로직에서 자기 팟 메시지를 무시하는 조건이 잘못 걸려 있었다. 새벽 3시에 고쳤다.

## 마무리

프론트엔드 ~3,500줄, 백엔드 ~1,500줄. 코드 양은 많지 않은데, 실시간 멀티플레이어 + 3D 렌더링 + 결정론적 동기화를 조합하면 예상 못 한 엣지 케이스가 계속 나온다.

가장 재미있었던 건 박막 간섭 셰이더 튜닝이었고, 가장 고통스러웠던 건 크로스팟 디버깅이었다.

담타 덕분에 시작한 프로젝트인데, 결국 만들고 나니까 나도 이걸로 쉬고 있다. 가입 없이 링크 하나로 들어와서 같이 비눗방울을 불 수 있다. 회사에서는 `Ctrl+Shift+M`으로.

---

**스택**: React 19, Three.js (R3F), Hono, Bun, Redis, MongoDB, Kubernetes

## 참고

- [담타 — damta.world](https://www.damta.world/)
- [Three.js InstancedMesh](https://threejs.org/docs/#api/en/objects/InstancedMesh)
- [Thin-Film Interference — Wikipedia](https://en.wikipedia.org/wiki/Thin-film_interference)
