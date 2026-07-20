---
title: "Bubbles: 비흡연자를 위한 담타를 만들었다"
description: "담타(damta.world)를 보고 생각했다 — 흡연자에게 담배 타임이 있다면 비흡연자에게는 비눗방울이 있어야 하지 않을까. 박막 간섭 셰이더와 seed 기반 결정론적 물리로, 3월 23일 저녁부터 이틀 동안 만든 멀티플레이어 비눗방울 웹앱 기록."
date: 2026-03-25
permalink: /bubbles-multiplayer-bubble-game
tags: [dev, WebGL, GLSL, ThreeJS, WebSocket, Redis, SideProject]
published: true
---

[담타(damta.world)](https://www.damta.world/)를 처음 봤을 때 생각이 꽤 오래 머물렀다.

한국 직장인이라면 누구나 아는 장면이 있다. 흡연자 동료들이 "담배 한 대만" 하고 나가면, 비흡연자는 자리에 남아서 전화를 받는다. 담타는 그 흡연 시간을 효율적으로 만들어주는 서비스인데, 나는 반대쪽에서 생각했다. 흡연자에게 담배 타임이 있다면, 비흡연자에게도 뭔가 있어야 하지 않을까.

커피는 너무 흔하고, 산책은 너무 길다. 비눗방울은 어떨까. 불면 몇 초 만에 끝나고, 같이 불면 묘하게 웃긴다. 아무 의미 없는데 그게 좋다. 경쟁 없이, 규칙 없이, 같은 공간에서 같은 걸 하는 것. 그래서 만들었다. 가입 없이 링크 하나로 들어와서 바로 같이 불 수 있어야 한다는 게 처음부터 끝까지의 원칙이었다.

첫 커밋이 3월 23일 저녁 8시 8분이다. 이 글을 쓰는 25일 저녁까지 커밋이 56개 쌓였는데, 그 이틀 동안 뭘 만들었고 어디서 갈아엎었는지를 적어둔다. 스택은 React 19 + Three.js(R3F)에 Hono/Bun 서버, 상태 동기화는 Redis다.

## 비눗방울답게 보이게 하기

동그란 반투명 구체를 띄우는 건 쉽다. 그런데 그건 유리구슬이지 비눗방울이 아니다. 비눗방울이 비눗방울처럼 보이려면 **박막 간섭**(thin-film interference)이 필요하다. 빛이 비누막 앞뒤면에서 반사되면서 파장별로 간섭을 일으켜 무지갯빛이 생기는 그 현상이다.

fragment shader의 핵심은 이 함수다. 굴절률 1.33(물)로 광경로차를 구하고, RGB 세 파장에 대해 각각 간섭을 계산한다.

```glsl
vec3 thinFilmInterference(float cosTheta, float thickness) {
  float n = 1.33;
  float sinThetaR = sin(acos(cosTheta)) / n;
  float cosThetaR = sqrt(1.0 - sinThetaR * sinThetaR);
  float opd = 2.0 * n * thickness * cosThetaR;

  vec3 interference;
  interference.r = 0.5 + 0.5 * cos(2.0 * 3.14159 * opd / 650.0);
  interference.g = 0.5 + 0.5 * cos(2.0 * 3.14159 * opd / 510.0);
  interference.b = 0.5 + 0.5 * cos(2.0 * 3.14159 * opd / 475.0);

  return interference;
}
```

막 두께는 `vUv.y`에 따라 아래쪽이 두꺼워지게 했다. 실제 비누막이 중력 때문에 아래로 처지는 것과 같은 원리다. 여기에 3옥타브 FBM 노이즈로 표면이 일렁이게 하고, Fresnel로 가장자리 반사를 얹었다. 버블마다 `u_seed`를 넣어서 같은 셰이더라도 무늬가 제각각이다.

환경맵 반사는 의도적으로 껐다. 버블 한 개일 때는 예뻤는데 여러 개가 모이니까 스프라이트 아티팩트가 보여서, `envMapIntensity = 0`으로 밀어버리고 셰이더의 간섭 패턴에만 의존했다. 그게 오히려 자연스러웠다. 코드에는 지금도 `// disable environment map reflection (removes sprite artifacts)` 주석이 남아 있다.

![실제 렌더링된 Bubbles 화면. 밤하늘의 공원 배경에 나무와 가로등, 벤치가 놓여 있고 그 사이로 반투명한 비눗방울이 여러 개 떠오르고 있다.](/images/posts/bubbles-multiplayer-bubble-game/scene.png)

사실 첫날 밤에 이미 한 번 갈아엎었다. 바닥에도 커스텀 셰이더를 썼다가 밤 9시 54분에 `MeshStandardMaterial`로 되돌린 커밋이 있다. 버블 셰이더는 공들일 가치가 있었지만 바닥까지 직접 짤 이유는 없었다.

그리고 예쁘게 만드는 것과 별개로, **보이게** 만드는 데 하루를 썼다. 배경이 밤 공원이다 보니 투명한 버블이 어두운 배경에 묻혀서 안 보였다. 24일 커밋 로그에 그 흔적이 그대로 있다. 오전에 "more transparent bubbles, ambient lighting" — 낮 12시 28분에 "Fresnel rim glow — visible against dark backgrounds" — 오후 1시 8분에 "much brighter scenes — stronger ambient, wider streetlamp spread". 셰이더 수식보다 조명 튜닝이 더 오래 걸렸다.

## 80개를 한 번에 그리기

처음에는 버블마다 개별 `Mesh`를 만들었다. 80개면 80 draw calls. 24일 오전에 **InstancedMesh**로 바꿔서 하나의 geometry + material로 최대 80개(`MAX_BUBBLES = 80`)를 한 번에 그린다.

문제는 버블마다 투명도가 달라야 한다는 것. `instanceColor`는 Three.js가 기본 지원하지만 인스턴스별 opacity는 없다. `onBeforeCompile`로 셰이더에 `instanceOpacity` attribute를 직접 주입했다. 슬롯 관리는 LIFO 스택이다. 버블이 터지면 슬롯 번호를 push, 새로 생기면 pop. 배열 재정렬 없이 O(1)이고, 클릭 판정은 `Raycaster`가 주는 `instanceId`를 슬롯→버블 ID 역방향 맵으로 되돌린다.

## 같은 버블을 같이 보기

멀티플레이어에서 가장 까다로운 부분이었다. 모든 클라이언트가 같은 버블 궤적을 봐야 한다.

서버가 매 프레임 위치를 브로드캐스트하면 80개 × 60fps × N명으로 대역폭이 터진다. 대신 **결정론적 물리**를 썼다. 서버는 버블 생성 시 seed와 만료 시각만 전달하고, 각 클라이언트가 동일한 seed로 동일한 궤적을 독립적으로 계산한다. PRNG는 mulberry32를 가져왔다.

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

바람은 3D simplex noise인데, 노이즈 함수 자체를 **방 seed**로 초기화한다. 같은 방에 접속한 클라이언트는 모두 같은 바람장을 계산하므로 버블이 같은 방향으로 흘러간다. 적분은 semi-implicit Euler로 부력, 속도 제곱에 비례하는 항력, 바람, 횡진동(wobble), 소프트 바운더리를 순서대로 적용한다.

터지는 것도 결정론적이어야 한다. 내 화면에서 터진 버블이 남의 화면에 살아있으면 곤란하니까. 수명의 70%를 넘기면 프레임 단위로 확률 체크를 하는데, 나이를 60fps 틱으로 양자화해서 프레임 레이트와 무관하게 모든 클라이언트가 같은 틱에서 같은 판정을 낸다.

```typescript
const tick = Math.floor(age * 60);
const rng = seededRandom(seed * 1337 + tick);
const popChance = ((progress - 0.7) / 0.3) * 0.02;
return rng() < popChance;
```

처음부터 이랬던 건 아니다. 24일 오전 9시 44분에 "sync bubble seed/expiresAt across clients", 10시 3분에 "sync bubble physics using server timestamps" 커밋이 연달아 있다. 첫날 밤 버전은 클라이언트마다 버블이 제멋대로 움직였고, 클라이언트 로컬 시간으로 나이를 재다가 늦게 들어온 사람에게는 궤적이 어긋났다. 기준 시각을 서버 타임스탬프로 통일하고 나서야 "같은 버블을 같이 보는" 게 됐다.

## 크로스팟 동기화

Kubernetes에서 서버 팟이 여러 개 뜨면, 유저 A는 Pod-1에, 유저 B는 Pod-2에 연결될 수 있다. 같은 방인데 서로의 버블이 안 보이면 멀티플레이어가 아니다.

각 팟은 Redis Pub/Sub의 `room:{placeId}` 채널을 구독한다. 로컬 유저의 메시지는 로컬 브로드캐스트 + Redis publish, 다른 팟에서 온 메시지는 `originPodId`가 자기 팟이면 무시하고 아니면 로컬에만 전달한다. 브로드캐스트할 때 `JSON.stringify`를 한 번만 하고 로컬 전송과 Redis publish에 같은 문자열을 재사용하는 것도 잊지 않았다.

접속자 수도 처음엔 팟이 자기 로컬 세션만 세고 있어서, 어느 팟에 붙었느냐에 따라 사람 수가 다르게 보였다. 유저 목록을 `room:{placeId}:members` Hash로 옮기고 `HLEN`으로 세도록 고친 게 24일 오후 4시 50분 커밋이다.

롤링 업데이트도 신경 썼다. 팟이 교체될 때 WebSocket이 그냥 끊기면 유저 입장에서는 장애다. 서버가 내려갈 때 close code **1012**(Service Restart)를 보내고, 클라이언트는 이 코드를 받으면 exponential backoff 없이 바로 재연결한다. thundering herd를 막기 위한 200~1500ms 랜덤 지터만 두고.

## 인증 티켓

WebSocket 연결 시 JWT를 URL 쿼리에 넣으면 서버 로그에 토큰이 남는다. 첫날 버전이 정확히 그렇게 하고 있었고, 24일 오전에 one-time ticket으로 교체했다. JWT는 헤더로 보내서 30초짜리 일회용 티켓을 발급받고, WebSocket은 그 티켓으로 연결한다. 서버는 Redis `GETDEL`로 검증과 동시에 티켓을 지우니까 로그에 남아도 재사용이 안 된다. 처음엔 티켓을 서버 메모리에 뒀는데, 그러면 티켓을 발급한 팟과 WebSocket이 붙는 팟이 다를 때 깨진다는 걸 리뷰에서 지적받고 Redis로 옮겼다.

이틀 내내 이런 리뷰 라운드를 돌렸다. 커밋 로그에 R05부터 R09까지 라운드 번호가 남아 있는데, [BurstPick](/posts/burstpick-ai-agent-reviewers)에서 쓴 것과 같은 방식의 AI 코드 리뷰다. 혼자 이틀 만에 만드는 프로젝트일수록 다른 눈이 필요하다.

## 스텔스 모드

담타에서 영감을 받은 만큼, 회사에서 쓸 수 있어야 한다는 생각이 자연스럽게 따라왔다.

`Ctrl+Shift+M`을 누르면 3D 버블 씬이 업무 스프레드시트로 바뀐다. 게임 상태가 그대로 매핑된다. 버블을 불면 상태 NEW인 태스크 행이 추가되고, 터뜨리면 DONE이 되고, 버블 색상은 Marketing이니 Operations니 하는 카테고리가 된다. 수식 바에는 `=SUM(B2:B15)`, `=VLOOKUP(A2,Sheet2!A:D,3,FALSE)` 같은 가짜 수식이 뜨는데, 행+열 해시로 골라서 같은 셀을 클릭하면 항상 같은 수식이 보인다. 시트 탭, 리본 메뉴, 상태 바까지 만들어서 스크린샷으로는 진짜와 구별이 어렵다.

![스텔스 모드 화면. 3D 비눗방울 씬이 'Task Tracker — Q1 Operations.xlsx'라는 업무 스프레드시트로 위장되어, 불었던 버블들이 상태 NEW·담당자·우선순위·카테고리를 가진 태스크 행으로 매핑되어 있다.](/images/posts/bubbles-multiplayer-bubble-game/stealth.png)

~~흡연자들이 담배 피러 나가면 스텔스 모드로 비눗방울을 불면 된다.~~

## 담타의 리듬

담타의 핵심이 "잠깐 나갔다 오는 것"이라면 Bubbles에도 그 리듬이 있어야 한다고 생각해서, 오늘(25일) 버블 브레이크 타이머를 넣었다. 커피 아이콘을 누르면 1분, 3분, 5분 중 하나를 고를 수 있고 프로그레스 링이 남은 시간을 보여준다. 오늘 몇 번 쉬었는지는 `localStorage`에 쌓인다. 기능적으로는 그냥 타이머인데, "비눗방울 불면서 3분만 쉬자"라는 구실을 만들어주는 게 포인트다.

하늘도 실제 시간을 따라간다. 새벽, 낮, 석양, 밤 네 구간의 그라데이션을 정의해두고 60초마다 현재 시각으로 보간한다. 퇴근 무렵에 접속하면 석양 아래서 비눗방울을 불 수 있다.

## 삽질 하나

24일 오후 5시 6분 커밋: "fix: React error #300 — useRef after conditional return in BubbleSpawner". 조건부 `return null` 뒤에 `useRef`를 선언했더니 모드 전환 시 Hook 개수가 달라져서 React가 트리를 통째로 다시 마운트했고, WebGL 컨텍스트까지 같이 날아갔다. 에러 메시지는 minified #300이라 처음엔 뭔지도 몰랐다. 모든 Hook을 조건부 반환 위로 올려서 해결. Rules of Hooks를 몰라서가 아니라, 스텔스/비주얼 모드 분기를 넣다가 무심코 저지른 거라 더 머쓱했다.

## 마무리

25일 저녁 기준으로 프론트엔드가 9천 줄, 서버가 2천7백 줄쯤 된다. 이틀치고는 코드가 꽤 나왔는데, 실시간 멀티플레이어 + 3D 렌더링 + 결정론적 동기화를 조합하면 예상 못 한 엣지 케이스가 계속 나와서 커밋의 절반이 fix다.

가장 재미있었던 건 박막 간섭 셰이더였고, 가장 오래 걸린 건 의외로 "밤 배경에서 버블이 보이게 하기"였다.

담타 덕분에 시작한 프로젝트인데, 만들고 나니까 나도 이걸로 쉬고 있다. [bubbles.jiun.dev](https://bubbles.jiun.dev)에서 가입 없이 링크 하나로 들어와서 같이 비눗방울을 불 수 있다. 회사에서는 `Ctrl+Shift+M`으로.

## 참고

- [담타 — damta.world](https://www.damta.world/)
- [Three.js InstancedMesh](https://threejs.org/docs/#api/en/objects/InstancedMesh)
- [Thin-Film Interference — Wikipedia](https://en.wikipedia.org/wiki/Thin-film_interference)
