var e=(e,t)=>()=>(e&&(t=e(e=0)),t),t=(e,t)=>()=>(t||e((t={exports:{}}).exports,t),t.exports);(function(){let e=document.createElement(`link`).relList;if(e&&e.supports&&e.supports(`modulepreload`))return;for(let e of document.querySelectorAll(`link[rel="modulepreload"]`))n(e);new MutationObserver(e=>{for(let t of e)if(t.type===`childList`)for(let e of t.addedNodes)e.tagName===`LINK`&&e.rel===`modulepreload`&&n(e)}).observe(document,{childList:!0,subtree:!0});function t(e){let t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin===`use-credentials`?t.credentials=`include`:e.crossOrigin===`anonymous`?t.credentials=`omit`:t.credentials=`same-origin`,t}function n(e){if(e.ep)return;e.ep=!0;let n=t(e);fetch(e.href,n)}})();var n,r=e((()=>{n=class{storageKey;constructor(e){this.storageKey=`playground:${e}:auth`}async loginIfAvailable(){let e=localStorage.getItem(this.storageKey);return e?JSON.parse(e):null}getUser(){let e=localStorage.getItem(this.storageKey);return e?JSON.parse(e):null}logout(){localStorage.removeItem(this.storageKey)}}})),i,a=e((()=>{i=class{storageKey;constructor(e){this.storageKey=`playground:${e}:scores`}async submit(e){let t=this.getHistorySync();t.push({score:e.score,timestamp:Date.now(),meta:e.meta}),localStorage.setItem(this.storageKey,JSON.stringify(t))}async getMyBest(){let e=this.getHistorySync();return e.length===0?null:e.reduce((e,t)=>t.score>e.score?t:e)}async getHistory(){return this.getHistorySync()}getHistorySync(){let e=localStorage.getItem(this.storageKey);return e?JSON.parse(e):[]}}})),o,s=e((()=>{o=class{storageKey;constructor(e){this.storageKey=`playground:${e}:saves`}async save(e){localStorage.setItem(this.storageKey,JSON.stringify({data:e,savedAt:Date.now()}))}async load(){let e=localStorage.getItem(this.storageKey);return e?JSON.parse(e).data:null}async sync(){}}})),c,l=e((()=>{c=class{storageKey;constructor(e){this.storageKey=`playground:${e}:leaderboard`}async top(e=10){return this.getEntriesSync().sort((e,t)=>t.score-e.score).slice(0,e).map((e,t)=>({...e,rank:t+1}))}async aroundMe(){return this.getEntriesSync()}getEntriesSync(){let e=localStorage.getItem(this.storageKey);return e?JSON.parse(e):[]}}})),u,d=e((()=>{u=class{handlers=[];connected=!1;async connect(){this.connected=!0,console.warn(`[PlaygroundSDK] Multiplayer: running in offline stub mode`)}async disconnect(){this.connected=!1,this.handlers=[]}async send(e){if(!this.connected)throw Error(`Not connected. Call connect() first.`);for(let t of this.handlers)t(e)}onMessage(e){return this.handlers.push(e),()=>{this.handlers=this.handlers.filter(t=>t!==e)}}}})),f,p=e((()=>{r(),a(),s(),l(),d(),f=class e{auth;scores;saves;leaderboard;multiplayer;config;constructor(e){this.config=e,this.auth=new n(e.game),this.scores=new i(e.game),this.saves=new o(e.game),this.leaderboard=new c(e.game),this.multiplayer=new u}static init(t){return new e(t)}getConfig(){return{...this.config}}}}));t((()=>{p();var e=null;try{e=f.init({apiUrl:`https://api.jiun.dev`,game:`bluff-party`})}catch{}var t=!1;try{e&&(t=!!e.auth.getUser())}catch{}async function n(){if(e)try{t=!!await e.auth.loginIfAvailable();let n=document.getElementById(`btn-sdk-login`);n&&(n.textContent=t?`👤`:`🔒`)}catch{}}var r=`playground_bluff-party_leaderboard`;function i(){try{let e=localStorage.getItem(r);return e?JSON.parse(e):[]}catch{return[]}}function a(e,t){try{let n=i(),a=n.find(t=>t.name===e);a||(a={name:e,totalWins:0,gamesPlayed:0,winRate:0,timestamp:Date.now()},n.push(a)),a.gamesPlayed++,t&&a.totalWins++,a.winRate=Math.round(a.totalWins/a.gamesPlayed*100),a.timestamp=Date.now(),n.sort((e,t)=>t.totalWins-e.totalWins||t.winRate-e.winRate),localStorage.setItem(r,JSON.stringify(n.slice(0,50)))}catch{}}function o(){let e=i().slice(0,10),t=[`🥇`,`🥈`,`🥉`];D.innerHTML=`
    <div class="title-bg" style="display:flex;flex-direction:column;align-items:center;height:100%;padding:20px;width:100%;overflow-y:auto;">
      <h2 style="font-size:28px;font-weight:900;color:${u.yellow};margin:20px 0 5px;">\u{1F3C6} \uB9AC\uB354\uBCF4\uB4DC</h2>
      <p style="font-size:13px;color:${u.gray};margin-bottom:20px;">\uBE14\uB7EC\uD504 \uD30C\uD2F0 \uCC54\uD53C\uC5B8</p>

      <div style="width:100%;max-width:380px;">
        <div style="display:flex;padding:8px 16px;color:${u.gray};font-size:12px;font-weight:700;">
          <span style="width:30px;">#</span>
          <span style="flex:1;">\uC774\uB984</span>
          <span style="width:60px;text-align:center;">\uC2B9\uB9AC</span>
          <span style="width:60px;text-align:right;">\uC2B9\uB960</span>
        </div>
        ${e.length===0?`<p style="text-align:center;color:${u.gray};padding:40px 0;">\uC544\uC9C1 \uAE30\uB85D\uC774 \uC5C6\uC2B5\uB2C8\uB2E4</p>`:e.map((e,n)=>`
            <div style="display:flex;align-items:center;padding:12px 16px;background:rgba(255,255,255,0.05);border-radius:12px;margin-bottom:6px;">
              <span style="width:30px;font-size:${n<3?`18`:`14`}px;font-weight:700;color:${n===0?u.yellow:n===1?`#c0c0c0`:n===2?`#cd7f32`:u.white};">${n<3?t[n]:n+1}</span>
              <span style="flex:1;font-weight:700;color:${u.white};">${e.name}</span>
              <span style="width:60px;text-align:center;font-weight:700;color:${u.yellow};">${e.totalWins}</span>
              <span style="width:60px;text-align:right;font-size:13px;color:${u.gray};">${e.winRate}%</span>
            </div>
          `).join(``)}
      </div>

      <button id="btn-lb-back" style="background:${u.secondary};color:white;border:none;padding:14px 40px;border-radius:50px;font-size:16px;font-weight:700;cursor:pointer;margin-top:20px;">
        \uB3CC\uC544\uAC00\uAE30
      </button>
    </div>
  `,document.getElementById(`btn-lb-back`).addEventListener(`click`,()=>{_=`title`,R()})}var s=[{normal:`고양이`,bluffer:`강아지`,category:`동물`},{normal:`피자`,bluffer:`햄버거`,category:`음식`},{normal:`비행기`,bluffer:`헬리콥터`,category:`탈것`},{normal:`해`,bluffer:`달`,category:`자연`},{normal:`기타`,bluffer:`바이올린`,category:`악기`},{normal:`사과`,bluffer:`딸기`,category:`과일`},{normal:`축구공`,bluffer:`농구공`,category:`스포츠`},{normal:`우산`,bluffer:`모자`,category:`소품`},{normal:`로봇`,bluffer:`외계인`,category:`캐릭터`},{normal:`성`,bluffer:`탑`,category:`건물`},{normal:`나무`,bluffer:`꽃`,category:`자연`},{normal:`자전거`,bluffer:`오토바이`,category:`탈것`},{normal:`펭귄`,bluffer:`오리`,category:`동물`},{normal:`케이크`,bluffer:`아이스크림`,category:`디저트`},{normal:`집`,bluffer:`텐트`,category:`건물`}],c=[{question:`지구에서 가장 높은 산은?`,correctAnswer:`에베레스트`,blufferAnswer:`K2`,category:`지리`},{question:`물의 화학식은?`,correctAnswer:`H2O`,blufferAnswer:`CO2`,category:`과학`},{question:`한국의 수도는?`,correctAnswer:`서울`,blufferAnswer:`부산`,category:`상식`},{question:`태양계에서 가장 큰 행성은?`,correctAnswer:`목성`,blufferAnswer:`토성`,category:`과학`},{question:`"로미오와 줄리엣"의 작가는?`,correctAnswer:`셰익스피어`,blufferAnswer:`괴테`,category:`문학`},{question:`올림픽은 몇 년마다 열리나?`,correctAnswer:`4년`,blufferAnswer:`2년`,category:`스포츠`},{question:`빛의 속도에 가장 가까운 것은?`,correctAnswer:`초속 30만 km`,blufferAnswer:`초속 15만 km`,category:`과학`},{question:`피카소의 국적은?`,correctAnswer:`스페인`,blufferAnswer:`프랑스`,category:`예술`},{question:`인체에서 가장 큰 장기는?`,correctAnswer:`피부`,blufferAnswer:`간`,category:`과학`},{question:`BTS의 데뷔곡은?`,correctAnswer:`No More Dream`,blufferAnswer:`Danger`,category:`음악`},{question:`일본의 수도는?`,correctAnswer:`도쿄`,blufferAnswer:`오사카`,category:`지리`},{question:`1 + 1 = ?`,correctAnswer:`2`,blufferAnswer:`11 (이진법)`,category:`수학`}],l=[{normal:`김치찌개`,bluffer:`된장찌개`,category:`한식`},{normal:`여름`,bluffer:`겨울`,category:`계절`},{normal:`학교`,bluffer:`회사`,category:`장소`},{normal:`결혼식`,bluffer:`장례식`,category:`행사`},{normal:`영화관`,bluffer:`놀이공원`,category:`장소`},{normal:`크리스마스`,bluffer:`할로윈`,category:`명절`},{normal:`아기`,bluffer:`할아버지`,category:`사람`},{normal:`바다`,bluffer:`산`,category:`자연`},{normal:`라면`,bluffer:`떡볶이`,category:`음식`},{normal:`지하철`,bluffer:`버스`,category:`교통`},{normal:`도서관`,bluffer:`카페`,category:`장소`},{normal:`생일`,bluffer:`졸업식`,category:`행사`},{normal:`강아지`,bluffer:`고양이`,category:`동물`},{normal:`운동회`,bluffer:`소풍`,category:`학교행사`},{normal:`비 오는 날`,bluffer:`눈 오는 날`,category:`날씨`}],u={bg:`#1a1a2e`,bgLight:`#16213e`,primary:`#e94560`,secondary:`#0f3460`,accent:`#533483`,yellow:`#f5c542`,green:`#27ae60`,blue:`#3498db`,orange:`#e67e22`,white:`#ecf0f1`,gray:`#7f8c8d`,darkGray:`#2c3e50`},d=[`#e94560`,`#3498db`,`#27ae60`,`#f5c542`,`#e67e22`,`#9b59b6`,`#1abc9c`,`#e74c3c`],m=[],h=0,g=5,_=`title`,v=null,y=0,b=[],x=[],S=[],C=[],w=[],T=[],E=[],D=document.getElementById(`app`),O=document.createElement(`style`);O.textContent=`
  /* Animated background gradient for title */
  @keyframes bgShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  .title-bg {
    background: linear-gradient(135deg, ${u.bg}, ${u.accent}, ${u.secondary}, ${u.bg}, #2a1a4e);
    background-size: 400% 400%;
    animation: bgShift 12s ease infinite;
  }

  /* Mask emoji rotation */
  @keyframes maskFloat {
    0%, 100% { transform: rotate(-5deg) scale(1); }
    25% { transform: rotate(5deg) scale(1.05); }
    50% { transform: rotate(-3deg) scale(1.02); }
    75% { transform: rotate(4deg) scale(1.04); }
  }
  .mask-emoji {
    display: inline-block;
    font-size: 96px;
    animation: maskFloat 4s ease-in-out infinite;
    filter: drop-shadow(0 4px 20px rgba(233, 69, 96, 0.3));
  }

  /* Player input focus */
  .player-input:focus {
    border-color: var(--player-color) !important;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--player-color) 30%, transparent);
    outline: none;
  }
  .player-input {
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  /* Role card flip */
  @keyframes cardFlipIn {
    0% { transform: perspective(600px) rotateY(90deg); opacity: 0; }
    100% { transform: perspective(600px) rotateY(0deg); opacity: 1; }
  }
  .role-card {
    animation: cardFlipIn 0.6s cubic-bezier(0.34, 1.2, 0.64, 1) both;
  }

  /* Bluffer pulse */
  @keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0 rgba(233, 69, 96, 0.5); }
    50% { box-shadow: 0 0 30px 8px rgba(233, 69, 96, 0.3); }
  }
  .pulse-bluffer {
    animation: pulseRed 2s ease-in-out infinite;
  }

  /* Citizen pulse */
  @keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.5); }
    50% { box-shadow: 0 0 30px 8px rgba(39, 174, 96, 0.3); }
  }
  .pulse-citizen {
    animation: pulseGreen 2s ease-in-out infinite;
  }

  /* Timer pulse red */
  @keyframes timerPulse {
    0%, 100% { transform: scale(1); color: ${u.primary}; }
    50% { transform: scale(1.15); color: #ff3333; }
  }
  .timer-urgent {
    animation: timerPulse 0.6s ease-in-out infinite;
  }

  /* Progress bar */
  .round-progress {
    width: 100%;
    max-width: 300px;
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    overflow: hidden;
    margin: 8px 0;
  }
  .round-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, ${u.primary}, ${u.yellow});
    border-radius: 3px;
    transition: width 0.5s ease;
  }

  /* Vote button styles */
  .vote-btn-enhanced {
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
  }
  .vote-btn-enhanced:active {
    transform: scale(0.95);
  }

  /* Vote bounce */
  @keyframes voteBounce {
    0% { transform: scale(1); }
    30% { transform: scale(1.08); }
    50% { transform: scale(0.95); }
    70% { transform: scale(1.03); }
    100% { transform: scale(1); }
  }
  .vote-selected {
    animation: voteBounce 0.5s ease-out;
  }

  /* Confetti dots */
  @keyframes confettiDot {
    0% { transform: translateY(0) rotate(0deg); opacity: 1; }
    100% { transform: translateY(-120px) rotate(720deg); opacity: 0; }
  }
  .confetti-container {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    overflow: hidden;
  }
  .confetti-dot {
    position: absolute;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: confettiDot var(--dur) ease-out var(--delay) infinite;
  }

  /* Phase dot glow */
  @keyframes dotGlow {
    0%, 100% { text-shadow: 0 0 4px currentColor; }
    50% { text-shadow: 0 0 12px currentColor, 0 0 20px currentColor; }
  }
  .phase-active {
    animation: dotGlow 1.5s ease-in-out infinite;
  }

  /* Winner announcement */
  @keyframes winnerPop {
    0% { transform: scale(0.5); opacity: 0; }
    60% { transform: scale(1.1); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
  }
  .winner-announce {
    animation: winnerPop 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both;
  }

  /* Slide in from bottom */
  @keyframes slideUp {
    from { transform: translateY(30px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  .slide-up {
    animation: slideUp 0.4s ease-out both;
  }

  /* Button hover glow */
  .btn-glow:active {
    filter: brightness(1.1);
    transform: scale(0.97);
  }
`,document.head.appendChild(O);function k(e,t=[]){let n=e.map((e,t)=>({item:e,i:t})).filter(e=>!t.includes(e.i));if(n.length===0){let t=Math.floor(Math.random()*e.length);return{item:e[t],index:t}}let r=n[Math.floor(Math.random()*n.length)];return{item:r.item,index:r.i}}function A(e){switch(e){case`drawing`:return`그림 그리기`;case`quiz`:return`퀴즈 대결`;case`describe`:return`설명하기`}}function j(e){switch(e){case`drawing`:return`🎨`;case`quiz`:return`🧠`;case`describe`:return`💬`}}function M(){m.forEach(e=>{e.isBluffer=!1,e.votedFor=-1});let e=E.slice(-2).map(e=>e.blufferIndex),t=m.map((e,t)=>t).filter(t=>!e.includes(t)),n=t.length>0?t:m.map((e,t)=>t),r=n[Math.floor(Math.random()*n.length)];m[r].isBluffer=!0}function N(){let e=[`drawing`,`quiz`,`describe`],t=e.filter(e=>!b.includes(e)),n=t.length>0?t[Math.floor(Math.random()*t.length)]:e[Math.floor(Math.random()*e.length)];switch(b.push(n),n){case`drawing`:{let{item:e,index:t}=k(s,x);return x.push(t),{type:`drawing`,normalPrompt:e.normal,blufferPrompt:e.bluffer,category:e.category}}case`quiz`:{let{item:e,index:t}=k(c,S);return S.push(t),{type:`quiz`,normalPrompt:e.correctAnswer,blufferPrompt:e.blufferAnswer,category:`${e.category}: ${e.question}`}}case`describe`:{let{item:e,index:t}=k(l,C);return C.push(t),{type:`describe`,normalPrompt:e.normal,blufferPrompt:e.bluffer,category:e.category}}}}function P(){let e=[{key:`role`,label:`역할 확인`},{key:`game`,label:`미니게임`},{key:`vote`,label:`투표`},{key:`result`,label:`결과`}],t=``;if(_===`roundIntro`||_===`roleReveal`||_===`passPhone`)t=`role`;else if(_===`miniGame`)t=`game`;else if(_===`discussion`||_===`voting`||_===`votePassPhone`)t=`vote`;else if(_===`results`)t=`result`;else return``;return`<div style="display:flex;align-items:center;justify-content:center;padding:8px 0;width:100%;flex-wrap:wrap;">${e.map((n,r)=>{let i=n.key===t,a=e.findIndex(e=>e.key===t)>r,o=i?u.yellow:a?u.green:u.gray,s=i?`700`:`400`,c=i?`phase-active`:``,l=i?`<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${u.yellow};margin-right:4px;vertical-align:middle;box-shadow:0 0 8px ${u.yellow};" class="${c}"></span>`:``,d=r<e.length-1?`<span style="color:${u.gray};margin:0 4px;font-size:10px;">▸</span>`:``;return`<span style="color:${o};font-weight:${s};font-size:11px;" class="${c}">${l}${n.label}</span>${d}`}).join(``)}</div>`}function F(e=`뒤로`){return`<button id="btn-back" style="position:absolute;top:12px;left:12px;background:rgba(255,255,255,0.1);color:${u.gray};border:1px solid rgba(255,255,255,0.15);padding:6px 14px;border-radius:20px;font-size:13px;cursor:pointer;z-index:10;">${e}</button>`}function I(){let e=[`#e94560`,`#f5c542`,`#27ae60`,`#3498db`,`#9b59b6`,`#e67e22`,`#1abc9c`,`#ff6b9d`],t=``;for(let n=0;n<30;n++){let r=e[n%e.length],i=Math.random()*100,a=60+Math.random()*40,o=1.5+Math.random()*2,s=Math.random()*2,c=6+Math.random()*6;t+=`<div class="confetti-dot" style="left:${i}%;top:${a}%;width:${c}px;height:${c}px;background:${r};--dur:${o}s;--delay:${s}s;"></div>`}return`<div class="confetti-container">${t}</div>`}function L(){return h<=0||g<=0?``:`<div class="round-progress" style="max-width:300px;margin:4px auto 8px;"><div class="round-progress-fill" style="width:${h/g*100}%"></div></div>`}function R(){switch(_){case`title`:z();break;case`setup`:B();break;case`roundIntro`:H();break;case`roleReveal`:W();break;case`passPhone`:U();break;case`miniGame`:G();break;case`discussion`:X();break;case`voting`:Q();break;case`votePassPhone`:Z();break;case`results`:$();break;case`finalResults`:ee();break}}function z(){D.innerHTML=`
    <div class="title-bg" style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;position:relative;">
      <div style="position:absolute;top:12px;right:12px;display:flex;gap:8px;">
        <button id="btn-leaderboard" style="background:rgba(255,255,255,0.1);border:none;border-radius:50%;width:36px;height:36px;font-size:16px;cursor:pointer;opacity:0.8;">\u{1F3C6}</button>
        <button id="btn-sdk-login" style="background:rgba(255,255,255,0.1);border:none;border-radius:50%;width:36px;height:36px;font-size:16px;cursor:pointer;opacity:0.6;">${t?`👤`:`🔒`}</button>
      </div>
      <div class="mask-emoji" style="margin-bottom:10px;">\uD83C\uDFAD</div>
      <h1 style="font-size:42px;font-weight:900;background:linear-gradient(to right,${u.primary},${u.yellow});-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px;">BLUFF PARTY</h1>
      <p style="font-size:18px;color:${u.yellow};margin-bottom:30px;font-weight:700;">블러프 파티</p>
      <p style="font-size:14px;color:${u.gray};margin-bottom:30px;max-width:300px;line-height:1.6;">
        폰 하나, 친구 여럿<br>거짓말쟁이를 찾아라!
      </p>
      <button id="btn-start" style="background:linear-gradient(135deg,${u.primary},#c0392b);color:white;border:none;padding:18px 60px;border-radius:50px;font-size:20px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(233,69,96,0.4);transition:transform 0.2s;margin-bottom:14px;">
        게임 시작
      </button>
      <button id="btn-rules" style="background:rgba(255,255,255,0.1);color:${u.white};border:2px solid rgba(255,255,255,0.2);padding:12px 40px;border-radius:50px;font-size:16px;font-weight:700;cursor:pointer;">
        게임 방법
      </button>
      <p style="font-size:12px;color:${u.gray};margin-top:20px;">2~8명 / 폰 하나로 플레이</p>
    </div>

    <div id="rules-modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:100;display:none;align-items:center;justify-content:center;padding:20px;">
      <div style="background:${u.bgLight};border-radius:20px;padding:28px 24px;max-width:360px;width:100%;text-align:center;">
        <h3 style="font-size:22px;font-weight:900;color:${u.yellow};margin-bottom:20px;">게임 방법</h3>
        <div style="text-align:left;font-size:14px;line-height:2;color:${u.white};">
          <p>1. 한 명이 블러퍼(거짓말쟁이)로 선정됩니다</p>
          <p>2. 블러퍼는 다른 미션을 받습니다</p>
          <p>3. 미니게임 후 투표로 블러퍼를 찾으세요</p>
          <p>4. 블러퍼를 찾으면 시민 승리, 못 찾으면 블러퍼 승리!</p>
        </div>
        <button id="btn-close-rules" style="background:${u.primary};color:white;border:none;padding:12px 40px;border-radius:50px;font-size:16px;font-weight:700;cursor:pointer;margin-top:20px;">
          닫기
        </button>
      </div>
    </div>
  `,document.getElementById(`btn-sdk-login`).addEventListener(`click`,()=>n()),document.getElementById(`btn-leaderboard`).addEventListener(`click`,()=>{o()}),document.getElementById(`btn-start`).addEventListener(`click`,()=>{_=`setup`,R()}),document.getElementById(`btn-rules`).addEventListener(`click`,()=>{let e=document.getElementById(`rules-modal`);e.style.display=`flex`}),document.getElementById(`btn-close-rules`).addEventListener(`click`,()=>{let e=document.getElementById(`rules-modal`);e.style.display=`none`})}function B(){let e=m.map(e=>e.name),t=Math.max(m.length,2);D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;height:100%;padding:20px;width:100%;max-width:400px;overflow-y:auto;position:relative;">
      ${F(`처음으로`)}
      <h2 style="font-size:24px;font-weight:900;margin:20px 0 5px;color:${u.yellow};">\uD83D\uDC65 플레이어 설정</h2>
      <p style="font-size:13px;color:${u.gray};margin-bottom:20px;">이름을 입력하고 게임을 시작하세요</p>
      <div id="name-error" style="color:${u.primary};font-size:13px;font-weight:700;margin-bottom:8px;display:none;"></div>

      <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <button id="btn-minus" style="background:${u.secondary};color:white;border:none;width:40px;height:40px;border-radius:50%;font-size:20px;cursor:pointer;font-weight:700;">-</button>
        <span style="font-size:20px;font-weight:700;min-width:40px;text-align:center;" id="player-count">${t}명</span>
        <button id="btn-plus" style="background:${u.secondary};color:white;border:none;width:40px;height:40px;border-radius:50%;font-size:20px;cursor:pointer;font-weight:700;">+</button>
      </div>

      <div id="name-inputs" style="width:100%;display:flex;flex-direction:column;gap:10px;margin-bottom:20px;"></div>

      <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <span style="font-size:14px;color:${u.gray};">라운드 수:</span>
        <button id="btn-rounds-minus" style="background:${u.secondary};color:white;border:none;width:36px;height:36px;border-radius:50%;font-size:18px;cursor:pointer;">-</button>
        <span id="rounds-count" style="font-size:18px;font-weight:700;min-width:30px;text-align:center;">${g}</span>
        <button id="btn-rounds-plus" style="background:${u.secondary};color:white;border:none;width:36px;height:36px;border-radius:50%;font-size:18px;cursor:pointer;">+</button>
      </div>

      <button id="btn-go" style="background:linear-gradient(135deg,${u.primary},#c0392b);color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(233,69,96,0.4);margin-top:10px;">
        게임 시작!
      </button>
    </div>
  `;let n=t;function r(){let t=document.getElementById(`name-inputs`);t.innerHTML=``;for(let r=0;r<n;r++){let n=d[r%d.length];t.innerHTML+=`
        <div style="display:flex;align-items:center;gap:10px;">
          <div style="width:32px;height:32px;border-radius:50%;background:${n};display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;flex-shrink:0;">${r+1}</div>
          <input id="name-${r}" type="text" placeholder="플레이어 ${r+1}" value="${e[r]||``}"
            class="player-input"
            style="flex:1;padding:12px 16px;border-radius:12px;border:2px solid ${u.secondary};background:${u.bgLight};color:white;font-size:16px;font-family:'Noto Sans KR',sans-serif;outline:none;--player-color:${n};"
            maxlength="10" />
        </div>
      `}document.getElementById(`player-count`).textContent=`${n}명`}r(),document.getElementById(`btn-minus`).addEventListener(`click`,()=>{n>2&&(n--,r())}),document.getElementById(`btn-plus`).addEventListener(`click`,()=>{n<8&&(n++,r())}),document.getElementById(`btn-rounds-minus`).addEventListener(`click`,()=>{g>3&&(g--,document.getElementById(`rounds-count`).textContent=`${g}`)}),document.getElementById(`btn-rounds-plus`).addEventListener(`click`,()=>{g<7&&(g++,document.getElementById(`rounds-count`).textContent=`${g}`)}),document.getElementById(`btn-back`).addEventListener(`click`,()=>{_=`title`,R()}),document.getElementById(`btn-go`).addEventListener(`click`,()=>{let e=!1;for(let t=0;t<n;t++){let n=document.getElementById(`name-${t}`);n.value.trim()?n.style.borderColor=u.secondary:(e=!0,n.style.borderColor=u.primary)}if(e){let e=document.getElementById(`name-error`);e.textContent=`모든 플레이어의 이름을 입력해주세요!`,e.style.display=`block`;return}m=[];for(let e=0;e<n;e++){let t=document.getElementById(`name-${e}`).value.trim();m.push({name:t,score:0,isBluffer:!1,votedFor:-1})}h=0,b=[],x=[],S=[],C=[],E=[],V()})}function V(){if(h++,h>g){_=`finalResults`,R();return}v=N(),M(),w=[],T=Array(m.length).fill(0),y=0,_=`roundIntro`,R()}function H(){let e=v;D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;background:linear-gradient(135deg,${u.bg},${u.secondary});position:relative;">
      ${P()}
      <div style="font-size:16px;color:${u.yellow};font-weight:700;margin-bottom:8px;">라운드 ${h} / ${g}</div>
      <div style="font-size:64px;margin:20px 0;">${j(e.type)}</div>
      <h2 style="font-size:28px;font-weight:900;margin-bottom:10px;">${A(e.type)}</h2>
      <p style="font-size:14px;color:${u.gray};margin-bottom:8px;">카테고리: ${e.category}</p>
      <div style="background:${u.darkGray};padding:16px 24px;border-radius:16px;margin:20px 0;max-width:320px;">
        <p style="font-size:14px;color:${u.gray};line-height:1.6;">
          ${e.type===`drawing`?`각자 제시어를 보고 그림을 그립니다.<br>한 명은 다른 제시어를 받습니다!<br>누가 다른 그림을 그렸는지 찾아보세요.`:``}
          ${e.type===`quiz`?`모두에게 같은 질문이 주어집니다.<br>한 명은 다른 답을 받고 그걸 방어해야 합니다!<br>누가 틀린 답을 받았는지 찾아보세요.`:``}
          ${e.type===`describe`?`각자 단어를 보고 설명합니다.<br>한 명은 다른 단어를 받습니다!<br>누가 다른 단어를 설명했는지 찾아보세요.`:``}
        </p>
      </div>
      <button id="btn-next" style="background:linear-gradient(135deg,${u.primary},#c0392b);color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;margin-top:10px;box-shadow:0 4px 15px rgba(233,69,96,0.4);">
        역할 확인 시작
      </button>
    </div>
  `,document.getElementById(`btn-next`).addEventListener(`click`,()=>{y=0,_=`passPhone`,R()})}function U(){let e=m[y],t=d[y%d.length];D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;background:linear-gradient(135deg,${u.bg},${u.bgLight});">
      ${P()}
      <div style="font-size:64px;margin-bottom:20px;">\uD83D\uDCF1</div>
      <h2 style="font-size:22px;font-weight:700;margin-bottom:10px;">폰을 넘겨주세요!</h2>
      <div style="background:${t};width:80px;height:80px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:36px;font-weight:900;margin:20px 0;">${y+1}</div>
      <p style="font-size:24px;font-weight:900;color:${t};margin-bottom:30px;">${e.name}</p>
      <p style="font-size:14px;color:${u.gray};margin-bottom:30px;">준비되면 아래 버튼을 누르세요<br>다른 사람이 보지 못하게 하세요!</p>
      <button id="btn-reveal" style="background:linear-gradient(135deg,${u.accent},${u.secondary});color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(83,52,131,0.4);">
        내 역할 보기 \uD83D\uDC40
      </button>
    </div>
  `,document.getElementById(`btn-reveal`).addEventListener(`click`,()=>{_=`roleReveal`,R()})}function W(){let e=m[y],t=v,n=e.isBluffer,r=n?t.blufferPrompt:t.normalPrompt;d[y%d.length];let i=``,a=``;t.type===`drawing`?(i=n?`🕵️ 블러퍼!`:`👤 시민`,a=n?`당신은 블러퍼입니다!<br>다른 사람과 <b>다른 제시어</b>를 받았어요.<br>들키지 않게 비슷하게 그리세요!`:`당신은 시민입니다.<br>제시어를 보고 그림을 그려주세요.`):t.type===`quiz`?(i=n?`🕵️ 블러퍼!`:`👤 시민`,a=n?`당신은 블러퍼입니다!<br><b>틀린 답</b>을 받았어요.<br>자신있게 방어하세요!`:`당신은 시민입니다.<br>정답을 기억하세요.`):(i=n?`🕵️ 블러퍼!`:`👤 시민`,a=n?`당신은 블러퍼입니다!<br><b>다른 단어</b>를 받았어요.<br>들키지 않게 설명하세요!`:`당신은 시민입니다.<br>단어를 기억하고 설명해주세요.`),D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;background:${n?`linear-gradient(135deg, #c0392b, ${u.primary})`:`linear-gradient(135deg, ${u.secondary}, ${u.accent})`};">
      ${P()}
      <p style="font-size:14px;color:rgba(255,255,255,0.7);margin-bottom:5px;">${e.name}</p>
      <div style="font-size:24px;font-weight:900;margin-bottom:20px;">${i}</div>
      <div class="role-card ${n?`pulse-bluffer`:`pulse-citizen`}" style="background:rgba(0,0,0,0.3);padding:20px 30px;border-radius:20px;margin-bottom:20px;max-width:300px;border:2px solid ${n?`rgba(233,69,96,0.4)`:`rgba(39,174,96,0.4)`};">
        <p style="font-size:13px;color:rgba(255,255,255,0.7);margin-bottom:8px;">${t.type===`quiz`?`당신의 답`:`당신의 제시어`}</p>
        <p style="font-size:36px;font-weight:900;color:${u.yellow};">${r}</p>
      </div>
      <p style="font-size:14px;color:rgba(255,255,255,0.8);line-height:1.6;margin-bottom:30px;max-width:280px;">${a}</p>
      <button id="btn-confirm" style="background:rgba(255,255,255,0.2);color:white;border:2px solid rgba(255,255,255,0.4);padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;backdrop-filter:blur(10px);">
        확인했어요
      </button>
    </div>
  `,document.getElementById(`btn-confirm`).addEventListener(`click`,()=>{y++,y<m.length?(_=`passPhone`,R()):(y=0,_=`miniGame`,R())})}function G(){switch(v.type){case`drawing`:K();break;case`quiz`:J();break;case`describe`:Y();break}}function K(){if(y>=m.length){_=`discussion`,R();return}let e=m[y],t=d[y%d.length];D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;height:100%;padding:15px;width:100%;background:${u.bg};">
      ${P()}
      <div style="display:flex;justify-content:space-between;width:100%;align-items:center;margin-bottom:6px;">
        <span style="font-size:13px;color:${u.gray};">라운드 ${h}/${g}</span>
        <span style="font-size:13px;color:${t};font-weight:700;">${e.name}의 차례</span>
      </div>
      <div class="round-progress"><div class="round-progress-fill" style="width:${h/g*100}%"></div></div>
      <p style="font-size:13px;color:${u.gray};margin-bottom:10px;">\uD83C\uDFA8 제시어를 기억하고 그려주세요! (30초)</p>
      <div id="timer" style="font-size:24px;font-weight:900;color:${u.yellow};margin-bottom:10px;">30</div>
      <canvas id="draw-canvas" width="340" height="340" style="background:white;border-radius:12px;touch-action:none;cursor:crosshair;max-width:100%;"></canvas>
      <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;justify-content:center;">
        <button class="color-btn" data-color="#000000" style="width:44px;height:44px;border-radius:50%;background:#000;border:3px solid ${u.yellow};cursor:pointer;"></button>
        <button class="color-btn" data-color="#e74c3c" style="width:44px;height:44px;border-radius:50%;background:#e74c3c;border:3px solid transparent;cursor:pointer;"></button>
        <button class="color-btn" data-color="#3498db" style="width:44px;height:44px;border-radius:50%;background:#3498db;border:3px solid transparent;cursor:pointer;"></button>
        <button class="color-btn" data-color="#27ae60" style="width:44px;height:44px;border-radius:50%;background:#27ae60;border:3px solid transparent;cursor:pointer;"></button>
        <button class="color-btn" data-color="#f39c12" style="width:44px;height:44px;border-radius:50%;background:#f39c12;border:3px solid transparent;cursor:pointer;"></button>
        <button class="color-btn" data-color="#9b59b6" style="width:44px;height:44px;border-radius:50%;background:#9b59b6;border:3px solid transparent;cursor:pointer;"></button>
        <button id="btn-clear" style="width:44px;height:44px;border-radius:50%;background:${u.darkGray};border:2px solid ${u.gray};cursor:pointer;font-size:14px;color:white;display:flex;align-items:center;justify-content:center;">\u2716</button>
      </div>
      <button id="btn-done" style="background:linear-gradient(135deg,${u.green},#219a52);color:white;border:none;padding:14px 40px;border-radius:50px;font-size:16px;font-weight:700;cursor:pointer;margin-top:12px;box-shadow:0 4px 15px rgba(39,174,96,0.4);">
        완료
      </button>
    </div>
  `;let n=document.getElementById(`draw-canvas`),r=n.getContext(`2d`),i=window.devicePixelRatio||1;n.width=340*i,n.height=340*i,r.scale(i,i);let a=!1,o=`#000000`,s=0,c=0;r.lineCap=`round`,r.lineJoin=`round`,r.lineWidth=4;function l(e){let t=n.getBoundingClientRect(),r=n.width/t.width,i=n.height/t.height;return`touches`in e?{x:(e.touches[0].clientX-t.left)*r,y:(e.touches[0].clientY-t.top)*i}:{x:(e.clientX-t.left)*r,y:(e.clientY-t.top)*i}}n.addEventListener(`mousedown`,e=>{a=!0;let t=l(e);s=t.x,c=t.y}),n.addEventListener(`mousemove`,e=>{if(!a)return;let t=l(e);r.strokeStyle=o,r.beginPath(),r.moveTo(s,c),r.lineTo(t.x,t.y),r.stroke(),s=t.x,c=t.y}),n.addEventListener(`mouseup`,()=>{a=!1}),n.addEventListener(`mouseleave`,()=>{a=!1}),n.addEventListener(`touchstart`,e=>{if(e.preventDefault(),e.touches.length>1){a=!1;return}a=!0;let t=l(e);s=t.x,c=t.y},{passive:!1}),n.addEventListener(`touchmove`,e=>{if(e.preventDefault(),!a)return;let t=l(e);r.strokeStyle=o,r.beginPath(),r.moveTo(s,c),r.lineTo(t.x,t.y),r.stroke(),s=t.x,c=t.y},{passive:!1}),n.addEventListener(`touchend`,()=>{a=!1}),document.querySelectorAll(`.color-btn`).forEach(e=>{e.addEventListener(`click`,()=>{o=e.dataset.color,document.querySelectorAll(`.color-btn`).forEach(e=>e.style.borderColor=`transparent`),e.style.borderColor=u.yellow})}),document.getElementById(`btn-clear`).addEventListener(`click`,()=>{r.clearRect(0,0,n.width,n.height)});let f=30,p=document.getElementById(`timer`),v=setInterval(()=>{f--,p.textContent=`${f}`,f<=10&&f>5?(p.style.color=u.primary,p.classList.add(`timer-urgent`)):f<=5&&(p.style.color=`#ff3333`),f<=0&&(clearInterval(v),b())},1e3);function b(){clearInterval(v),w.push({playerIndex:y,dataUrl:n.toDataURL()}),y++,y<m.length?q():(_=`discussion`,R())}document.getElementById(`btn-done`).addEventListener(`click`,b)}function q(){let e=m[y],t=d[y%d.length];D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;background:${u.bg};">
      <div style="font-size:48px;margin-bottom:20px;">\uD83D\uDCF1</div>
      <p style="font-size:18px;color:${u.gray};margin-bottom:10px;">다음 플레이어에게 넘겨주세요</p>
      <div style="background:${t};width:70px;height:70px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:30px;font-weight:900;margin:15px 0;">${y+1}</div>
      <p style="font-size:22px;font-weight:900;color:${t};margin-bottom:25px;">${e.name}</p>
      <button id="btn-ready" style="background:linear-gradient(135deg,${u.accent},${u.secondary});color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;">
        준비 완료
      </button>
    </div>
  `,document.getElementById(`btn-ready`).addEventListener(`click`,()=>{_=`miniGame`,R()})}function J(){let e=v.category;D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;background:linear-gradient(135deg,${u.bg},${u.secondary});">
      ${P()}
      <div style="font-size:14px;color:${u.yellow};margin-bottom:5px;">라운드 ${h}/${g}</div>
      ${L()}
      <div style="font-size:48px;margin-bottom:15px;">\uD83E\uDDE0</div>
      <h2 style="font-size:20px;font-weight:700;margin-bottom:20px;">퀴즈 대결</h2>
      <div style="background:${u.darkGray};padding:20px 24px;border-radius:16px;margin-bottom:20px;max-width:340px;">
        <p style="font-size:13px;color:${u.gray};margin-bottom:8px;">질문</p>
        <p style="font-size:18px;font-weight:700;line-height:1.5;">${e.split(`: `).slice(1).join(`: `)}</p>
      </div>
      <div style="background:rgba(0,0,0,0.2);padding:16px 24px;border-radius:12px;margin-bottom:20px;max-width:340px;">
        <p style="font-size:13px;color:${u.gray};line-height:1.6;">
          모든 플레이어가 자신이 받은 답을 발표합니다.<br>
          한 명이 다른 답을 받았다는 것을 기억하세요!<br>
          돌아가며 자신의 답이 왜 맞는지 설명하세요.
        </p>
      </div>
      <p style="font-size:14px;color:${u.gray};margin-bottom:20px;">충분히 토론한 후 다음으로 진행하세요</p>
      <button id="btn-vote" style="background:linear-gradient(135deg,${u.primary},#c0392b);color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(233,69,96,0.4);">
        투표하기
      </button>
    </div>
  `,document.getElementById(`btn-vote`).addEventListener(`click`,()=>{y=0,_=`votePassPhone`,R()})}function Y(){let e=v;D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;background:linear-gradient(135deg,${u.bg},${u.accent});">
      ${P()}
      <div style="font-size:14px;color:${u.yellow};margin-bottom:5px;">라운드 ${h}/${g}</div>
      ${L()}
      <div style="font-size:48px;margin-bottom:15px;">\uD83D\uDCAC</div>
      <h2 style="font-size:20px;font-weight:700;margin-bottom:20px;">설명하기</h2>
      <div style="background:rgba(0,0,0,0.3);padding:20px 24px;border-radius:16px;margin-bottom:20px;max-width:340px;">
        <p style="font-size:13px;color:rgba(255,255,255,0.7);margin-bottom:8px;">카테고리: ${e.category}</p>
        <p style="font-size:14px;color:rgba(255,255,255,0.8);line-height:1.6;">
          각자 받은 단어를 <b>단어를 직접 말하지 않고</b> 설명합니다.<br><br>
          한 명씩 돌아가며 자신의 단어에 대해<br>한 문장씩 설명해주세요.
        </p>
      </div>
      <div style="background:rgba(255,255,255,0.1);padding:14px 20px;border-radius:12px;margin-bottom:20px;">
        <p style="font-size:13px;color:${u.yellow};">진행 순서</p>
        <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;justify-content:center;">
          ${m.map((e,t)=>`<span style="background:${d[t]};padding:4px 12px;border-radius:20px;font-size:13px;font-weight:700;">${e.name}</span>`).join(``)}
        </div>
      </div>
      <p style="font-size:14px;color:${u.gray};margin-bottom:20px;">2~3바퀴 돌린 후 투표하세요!</p>
      <button id="btn-vote" style="background:linear-gradient(135deg,${u.primary},#c0392b);color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(233,69,96,0.4);">
        투표하기
      </button>
    </div>
  `,document.getElementById(`btn-vote`).addEventListener(`click`,()=>{y=0,_=`votePassPhone`,R()})}function X(){let e=v,t=``;e.type===`drawing`&&w.length>0&&(t=`
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;width:100%;max-width:360px;margin-bottom:20px;">
        ${w.map(e=>{let t=m[e.playerIndex],n=d[e.playerIndex%d.length];return`
            <div style="text-align:center;">
              <img src="${e.dataUrl}" style="width:100%;border-radius:10px;border:3px solid ${n};" />
              <p style="font-size:13px;font-weight:700;color:${n};margin-top:4px;">${t.name}</p>
            </div>
          `}).join(``)}
      </div>
    `),D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;height:100%;padding:20px;width:100%;overflow-y:auto;background:${u.bg};">
      ${P()}
      <div style="font-size:14px;color:${u.yellow};margin-bottom:5px;">라운드 ${h}/${g}</div>
      <h2 style="font-size:22px;font-weight:900;margin-bottom:5px;">\uD83D\uDD0D 토론 시간!</h2>
      <p style="font-size:13px;color:${u.gray};margin-bottom:15px;">누가 블러퍼인지 이야기해보세요</p>
      ${t}
      <button id="btn-vote" style="background:linear-gradient(135deg,${u.primary},#c0392b);color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(233,69,96,0.4);">
        투표하기
      </button>
    </div>
  `,document.getElementById(`btn-vote`).addEventListener(`click`,()=>{y=0,_=`votePassPhone`,R()})}function Z(){if(y>=m.length){_=`results`,R();return}let e=m[y],t=d[y%d.length];D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;background:${u.bg};">
      ${P()}
      <div style="font-size:48px;margin-bottom:20px;">\uD83D\uDDF3\uFE0F</div>
      <p style="font-size:16px;color:${u.gray};margin-bottom:10px;">투표할 차례입니다</p>
      <div style="background:${t};width:70px;height:70px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:30px;font-weight:900;margin:15px 0;">${y+1}</div>
      <p style="font-size:22px;font-weight:900;color:${t};margin-bottom:25px;">${e.name}</p>
      <p style="font-size:13px;color:${u.gray};margin-bottom:20px;">다른 사람이 보지 못하게 하세요!</p>
      <button id="btn-vote-ready" style="background:linear-gradient(135deg,${u.accent},${u.secondary});color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;">
        투표하기
      </button>
    </div>
  `,document.getElementById(`btn-vote-ready`).addEventListener(`click`,()=>{_=`voting`,R()})}function Q(){let e=m[y],t=d[y%d.length],n=m.map((e,t)=>({name:e.name,index:t})).filter(e=>e.index!==y);D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding:20px;text-align:center;width:100%;background:linear-gradient(135deg,${u.bg},${u.secondary});">
      ${P()}
      <p style="font-size:14px;color:${t};font-weight:700;margin-bottom:5px;">${e.name}의 투표</p>
      <h2 style="font-size:22px;font-weight:900;margin-bottom:20px;">누가 블러퍼일까요?</h2>
      <div style="display:flex;flex-direction:column;gap:12px;width:100%;max-width:300px;">
        ${n.map(e=>{let t=d[e.index%d.length],n=e.name.slice(0,1);return`
            <button class="vote-btn vote-btn-enhanced" data-idx="${e.index}" style="background:${u.darkGray};color:white;border:3px solid ${t};padding:14px 20px;border-radius:16px;font-size:16px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:12px;">
              <div style="width:40px;height:40px;border-radius:50%;background:${t};display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:900;flex-shrink:0;box-shadow:0 2px 8px ${t}44;">${n}</div>
              ${e.name}
            </button>
          `}).join(``)}
      </div>
    </div>
  `,document.querySelectorAll(`.vote-btn`).forEach(e=>{e.addEventListener(`click`,()=>{let t=parseInt(e.dataset.idx);e.classList.add(`vote-selected`),e.style.background=d[t%d.length]+`33`,m[y].votedFor=t,T[t]++,setTimeout(()=>{y++,_=`votePassPhone`,R()},400)})})}function $(){let e=m.findIndex(e=>e.isBluffer),t=m[e],n=d[e%d.length],r=v,i=Math.max(...T),a=T.indexOf(i)===e&&i>0;a?m.forEach((t,n)=>{n!==e&&(t.score+=10)}):t.score+=15,m.forEach(t=>{t.votedFor===e&&!t.isBluffer&&(t.score+=5)}),E.push({round:h,type:r.type,blufferIndex:e,caught:a});let o=m.map((t,n)=>{if(t.votedFor===-1)return``;let r=m[t.votedFor].name,i=t.votedFor===e;return`<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
      <span style="color:${d[n]};font-weight:700;">${t.name}</span>
      <span>${r} ${i?`<span style="color:#27ae60;">✓</span>`:`<span style="color:#e74c3c;">✗</span>`}</span>
    </div>`}).join(``);D.innerHTML=`
    <div style="display:flex;flex-direction:column;align-items:center;height:100%;padding:20px;width:100%;overflow-y:auto;background:linear-gradient(135deg,${u.bg},${a?`#1a4a2e`:`#4a1a1a`});position:relative;">
      ${a?I():``}
      ${P()}
      <div style="font-size:14px;color:${u.yellow};margin-bottom:8px;">라운드 ${h} 결과</div>
      <div class="winner-announce" style="font-size:56px;margin-bottom:10px;">${a?`🎉`:`😈`}</div>
      <h2 class="slide-up" style="font-size:24px;font-weight:900;margin-bottom:5px;color:${a?u.green:u.primary};">
        ${a?`블러퍼 적발!`:`블러퍼 승리!`}
      </h2>
      <p style="font-size:16px;margin-bottom:15px;">
        블러퍼는 <span style="color:${n};font-weight:900;">${t.name}</span> 이었습니다!
      </p>
      <div style="background:rgba(0,0,0,0.3);padding:14px 20px;border-radius:12px;margin-bottom:15px;max-width:300px;width:100%;">
        <p style="font-size:13px;color:${u.gray};margin-bottom:4px;">정답: <span style="color:${u.yellow};font-weight:700;">${r.normalPrompt}</span></p>
        <p style="font-size:13px;color:${u.gray};">블러퍼: <span style="color:${u.primary};font-weight:700;">${r.blufferPrompt}</span></p>
      </div>
      <div style="background:rgba(0,0,0,0.2);padding:14px 20px;border-radius:12px;margin-bottom:15px;max-width:300px;width:100%;">
        <p style="font-size:14px;font-weight:700;margin-bottom:8px;">투표 결과</p>
        ${o}
      </div>
      <div style="background:rgba(0,0,0,0.2);padding:14px 20px;border-radius:12px;margin-bottom:20px;max-width:300px;width:100%;">
        <p style="font-size:14px;font-weight:700;margin-bottom:8px;">현재 점수</p>
        ${m.map((e,t)=>`
          <div style="display:flex;justify-content:space-between;padding:4px 0;">
            <span style="color:${d[t]};font-weight:700;">${e.name}</span>
            <span style="font-weight:700;">${e.score}점</span>
          </div>
        `).join(``)}
      </div>
      <button id="btn-next-round" style="background:linear-gradient(135deg,${u.primary},#c0392b);color:white;border:none;padding:16px 50px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(233,69,96,0.4);">
        ${h<g?`다음 라운드`:`최종 결과 보기`}
      </button>
    </div>
  `,document.getElementById(`btn-next-round`).addEventListener(`click`,()=>{V()})}function ee(){let t=m.map((e,t)=>({...e,originalIndex:t})).sort((e,t)=>t.score-e.score),n=t[0],r=[`🥇`,`🥈`,`🥉`],i=[`파티 킹`,`명탐정`,`뛰어난 추리꾼`,`예리한 관찰자`,`참여상`,`참여상`,`참여상`,`참여상`],o=E.filter(e=>e.caught).length,s=E.filter(e=>!e.caught).length;if(t.forEach((e,t)=>{a(e.name,t===0)}),e)try{e.scores.submit({score:n.score,meta:{rounds:g,players:m.length,wins:o,blufferWins:s}})}catch{}D.innerHTML=`
    <div class="title-bg" style="display:flex;flex-direction:column;align-items:center;height:100%;padding:20px;width:100%;overflow-y:auto;position:relative;">
      ${I()}
      <div class="winner-announce" style="font-size:56px;margin:15px 0;">\uD83C\uDFC6</div>
      <h1 class="slide-up" style="font-size:28px;font-weight:900;margin-bottom:5px;background:linear-gradient(to right,${u.yellow},${u.orange});-webkit-background-clip:text;-webkit-text-fill-color:transparent;">최종 결과</h1>
      <p style="font-size:14px;color:${u.gray};margin-bottom:20px;">${g}라운드 완료!</p>

      <div style="background:rgba(0,0,0,0.3);padding:20px;border-radius:20px;margin-bottom:15px;max-width:340px;width:100%;text-align:center;">
        <p style="font-size:48px;margin-bottom:5px;">\uD83D\uDC51</p>
        <p style="font-size:22px;font-weight:900;color:${d[n.originalIndex]};">${n.name}</p>
        <p style="font-size:14px;color:${u.yellow};font-weight:700;">${i[0]} - ${n.score}점</p>
      </div>

      <div style="width:100%;max-width:340px;display:flex;flex-direction:column;gap:8px;margin-bottom:15px;">
        ${t.map((e,t)=>{let n=d[e.originalIndex%d.length];return`
            <div style="display:flex;align-items:center;gap:12px;background:rgba(0,0,0,0.2);padding:12px 16px;border-radius:12px;border-left:4px solid ${n};">
              <span style="font-size:20px;min-width:28px;">${t<3?r[t]:`${t+1}.`}</span>
              <span style="flex:1;font-weight:700;color:${n};">${e.name}</span>
              <span style="font-size:12px;color:${u.gray};">${i[Math.min(t,i.length-1)]}</span>
              <span style="font-weight:900;font-size:16px;">${e.score}</span>
            </div>
          `}).join(``)}
      </div>

      <div style="background:rgba(0,0,0,0.2);padding:14px 20px;border-radius:12px;margin-bottom:20px;max-width:340px;width:100%;">
        <p style="font-size:14px;font-weight:700;margin-bottom:8px;">\uD83D\uDCCA 게임 통계</p>
        <p style="font-size:13px;color:${u.gray};padding:3px 0;">총 라운드: ${g}</p>
        <p style="font-size:13px;color:${u.green};padding:3px 0;">블러퍼 적발: ${o}회</p>
        <p style="font-size:13px;color:${u.primary};padding:3px 0;">블러퍼 승리: ${s}회</p>
      </div>

      <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;">
        <button id="btn-again" style="background:linear-gradient(135deg,${u.primary},#c0392b);color:white;border:none;padding:16px 40px;border-radius:50px;font-size:16px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(233,69,96,0.4);">
          다시 하기
        </button>
        <button id="btn-home" style="background:rgba(255,255,255,0.1);color:white;border:2px solid rgba(255,255,255,0.3);padding:16px 40px;border-radius:50px;font-size:16px;font-weight:700;cursor:pointer;">
          처음으로
        </button>
      </div>
    </div>
  `,document.getElementById(`btn-again`).addEventListener(`click`,()=>{m.forEach(e=>{e.score=0,e.isBluffer=!1,e.votedFor=-1}),h=0,b=[],x=[],S=[],C=[],E=[],V()}),document.getElementById(`btn-home`).addEventListener(`click`,()=>{m=[],h=0,_=`title`,R()})}R()}))();