---
title: Review NCSOFT
date: 2024-06-30 12:00:00
description: 3년간의 NCSOFT에서의 경험을 정리
tag: review, NCSOFT, Graphics AI Lab, Game AI, Motion AI, Facial Motion Generation, Digital Human, Text-to-Texture Generation, NC Research
author: jiunbae
category: post
---

# NCSOFT에서의 3년간

## NCSOFT

2021년 2월, 전문연구요원으로 NCSOFT GameAI Lab에 입사했습니다. 컴퓨터 비전 연구실에서 Object Detection, Tracking관련 연구를 해 왔어서 GameAI Lab에서 Graphics 분야의 연구를 잘 진행할 수 있을 지 걱정이 되었지만, GameAI 랩에서는 독특한 데이터를 수집하고 처리할 수 있고 이를 통해 연구를 진행하면서 피드백을 받아 빠르게 수정해 나가는 방식으로 내가 늘 생각하던 "기술로 실제 문제를 해결" 하는 방향으로 기여할 수 있을 것 같았습니다. 도중에 Graphics AI Lab으로 이름이 바뀌고 연구하던 분야도 조금씩 바뀌었지만 3년간 NC Research에서 연구를 진행하면서 많은 것을 배우고 성장할 수 있었습니다. 결국 NCSOFT를 나와 새로운 시작을 하게 되었지만 지난 3년간의 경험이 많은 도움이 되었기 때문에 경험을 공유해보고자 합니다.

저는 게임을 좋아하며 간단한 게임을 만들어보기도 했고, 프로그래밍을 처음 시작하게 된 계기도 게임을 만들고 싶어서 였지만 전문연구요원을 편입을 준비하기 전까지만해도 게임회사에 들어갈 줄은 몰랐습니다. 회사를 탐색할 때 목표했던 것은 "기술(Machine Learning)로 실제 문제를 해결하는 일"이었습니다. 



## GameAI Lab, Motion AI 팀 그리고 Facial Motion Generation

<iframe width="560" height="315" src="https://www.youtube.com/embed/ahEZAJ-bxoI?si=UPj43xe8-pv_hK1K" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

2021년은 COVID-19의 해였습니다. 거대하고 강력한 바이러스가 세계를 휩쓸며 모두의 삶을 크게 변화시켰고, 거기에는 아주 작은 제 경험도 포함되어 있었습니다. 비대면 재택 근무가 활발하던 시기였기에 안내를 위한 버디 한 분을 제외하곤 비어있는 사무실로 첫 출근을 했었습니다.

저는 재택 근무가 생산성을 악화 시키지 않는다고 믿고있고, 효율적인 근무에 도움이 된다고 생각하는 사람이지만, 업무 전반에 대한 이해 와 동료와의 유대 관계가 형성되기 전부터 재택 근무로 일해야하는것은 생각보다도 더 적응하기 어려운 일이었습니다. 많은 것들을 새롭게 배워나가야 하는 입장에서 비대면은 조금 지치고 힘든 일이었습니다. 어떤 부분에선 지루하고 진전 없는 회의의 연속에 지치기도 했지만 새로운 실험과 연구들을 시간과 공간에 구애받지 않고 자유롭게 해볼 수 있는 점은 재택 근무의 장점이라고 생각했습니다.

**MotionAI Lab**은 *사람의 움직임을 그대로 디지털에서 재현하는 것을 목적으로 하는 조직*이었습니다. 여기에는 데이터를 수집하는 과정부터 모델을 설계하고 학습시키고 실제 서비스에 적용하는 과정까지 다양한 세부 조직과 목표들이 있었습니다. 연구실과 크게 달랐던 점은 데이터 수집과 전처리 부분이었습니다. 연구실에서는 정제된 챌린지 데이터셋에서 정해진 메트릭으로 좋은 결과를 얻어내는 실험들을 주로 진행했었습니다. 산학 협력 과제로 수집된 원시 데이터로부터 목적에 맞게 데이터 처리 방법을 고안하고 실험해 본 경험이 있었지만 비교적 잘 알려진 비전 도메인은 참고할 수 있는 자료가 많이 있었습니다. 그에 비해 MotionAI Lab에서는 고퀄리티 데이터를 위한 모션 캡쳐와 페이셜 캡쳐를 진행하고 있었고, 이렇게 수집된 데이터를 제대로 활용하기 위해서는 많은 전처리 과정을 통해 필요한 부분을 정제하는 과정에도 많은 심혈을 기울이고 있었습니다. 또한 기존에 연구하던 Image나 Text에 비해 Motion(Animation)은 사용할 수 있는 데이터가 매우 부족했고, Motion에 대한 평가 방법을 설계하기 매우 까다로워서 많은 고민을 했습니다.

제가 연구했던 부분은 Facial Animation Generation Model 이었습니다. 

## Graphics AI Lab에서의 Digital Human

<iframe width="560" height="315" src="https://www.youtube.com/embed/4mfESaPbLI4?si=OAIG5zqZmoihMYaX" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

https://about.ncsoft.com/news/article/techstandard-3-230404



## Texture Copilot Project

<iframe width="560" height="315" src="https://www.youtube.com/embed/6qGsSw7CffQ?si=5Myw9oyaiFxZJVrx" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Texture Copilot Project의 첫 시작은 2023년 하반기에 새로 입사한 인턴(이하 J)님이 시작한 연구였습니다. Texture Copilot Project는 3D 오브젝의 텍스쳐를 빠르게 프로토타이핑 하여 실제 아티스트의 파이프라인을 효율적으로 개선하는 목적을 가지고 있었습니다. Stable Diffusion 이후 2D 이미지 생성 모델의 성능이 크게 향상되면서 게임 개발 프로세스에서 많은 부분을 차지하는 그래픽 에셋 생성에 도움을 줄 수 있을것이라는 기대가 있었습니다. 저희는 여기서 더 나아가 3D 모델의 텍스쳐 생성을 개선해서 텍스트나 기존 이미지로부터 빠르고 쉽게 만들어 내고 의도대로 변형 할 수 있다면 주로 사용되고 있는 3D 모델링 파이프라인을 크게 개선할 수 있을 것이라고 생각했고, 더 빠른 프로토타이핑과 전체 게임 제작의 비용 감소로 이어질 수 있다고 생각했습니다.

refer: https://arxiv.org/abs/2302.01721

J는 이전에 Text-to-Image 연구를 진행하면서 Text-to-Texture 연구를 진행하고 싶다고 했고, 이에 대해 토론을 진행하게 되었습니다. Text-to-Texture 연구는 Text-to-Image 연구와는 다르게 텍스처를 생성하는 것이 목표였고, 이를 위해 Text-to-Image 연구에서 사용되는 모델을 활용하면서도 텍스처 생성에 특화된 모델을 설계해야 했습니다. 이에 대해 J와 함께 논의하고 실험을 진행하면서 점점 더 발전해 나가는 모습을 보여주었습니다.
