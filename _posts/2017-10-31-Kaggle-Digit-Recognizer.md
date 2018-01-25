---
title: Kaggle로 시작하는 데이터 사이언스 Digit Recognizer
date: 2017-10-31 16:05:00
description: Data Science starting with Kaggle-Digit Recognizer
tag: DataScience, Kaggle
author: MaybeS
category: tech
---

# Kaggle로 시작하는 데이터 사이언스: Digit Recognizer

머신러닝에 관련해서 가장 유명한 "Hello, World!"같은 예제는 아마 [MNIST](https://en.wikipedia.org/wiki/MNIST_database)일 것 입니다. Kaggle에도 MNIST 데이터를 사용한 [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)가 존재합니다.



`Overview` 에는 간략한 설명이 적혀있습니다. `Data`탭에는  train, test, sample_submission이 들어있습니다. 보통 `csv`확장자 형태로 주어지며, Description에 각각 무엇을 나타내는지 설명이 있습니다. train에는 라벨과 28x28의 1채널 이미지가 42000개 들어있습니다. test에는 라벨을 제외하고 1채널 이미지 28000개가 들어있습니다. `sample_submission.csv`를 확인해 보면 28000개의 test 데이터의 라벨을 만들어 내는 것 임을 알 수 있습니다.



## 환경 준비하기

저는 [python](https://www.python.org/)을 가장 좋아하며, 데이터 사이언스를 하기에 특히 더 좋은 언어라고 생각합니다. 특히 [Jupyter](http://jupyter.org/)와 함께 사용하면 쉽고 빠르게 코드를 작성하고 실행해 볼 수 있습니다. Jupyter의 설치를 완료하고 실행하면 다음과 같은 화면을 볼 수 있습니다.

![Jupyter](https://drive.google.com/uc?id=0BwQhFb-IfuTFMnZISmF0TGp1VkE)



파이썬에는 수 많은 패키지들이 있으며 그 중에서는 데이터 사이언스에 특화된 좋은 패키지들도 많이 있습니다. 아래의 패키지들은 매우 자주 사용되고, 사용하기 편합니다.

- [Numpy](http://www.numpy.org/): 두말 할 나위 없는 최고의 파이썬 라이브러리입니다.
- [Pandas](http://pandas.pydata.org/): 데이터를 쉽게 가공할 수 있습니다.
- [Matplotlib](https://matplotlib.org/): 시각화를 위한 라이브러리입니다.




## 데이터 확인하기

먼저 주어진 `train.csv`, `test.csv`에 어떠한 내용이 있는지 확인해 봅시다. 아래의 화면은 주어진 데이터들을 불러온 화면입니다. (pd는 pandas입니다.)

![Load data](https://drive.google.com/uc?id=0BwQhFb-IfuTFNW5ONnI2V0FIU2s)

데이터 설명에서도 확인할 수 있지만, 28x28이미지가 784개의 column으로 나타내어져있고, 42000개의 이미지가 `train.csv`에, 28000개가 `test.csv`에 있는것을 확인할 수 있습니다. 아래의 코드는 33번째 이미지가 9이며 아래와 같게 나타나는 것을 볼 수 있습니다.

![Data visualization](https://drive.google.com/uc?id=0BwQhFb-IfuTFOEcxUHg0TXBJX2s)

그럼 먼저 샘플 제출을 확인해 봅시다. `sample_submission.csv`파일을 보면 모든 이미지를 0으로 만들어 둔 것을 확인할 수 있는데요, 실제로 제출해보면 0.10014점으로 테스트이미지에 10%정도 맞춘것을 확인할 수 있습니다. (테스트 데이터에 0이 10%정도 있어서 그렇습니다.) 그러면 모든 값을 1로 바꾼후 제출해 봅시다. 아래의 코드는 `sample_submission.csv`를 읽어온 후 라벨을 모두 1로 변경시키고 `first_submission.csv`로 저장합니다. 아마 0.11614점을 얻을 수 있을 겁니다.

![first submission](https://drive.google.com/uc?id=0BwQhFb-IfuTFUTVsZVNxcWRZLW8)



## 간단한 뉴럴네트워크

[Tensorflow Tutorial](https://www.tensorflow.org/get_started/mnist/beginners)에서는 간단한 Softmax Regression과 그 구현방법을 설명하고 있습니다. 샘플 코드도 쉽게 구할 수 있으므로 아래의 코드와 같이 실행해 보는 것은 어려운 일이 아닐 것입니다.

![train](https://drive.google.com/uc?id=0BwQhFb-IfuTFaWdSMnZQY0FmWnc)

softmax를 사용해서 간단한 머신러닝을 돌려보았습니다. 이제 잘 학습되었는지 검증해 보아야 합니다.



아래의 **Evaluate**라벨 아래의 코드는 테스트 데이터에서 라벨을 추측하는 과정입니다. 그 밑의 **test with train data**는 train데이터를 다시 입력해서 얼마나 잘 학습되었는지 개략적으로 확인하는 과정입니다. 86%의 정확도를 보여주고 있습니다. 물론 학습에 사용했던 데이터로 검증을 하는 것은 **매우 좋지 않은**방법입니다. 될수 있다면 학습 데이터와 검증 데이터를 미리 나눠 사용하도록 하세요. 아래에서 생성한 `predicted_labels`에는 테스트 데이터 28000개에 대한 추측값이 들어가 있습니다. 우리는 학습시킬때 y값으로 0~9에 해당하는지 아닌지를 [one hot vector](https://en.wikipedia.org/wiki/One-hot)로 집어넣었기 때문에 결과로는 0일 확률 ~%, 1일 확률 ~%, ... 9일 확률 ~%가 나오게 됩니다. 이 중 가장 큰 값을 `tf.argmax`를 사용해서 가져오는 과정이 적용되어 결국엔 이 이미지가 어떤 숫자인지 추측하게 됩니다.

![eval](https://drive.google.com/uc?id=0BwQhFb-IfuTFelFINDYwTW1vZms)

여기 까지 진행했다면 학습된 모델로 테스트 데이터를 추측한 값을 제출해볼 차례입니다. `predicted_labels`를 값으로 가지는 dataframe을 새로 만들어서 `sample_submission.csv`를 참조해 submission을 만들 수 있습니다.

![make submission](https://drive.google.com/uc?id=0BwQhFb-IfuTFRENJc1ZOMXhJX2M)

제출을 통해 0.92214로 간단한 softmax regression도 매우 높은 결과를 낼 수 있음을 알 수 있습니다. 이제 여기에 여러 다른 방법을 구현함으로써 더  높은 결과를 얻어낼 수 있습니다.

![submission](https://drive.google.com/uc?id=0BwQhFb-IfuTFUlZzcEVVT2dyV0k)
