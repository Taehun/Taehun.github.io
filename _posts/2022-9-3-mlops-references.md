---
layout: post
title: 2022년 MLOps 추천 자료 Top 5
categories: mlops
tags: [mlops]
toc: true
comments: true
excerpt_separator: <!--more-->
---

> MLOps가 뭐지? -> [MLOps 소개](https://taehun.github.io/mlops/2022/04/19/introduction-mlops.html)

머신러닝 엔지니어로 커리어 전환하면서, 저의 업무 중 많은 비중을 차지하고 있는 것이 MLOps 입니다. 새로운 분야가 항상 그렇듯이, 보고 배울만한 자료가 그렇게 많지 않습니다. MLOps에 처음 발을 들이신 분들은 MLOps가 좋다고는 하는데, 어떤 자료를 참고하여 어떻게 도입 해야 할지 갈피를 잡기가 힘듭니다. 

그런 분들을 위해 제가 봤던 MLOps 자료 중에 좋았던 것 5개를 추려 보았습니다. 아래 순위는 매우 주간적이니, 순위에 연연치 마시고 아래 자료들은 모두 좋은 자료들이니 참고하여 MLOps 도입에 도움이 되었으면 합니다.

<!--more-->

## 1. Practitioners guide to MLOps

* [Practitioners guide to MLOps: A framework for continuous delivery and automation of machine learning](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf)
* 유형: 문서

2020년도에 처음 MLOps 업무를 맡았을때 Google Cloud의 _[MLOps: 머신러닝의 지속적 배포 및 자동화 파이프라인](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)_  기사에서 많은 도움을 받았습니다. 2021년도에 이 기사에서 한층 업그레이드된 _MLOps 실무자 가이드_ 문서가 출판되었습니다. MLOps는 빠르게 변하고 발전하고 있는 분야라 조금은 오래된 자료가 되었지만, 아직도 변함없이 좋은 자료임에는 틀림 없습니다.

## 2. Designing Machine Learning Systems

* [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969/ref=d_pd_sbs_sccl_3_1/132-3569047-6311147?pd_rd_w=0ibEX&content-id=amzn1.sym.3676f086-9496-4fd7-8490-77cf7f43f846&pf_rd_p=3676f086-9496-4fd7-8490-77cf7f43f846&pf_rd_r=RKX1J3G22AVS8K5NBNTY&pd_rd_wg=AlVlL&pd_rd_r=e753fdc1-2b22-419a-9fed-9a2d51218d95&pd_rd_i=1098107969&psc=1)
* 유형: 책

스탠퍼드 [CS329S 강의](https://stanford-cs329s.github.io) 내용을 책으로 엮은 것 입니다. 다양한 사례를 들면서 MLOps의 각 컴포넌트에 대해 설명하고 있습니다. CS329S 강의 자료와 함께 보시면 좀 더 많은 도움이 됩니다. MLOps 관련 툴과 데이터 엔지니어링에 대한 내용도 각각 한 챕터를 할당하여 정리되어 있습니다.

## 3. Full Stack Deep Learning
* [Full Stack Deep Learning - Course 2022](https://fullstackdeeplearning.com/course/2022/)
* 유형: 온라인 강의

딥러닝 모델 MLOps 시스템을 구축할때 도움이 되는 내용이 많은 온라인 강의 입니다. 빠르게 변하고 발전하는 MLOps 분야에 맞춰 강의 내용도 매년 갱신되고 있습니다. 딥러닝 모델 MLOps 시스템 구현시 고민이 많이 되는 GPU 리소스와 분산 학습에 대한 내용이 좋았습니다. 딥러닝에 포커싱 되어 있긴 하지만, 일반적인 머신러닝 모델 MLOps 구현에도 도움이 되는 내용도 많습니다.

2022년도 강의는 현재 진행 중 입니다. (2022.08 ~ 2022.10)

## 4. Made With ML
* [Made With ML](https://madewithml.com)
* 유형: 웹 사이트

새로운 프로그래밍 언어나 라이브러리 사용법을 익힐때는 그것을 사용하여 실제로 무언가를 만들어 보는 것 (즉, 실습)이 가장 효과적 입니다. 이 싸이트는 실습 위주로 구성된 MLOps 학습 싸이트 입니다. MLOps의 각 항목에 사용되는 다양한 툴들을 직접 실습해 볼 수 있도록 각종 예제들을 제공하고 있습니다. 

## 5. Neptune Blog
* [neptuneblog](https://neptune.ai/blog)
* 유형: 블로그

[Neptune.ai](https://neptune.ai)라는 MLOps 전문 업체의 기술 블로그 입니다. 이 블로그에 MLOps와 관련된 좋은 기사들이 많이 올라오고 있습니다. 다만, MLOps 툴과 관련된 내용에는 마케팅 차원에서 항상 Neptune을 포함하고 있으니 그 부분만 숙지하시고 보시기 바랍니다.

Neptune 블로그 기사 중에 내용이 너무 마음에 들어 제가 번역한 기사도 있습니다:
- [[번역] MLOps 아키텍처 가이드](https://taehun.github.io/mlops/2022/04/27/MLOps-architecture-guide.html)

## 기타

- [Awesome MLOps](https://github.com/visenger/awesome-mlops): MLOps 자료 링크 모음집
- [Coursera - MLOps 특화과정](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops): TFX 종속적인 내용만 아니었으면 추천 자료에 포함 되었을지도...
- [MLOps 도입 가이드](http://www.yes24.com/Product/Goods/109048610): 분량이 얼마안되 가볍게 보기에 좋습니다. 거버넌스에 대한 내용이 좋았습니다.
- [Valohai - Practical MLOPS](https://valohai.com/assets/files/practical-mlops-ebook.pdf): 이 자료도 처음 나왔을때는 MLOps 업무에 도움이 많이 되었습니다. 지금은 철지난 자료가 되었습니다.
- [머신러닝 디자인 패턴](http://www.yes24.com/Product/Goods/104426143): 패턴 좋아하시는 분들을 위해 ML 시스템을 30가지 패턴화해서 설명한 책 입니다. 제가 아직 읽어보지 않아서 추천해드리지 못합니다.

올해 (2022년)는 감사하게도 MLOps와 관련된 많은 자료들이 쏟아져 나왔습니다. 제가 미처 추천드리지 못한 MLOps와 관련된 좋은 자료가 있으면 덧글로 알려주세요.