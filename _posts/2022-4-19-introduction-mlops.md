---
layout: post
title: "MLOps 소개 (작성중)"
categories: mlops
---

**MLOps (Machine Learning + Operation)** 를 한마디로 정의하면 *머신러닝 어플리케이션에 DevOps 원칙을 적용 한 것*이라고 할 수 있습니다. (by [MLOps Sig](https://github.com/cdfoundation/sig-mlops/blob/master/roadmap/2020/MLOpsRoadmap2020.md)) MLOps에서는 머신러닝 모델을 학습하고 추론하기 위한 워크플로우 및 시스템을 개발하고 운영합니다. 다른 의미로 MLOps는 *프로덕션 환경에서 머신러닝 모델을 안정적이고 효율적으로 배포 및 유지 관리하는 것을 목표로 한 일종의 관행*으로 볼 수 있습니다. (by [Breuel](https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f))

DevOps가 개발/QA/운영의 교차점이라면, MLOps는 머신러닝/데이터 엔지니어링/DevOps의 교차점 입니다. 이러한 관점에서 MLOps는 *머신러닝, DevOps 및 데이터 엔지니어링을 결합한 일련의 방식으로 프로덕션 환경에서 ML 시스템을 안정적이고 효율적으로 배포 및 유지 관리하는 것*이라고 할 수 있습니다.

![DevOpsMLOps](https://github.com/Taehun/taehun.github.io/blob/master/imgs/devops_mlops.png?raw=true)

- 그림 1. DevOps와 MLOps

사전적 정의를 떠나서 MLOps를 실제 프로젝트에 적용해야하는 팀이나 MLOps 엔지니어와 같은 담당자 관점에서 MLOps는 *End-to-End 머신러닝 워크플로우를 자동화 하는 것*이라고 할 수 있습니다.

## End-to-End 머신러닝 워크플로우

![ML Workflow](https://github.com/Taehun/taehun.github.io/blob/master/imgs/ml-workflow.png?raw=true)

- 그림 2. End-to-End 머신러닝 워크플로우

<그림 2>는 머신러닝 프로젝트의 End-to-End 워크플로우를 도식화한 것 입니다:

1. **데이터 추출**&nbsp;&nbsp;
   데이터 레이크와 같은 데이터 소스에서 관련된 데이터를 추출 합니다.
2. **데이터 탐색**&nbsp;&nbsp;
   데이터에 대한 이해를 위해 데이터 분석(EDA)을 수행합니다.
3. **데이터 가공**&nbsp;&nbsp;
   모델 학습에 필요한 데이터셋을 만들기 위해 특정 스키마로 데이터를 가공 합니다.
4. **데이터 검증**&nbsp;&nbsp;
   가공된 데이터에 유효하지 않는 데이터가 포함되어 있진 않은지 검증 합니다.
5. **모델 개발**&nbsp;&nbsp;
   머신 러닝 모델을 준비 합니다. 실제 모델 개발은 1~4 과정과 함께 이루어 집니다.
6. **모델 학습**&nbsp;&nbsp;
   준비된 데이터셋으로 모델을 학습 합니다.
7. **모델 평가**&nbsp;&nbsp;
   테스트 데이터셋에서 학습된 모델을 평가합니다.
8. **모델 배포**&nbsp;&nbsp;
   학습된 모델을 배포 포맷으로 변환하여 모델 레지스트리에 배포 합니다.
9. **모델 서빙**&nbsp;&nbsp;
   모델 레지스트리에 있는 모델을 로드하여 모델 추론 REST API 서비스를 배포 합니다.
10. **A/B 테스트**&nbsp;&nbsp;
    새로 배포된 모델 추론 서비스가 실제로 개선이 되었는지 판단하기 위해 A/B 테스트를 수행합니다.
11. **로깅**&nbsp;&nbsp;
    배포된 추론 서비스의 요청 및 응답 과정을 로깅 합니다.
12. **모니터링**&nbsp;&nbsp;
    로그를 바탕으로 모델의 추론 성능을 모니터링 합니다.

이 워크플로우는 머신러닝 모델을 사용하는 웹 앱이나 모바일 앱의 백엔드에 배포하여 사용하는 예시 입니다. (장치에 직접 모델을 배포하여 추론하는 *온 디바이스(On-Device) AI*는 모델 평가 단계부터 모니터링까지의 과정이 상이합니다. 다음에 기회가 있으면 다루어 보도록 하겠습니다.)

<그림 1>의 MLOps 벤 다이어그램에서 MLOps는 머신러닝, 데이터 엔지니어링, DevOps의 교차점이라고 하였습니다. <그림 2>의 머신러닝 워크플로우에서 초록색은 _데이터 엔지니어링_, 노란색은 _머신러닝_, 파란색은 *DevOps*에서 수행하는 단계들 입니다. *MLOps 엔지니어*가 End-to-End 머신러닝 워크플로우를 혼자서 모두 다 할 수도 있겠지만, 가장 현실성 있는 시나리오는 데이터 엔지니어링은 _데이터 엔지니어_, 머신러닝은 _데이터 과확자_, DevOps는 *DevOps 엔지니어*로 구성된 팀으로서 머신러닝 워크플로우를 수행하는 것 입니다.

앞서 MLOps를 실제 머신러닝 프로젝트에 적용하는 것은 End-to-End 머신러닝 워크플로우를 자동화 하는 것이라고 하였습니다. 각 분야별 담당자로 구성된 머신러닝 팀이라면 이런 목적을 달성하기 위해 모든 구성원이 MLOps를 이해하고 하나의 자동화된 End-to-End 머신러닝 시스템을 만들어 나가야 합니다. 머신러닝 워크플로우를 자동화하는 대표적인 구현체가 머신러닝 파이프라인(이하 ML 파이프라인) 입니다.

<img src="https://v0-6.kubeflow.org/docs/images/pipelines-xgboost-graph.png" alt="drawing" width="600"/>

- 그림 3. Kubeflow Pipeline

MLOps에는 ML 파이프라인을 포함하여 머신러닝 모델을 제품화하는데 필요한 다양한 구성 요소들이 있습니다.

## MLOps 구성 요소

<img src="https://mlops-for-all.github.io/images/docs/introduction/mlops-component.png" alt="drawing" width="600"/>

- 그림 4. MLOps 구성 요소

구글의 [Practitioners guide to MLOps: A framework for continuous delivery and automation of machine learning](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf) 문서에서 정의한 MLOps 구성 요소들 입니다.

### 실험(Experimentation)

데이터 과학자와 ML 연구원은 공동으로 데이터를 분석하고, 프로토타입 모델을 만들고 훈련을 구현할 수 있습니다. ML 환경은 버전 관리되는 모듈식, 재사용 가능 및 테스트 가능한 소스 코드를 작성할 수 있도록 해야 합니다. 실험의 주요 기능은 다음과 같습니다:

- Git과 같은 버전 관리 도구와 통합된 노트북 환경을 제공
- 데이터, 하이퍼파라미터 및 재현성 및 비교를 위한 평가 메트릭에 대한 정보를 포함한 실험 추적
- 데이터와 모델을 분석하고 시각화
- 데이터세트 탐색, 실험 검색 및 구현 검토를 지원
- 플랫폼의 다른 데이터 서비스 및 ML 서비스와 통합

### 데이터 처리(Data processing)

머신러닝 모델 개발, 지속적인 학습 파이프라인, 모델 서빙에서 대규모 ML용 데이터를 준비하고 변환할 수 있습니다. 데이터 처리의 주요 기능은 다음과 같습니다:

- 신속한 실험과 프로덕션에서 장시간 실행되는 작업을 위해 노트북과 같은 대화형 실행을 지원
- 다양한 데이터 구조 및 형식을 위한 데이터 인코더 및 디코더뿐만 아니라 광범위한 데이터 소스 및 서비스에 데이터 커넥터를 제공
- 정형(테이블) 및 비정형 데이터(텍스트, 이미지)에 대한 다양하고 효율적인 데이터 변환과 ML 피처 엔지니어링을 모두 제공
- ML 학습 및 서빙 워크로드를 위해 확장 가능한 배치 및 스트림 데이터 처리를 지원

### 모델 학습(Model training)

모델 학습은 학습에 사용되는 모델과 데이터 세트의 크기에 따라 확장할 수 있어야 합니다. 모델 학습의 주요 기능은 다음과 같습니다:

- 일반적인 ML 프레임워크를 지원하고 사용자 지정 런타임 환경을 지원
- 여러 GPU와 여러 작업자를 위한 다양한 전략으로 대규모 분산 학습을 지원
- ML 가속기를 온디맨드 방식으로 사용
- 효율적인 하이퍼 파라미터 튜닝 및 규모에 맞는 최적화를 지원
- 자동화된 모델 아키텍처 검색 및 자동화된 피처 엔지니어링을 포함한 AutoML 기능을 제공

### 모델 평가(Model evaluation)

실험 및 제품에서 자동으로 모델의 성능을 평가할 수 있습니다. 모델 평가의 주요 기능은 다음과 같습니다:

- 대규모 평가 데이터 세트에서 모델의 일괄 채점을 수행
- 여러 데이터 조각에서 모델에 대한 미리 정의된 평가 메트릭 또는 사용자 지정 평가 메트릭을 계산
- 다양한 지속적 학습 실행에 걸쳐 훈련된 모델 예측 성능을 추적
- 여러 모델의 성능을 시각화 및 비교
- What-if 분석 및 편향 및 공정성 문제를 식별할 수 있는 도구를 제공
- 다양한 설명 가능한(explainable) AI 기술을 사용하여 모델 행동 해석을 활성화

### 모델 서빙(Model serving)

프로덕션 환경에서 모델을 배포하고 서빙 할 수 있습니다. 모델 서빙의 주요 기능은 다음과 같습니다:

- 저 지연 실시간(온라인) 추론 및 높은 처리량 배치(오프라인) 추론 지원
- 일반적인 ML 서비스 프레임워크(예: Scikit-learn 및 XGBoost 모델의 경우 TensorFlow Serving, TorchServe, Nvidia Triton 등)와 사용자 지정 런타임 환경에 대한 내장 지원을 제공
- 복합 예측 루틴을 사용하도록 설정. 복합 예측 루틴에서는 여러 모델이 계층적으로 또는 동시에 호출된 후 결과를 집계
- 자동 확장 기능이 있는 ML 추론 가속기를 효율적으로 사용하여 급증하는 워크로드를 일치시키고 비용과 지연 시간의 균형을 맞춤
- 주어진 모델 추론에 대한 기능 속성과 같은 기술을 사용하여 모델 설명 가능성을 지원
- 분석을 위한 추론 서비스 요청 및 응답의 로깅을 지원

### 온라인 실험(Online experimentation)

### 모델 모니터링(Model monitoring)

### ML 파이프라인(ML pipelines)

### 모델 레지스트리(Model registry)

### 데이터셋과 피처 스토어(Dataset and feature repository)

### ML 메타데이터와 아티팩트 추적(ML metadata and artifact tracking)
