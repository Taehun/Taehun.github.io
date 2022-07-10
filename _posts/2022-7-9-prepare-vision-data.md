---
layout: post
title: 비전 데이터셋 아키텍처
categories: mlops
tags: [Deep Learing, GCP, mlops, labeling]
toc: true
comments: true
excerpt_separator: <!--more-->
---

## 개요

실험실을 벗어나 상용 딥러닝 모델 개발 경험이 있는 분이라면 데이터셋을 준비하는 것이 매우 힘들다는 것에 공감하실 겁니다. 더구나 데이터셋 관리가 되지 않아 재현성을 포기한채 모델 개발하는 곳도 보았습니다. 이런 '해석 불가능한' 모델은 [모델 거버넌스](https://www.datarobot.com/blog/what-is-model-governance/)로 인해 힘들게 개발을 완료 하더라도 법적인 제약으로 상용화가 불가능 할 수도 있습니다.

이 기사에는 비전 데이터를 예시로 수집된 데이터에서 데이터셋 생성까지 과정을 자동화하는 샘플 아키텍처 제시하고, 비디오 데이터에서 이미지 데이터를 샘플링하는 간단한 파이프라인 예제를 작성하였습니다.

<!--more-->

비전 데이터셋은 데이터셋에 추가 할 이미지 샘플링을 어떻게 하느냐에 따라 딥러닝 프로세스 전체에 막대한 영향을 끼칩니다. 가장 좋은 방법은 SME (Subject Matter Expert)가 직접 데이터셋에 필요한 이미지 파일을 캡처해서 수집하는 것이지만, 이런 방식은 프로젝트에 따라 매우 힘들거나 너무 비효율적 일 수가 있습니다. 이 글에서 제시하는 아키텍처는 비디오 데이터를 수집 데이터로 하고, 데이터 라벨링이 필요한 이미지 데이터를 추출해서 데이터셋을 생성하는 것으로 가정 하였습니다.

AI 선도 업체들은 이미 *[반지도 학습 (Semi-supervised Learning)](https://en.wikipedia.org/wiki/Semi-supervised_learning), [능동적 학습 (Active Learning)](https://blogs.nvidia.co.kr/2020/01/29/what-is-active-learning/), [자동 라벨링 (Auto Labeling)](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-automated-labeling.html)*등의 현대적인 기법들을 적용하여 라벨링에 소요되는 비용을 최소화하고, 효율적으로 데이터셋 생성을 하고 있습니다. 이 글은 최신 라벨링 기법들 적용에 앞서 데이터셋 아키텍처를 어떻게 해야 할지 갈피를 잡지 못하는 분들을 위해 작성하였습니다.

## 비전 데이터셋 아키텍처 on GCP

![Data Labeling Process](https://github.com/Taehun/taehun.github.io/blob/master/imgs/labeling-process-gcp.png?raw=true)

_<center>딥러닝 비전 데이터셋 아키텍처 (GCP)</center>_

### Raw Data

> 솔루션: [Cloud Storage](https://cloud.google.com/storage)

녹화된 비디오 파일을 GCS에 업로드 합니다. 수집 단말에 녹화가 끝난후 비디오 파일이 생성되면 아래와 같은 코드로 자동 업로드 되도록 구현할 수 있습니다. (수집 단말에 인터넷 접속 및 [GCP 서비스 계정](https://cloud.google.com/iam/docs/service-accounts?hl=ko) 필요) 이 항목에서는 프로젝트에 따라 비디오 파일명 (경로 포함)을 어떻게 할지 표준을 정하는 것이 중요합니다. `video_id`와 같은 유니크한 키 값이 있다면 `video_id.mp4`처럼 파일명을 지정할 수 있습니다.

- GCS에 비디오 파일 업로드 Python 예제

```python
from google.cloud import storage
import os

# video_file_name = "<SOME_VIDEO_FILE>.mp4"

raw_data_bucket = os.environ["RAW_DATA_BUCKET"]
collector_id = os.environ["COLLECTOR_ID"]

storage_client = storage.Client()
bucket = storage_client.bucket(raw_data_bucket)
blob = bucket.blob(f"{collector_id}/{video_file_name}")

blob.upload_from_filename(video_file_name)
```

### Trigger

> 솔루션: [Pub/Sub](https://cloud.google.com/pubsub)

GCS 버킷에 비디오 파일 및 이미지 파일 업로드 이벤트를 가져오기 위해 Pub/Sub 토픽을 생성합니다. 버킷에 `notification.create()` 함수로 알림을 구성 할 수 있습니다. 사용하는 [GCS 이벤트 유형](https://cloud.google.com/storage/docs/pubsub-notifications?hl=ko#events)은 `OBJECT_FINALIZE` 입니다.

- 버킷 생성 및 이벤트 알림 생성 예제

```python
from google.cloud import storage

def create_bucket_notifications(bucket_name, topic_name):
    """Creates a notification configuration for a bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The name of a topic
    # topic_name = "your-topic-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    notification = bucket.notification(topic_name=topic_name)
    notification.create()

    print(f"Successfully created notification with ID {notification.notification_id} for bucket {bucket_name}")
```

좀 더 자세한 내용은 [Cloud Storage용 Cloud Pub/Sub 알림](https://cloud.google.com/storage/docs/pubsub-notifications?hl=ko), [Cloud Storage용 Pub/Sub 알림 구성](https://cloud.google.com/storage/docs/reporting-changes?hl=ko#storage_create_bucket_notifications-python), [Notification Polling 예제](https://github.com/googleapis/python-storage/blob/main/samples/snippets/notification_polling.py)를 참고하세요.

### Extracting

> 솔루션: [Dataflow](https://cloud.google.com/dataflow?hl=ko)

이 아키텍처에서 핵심이 되는 부분 입니다. 수집한 비디오 파일에서 데이터셋으로 사용할 이미지를 샘플링하는 ETL에 해당 됩니다. _능동적 학습_ 알고리즘을 여기에 추가 하시면 됩니다. 먼저, 가장 단순하게 **1초마다 1 프레임을 추출**하는 예제로 ETL을 만들어 보겠습니다.

가장 먼저 비디오 파일에서 프레임을 추출하려면 파이썬 기준 [cv2 패키지](https://pypi.org/project/opencv-python/)가 필요합니다. Dataflow 워커에서 `cv2` 패키지 사용하기 위해 여러 방법이 있지만, 저는 [커스텀 컨테이너](https://cloud.google.com/dataflow/docs/guides/using-custom-containers?hl=ko#python_6)를 사용하였습니다.

- Dockerfile 샘플

```
FROM apache/beam_python3.8_sdk:2.40.0
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python google-cloud-storage google-cloud-pubsub
```

Pub/Sub 토픽에서 비디오 파일명을 읽어와서 프레임을 추출해야 합니다. `cv2` 패키지를 사용하여 비디오 파일의 Frame rate 설정에 따라 1초마다 1 프레임씩 이미지 파일을 추출합니다.

```python
class ExtractFrames(beam.DoFn):
    def process(self, element):
        import math
        import os
        from pathlib import Path

        import cv2

        logging.info(f"ExtractFrames: {element}")

        videoFile = element
        video_id, _ = os.path.splitext(videoFile)
        imagesFolder = f"/tmp/{video_id}"
        os.makedirs(imagesFolder, exist_ok=True)
        cap = cv2.VideoCapture(videoFile)
        frameRate = cap.get(5)  # frame rate
        while cap.isOpened():
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if ret != True:
                logging.warning("ExtractFrames: Failed to extract frame!")
                break
            if frameId % math.floor(frameRate) == 0:
                filename = f"{imagesFolder}/{str(int(frameId))}.jpg"
                cv2.imwrite(filename, frame)
        cap.release()

        for f in Path(imagesFolder).rglob("*.jpg"):
            yield str(f)
```

추출한 이미지 파일 경로를 입력 받아 이미지 버킷 (`Unlabeled Data`)에 업로드 합니다.

```python
def UploadFrames(element):
    from google.cloud import storage

    image_data_bucket = "IMAGE_BUCKET_NAME"

    local_prefix = len("/tmp/")
    blob_name = element[local_prefix:]

    storage_client = storage.Client()
    bucket = storage_client.bucket(image_data_bucket)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(element)
    blob_path = f"gs://{image_data_bucket}/{blob_name}"
    logging.info(f"UploadFrames: {blob_path}")
```

마지막으로 파이프라인 옵션을 설정하고, 파이프라인 각 단계를 설정합니다. 파이프라인 옵션에 `experiments`와 `sdk_container_image` 옵션은 커스텀 컨테이너 이미지를 사용하기 위해 추가하는 옵션 입니다.

```python
pipeline_options = PipelineOptions(experiments="use_runner_v2",
                                    sdk_container_image="YOUR_CUSTOM_IMAGE_URI",
                                    runner='DataflowRunner',
                                    project='YOUR_PROJECT_ID',
                                    job_name='extract-video-frame-' + str(int(time.time())),
                                    temp_location='gs://TEMP_BUCKET/temp',
                                    region='REGION')

with beam.Pipeline(options=pipeline_options) as pipeline:
    result = (
        pipeline
        | "Read from Pub/Sub" >> io.ReadFromPubSub(topic=input_topic)
        | "Get video files" >> GetVideoFilesFromPubsub()
        | "Extract video frames" >> beam.ParDo(ExtractFrames())
        | "Upload frames" >> beam.Map(UploadFrames)
    )
```

### Labeling Task

> 솔루션: [Cloud Functions](https://cloud.google.com/functions?hl=ko)

Unlabeled Data 버킷에 이미지 파일이 업로드되면 Pub/Sub은 Labeling Task 함수를 트리거 합니다. 이 함수는 라벨링 업체에 업로드된 이미지 파일의 라벨링 요청을 보내는 역활을 수행 합니다. 이 항목은 라벨링 업체마다 모두 상이하므로 계약한 업체의 API에 맞게 작성하시기 바랍니다.

- [Scale AI](https://scale.com/)의 작업 요청 API 예제

```python
def request_labeling(event, context):
    import base64
    import requests
    from scaleapi.tasks import TaskType
    from scaleapi.exceptions import ScaleDuplicateResource

    print("""This Function was triggered by messageId {} published at {} to {}
    """.format(context.event_id, context.timestamp, context.resource["name"]))

    payload = dict(
        project = "PROJECT_NAME",
        callback_url = "http://www.example.com/callback",
        instruction = "Draw a box around each cars.",
        attachment_type = "image",
        attachment = "SAMPLE_IMAGE_URL",
        unique_id = "c235d023af73",
        geometries = {
            "box": {
                "objects_to_annotate": ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]
                "min_height": 10,
                "min_width": 10,
            }
        },
    )

    try:
        client.create_task(TaskType.ImageAnnotation, **payload)
    except ScaleDuplicateResource as err:
        print(err.message)  # If unique_id is already used for a different task
```

### Labeling Data

> 솔루션: [Cloud Firestore](https://firebase.google.com/docs/firestore)

라벨링 작업이 끝나면 업체에서 어노테이션 데이터를 전달해 줍니다. 업체에서 제공하는 API가 모두 다르므로 어노테이션 데이터를 받는 과정도 모두 상이합니다. 여기서는 가장 범용으로 사용할 수 있는 NoSQL 저장소인 Firestore로 어노케이션 데이터를 수신하도록 구성하였습니다. (라벨링 업체측에 Cloud Firestore에 접근이 가능한 [서비스 계정](https://cloud.google.com/iam/docs/service-accounts)이 있어야 합니다.)

### Data Warehouse

> 솔루션: [BigQuery](https://cloud.google.com/bigquery?hl=ko)

Firestore에 저장된 어노테이션 데이터를 BigQuery로 추출하여, 데이터 과학자들께 데이터 분석 환경을 제공합니다. BigQuery를 오프라인 피처 스토어로 설정하여 이미지 데이터와 함께 딥러닝 모델 학습에 사용 할 수 있습니다.

## 비전 데이터셋 아키텍처 고도화

GCP에서 가장 기본적인 비전 데이터셋 아키텍처를 구성해 보았습니다. 이렇게 아키텍처를 구성하면 크게 두가지 문제에 직면하게 됩니다.

### 클라우드 비용 문제

비디오 데이터는 저장 장치 용량을 많이 차지하므로 수집되는 데이터가 많아질수록 클라우드 사용료가 큰 폭으로 증가하게 됩니다. 이런 문제를 해결하기 위해 아래와 같이 추출한 이미지 파일만 클라우드에 업로드하는 하이브리드 클라우드 방식으로도 아키텍처링 할 수 있습니다. 기존 GCP의 서비스와 동일한 동작을 하는 On-premise용 솔루션을 활용하여 클라우드 사용료를 절감 할 수 있습니다. On-premise로 일부 서비스가 이전됨에 따라 그만큼 인프라 관리 비용이 증가하는 점을 잊지마시기 바랍니다. (`클라우드 사용료` > `인프라 관리 비용`?)

- `Cloud Storage` -> `MinIO`
- `Cloud Pub/Sub` -> `RabbitMQ`
- `Cloud Dataflow` -> `Apache Beam`

![Mixed Data Labeling Process](https://github.com/Taehun/taehun.github.io/blob/master/imgs/labeling-process-mixed.png?raw=true)

_<center>딥러닝 비전 데이터셋 아키텍처 (On-premise + GCP)</center>_

### 유사한 이미지 제거

자율주행 데이터셋을 수집하는 시나리오를 생각해보면, 1초마다 한 프레임을 추출하는 것은 자동차가 정지된 구간에는 유사한 이미지를 여러장 추출하여 라벨링 비용만 낭비할 가능성이 매우 큽니다. _능동적 학습_ 알고리즘을 적용하기 앞서 *이미지 유사도*를 검사하여 유사도가 높은 이미지는 샘플링 하지 않는 간단한 알고리즘으로도 샘플링 이미지 개수를 줄일수 있을 것 입니다.

파이썬에는 `image-similarity-measures` 패키지로 두 이미지의 유사도를 측정 할 수 있습니다. 아래는 두 이미지간의 유사도를 측정하는 파이썬 샘플 코드 입니다.

```
$ pip3 install image-similarity-measures
```

```python
import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import ssim

origin_img = argv[1]
second_img = argv[2]

test_img = cv2.imread(origin_img)

ssim_measures = {}

scale_percent = 100 # percent of original img size
width = int(test_img.shape[1] * scale_percent / 100)
height = int(test_img.shape[0] * scale_percent / 100)
dim = (width, height)

data_img = cv2.imread(second_img)
resized_img = cv2.resize(data_img, dim, interpolation = cv2.INTER_AREA)
ssim_measures = ssim(test_img, resized_img)

print(f"Image similaraty is {ssim_measures}")
```

> $ python similaraty.py /tmp/sample-15s/0.jpg /tmp/sample-15s/406.jpg<br/>
> Image similaraty is 0.7200022681160081
