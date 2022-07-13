---
layout: post
title: Flax/JAX로 시작하는 딥러닝
categories: deep_learning
tags: [Deep Learing, Flax, JAX]
toc: true
comments: true
excerpt_separator: <!--more-->
---

> ~~_"플렉스 해버렸지 뭐야"_~~

## 개요

[JAX](https://github.com/google/jax)가 출시된지 2년이 지나면서 완성도가 많이 높아졌습니다. 이제는 실험 단계를 넘어 상용 모델 개발에 JAX 도입을 검토해 볼만한 단계가 온 것 같습니다. DeepMind나 Hugging Face와 같은 AI 업계를 선도하는 회사들은 이미 JAX로 이전했거나 이전하고 있는 과정에 있기도 하구요. ([참고링크1](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research), [참고링크2](https://twitter.com/huggingface/status/1331255460033400834))

이 기사에는 [Flax](https://github.com/google/flax)라는 JAX용 딥러닝 라이브러리를 사용하여, 간단한 딥러닝 모델을 만드는 방법을 다룹니다. Flax의 API는 딥러닝 개발자 경험 (DX, Developer Experience) 고려하여 개발 되었기에 기존 Tensorflow나 PyTorch 경험이 있는 분이면 쉽게 익힐 수 있습니다.

<!--more-->

### JAX

[JAX](https://github.com/google/jax)는 자동 미분 기능 (AutoGrad)를 가지고 있는 CPU/GPU/TPU에서 동작하는 NumPy 입니다.

기존 NumPy API(+필요시 특수한 가속기 작업을 위한 추가적인 API)를 사용하여 빠른 과학적 계산 및 머신 러닝을 제공합니다.

JAX는 다음과 같은 강력한 기본 요소들이 제공됩니다:

- **Autodiff (`jax.grad`)**: 모든 변수에 대한 효율적인 임의의 차수 그레이디언트
- **[JIT (Just-In-Time) 컴파일](https://ko.wikipedia.org/wiki/JIT_%EC%BB%B4%ED%8C%8C%EC%9D%BC) (`jax.jit`)**: 모든 기능 추적 → 퓨전된 가속기 ops
- **벡터화 (`jax.vmap`)**: 개별 샘플에 대한 자동 배치 코드 작성
- **병렬화 (`jax.pmap`)**: 여러 가속기(예: TPU 포드를 위한 호스트들 포함) 간에 자동으로 코드 병렬화

### Flax

[Flax](https://github.com/google/flax)는 유연성을 위해 설계된 JAX를 위한 딥러닝 라이브러리 입니다. Flax는 프레임워크에 기능을 추가하는 것이 아니라 예제와 학습 루프를 수정하여 새로운 형태의 모델 학습을 시도합니다.

Flax는 JAX 팀과 긴밀히 협력하여 개발 중이며 다음과 같은 딥러닝 연구를 시작하는 데 필요한 모든 것이 제공됩니다:

- **신경망 API (`flax.linen`)**: Dense, Conv, Batch/Layer/Group 정규화, Attention, Pooling, LSTM/GRU 셀, Dropout
- **유틸리티 및 패턴**: 복제된 학습, 직렬화 및 체크포인트, 메트릭, 장치에서 사전 검색
- 즉시 사용할 수 있는 **학습 예제**: MNIST, LSTM seq2seq, GNN (Graph Neural Networks), 시퀀스 태깅
- **빠르고 튜닝된 대규모 종단간 예제**: CIFAR10, ImageNet의 ResNet, Transformer LM1b

### Flax를 배워야 하는 이유

> 2022-7-13일에 추가한 내용 입니다.

이미 PyTorch나 Tensorflow로 딥러닝 연구 & 개발을 잘하고 있는데 왜 Flax를 배워야 할까요?

- Flax의 모델 정의는 기존 딥러닝 프레임웍과 거의 차이가 없습니다.
- Flax는 설계상 매우 유연하고 확장 가능합니다.
- **코드 변경 없이 [TPU](https://cloud.google.com/tpu?hl=ko) 학습이 가능합니다.**
- GPU 학습시 입력 데이터가 크면 PyTorch에 비해 학습 시간이 빠릅니다. ([참고링크](https://github.com/google/jax/discussions/8497#discussioncomment-1626017))

현재까지는 Flax에는 데이터 로드 및 처리 기능이 없어서 PyTorch의 [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)나 [transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html) 같은 것을 JAX와 조합하여 직접 구현해야하는 불편함이 있습니다. 하지만, 위와 같은 장점들이 있으므로 시간을 들여 한번쯤 학습을 하는건 나쁘지 않다고 생각합니다. 특히, 자연어 처리를 위해 Transformer 계열 모델을 실험 하시는 분께 추천 드립니다.

- Flax Transformer 예제: [Fine-tuning a 🤗 Transformers model on TPU with Flax/JAX](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb)

## 의류 이미지 분류 예제

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kUtM8o62QPP0BkzXHPmHmXgsdPOmBynd?usp)

이 장에서는 Tensorflow 공식 문서에 있는 [기본 분류: 의류 이미지 분류](https://www.tensorflow.org/tutorials/keras/classification?hl=ko) 예제를 Flax로 구현해보겠습니다.

먼저, Flax 패키지를 설치하고 필요한 패키지를 import 합니다.

```
!pip install -q flax
```

```python
import jax
import jax.numpy as jnp                # JAX NumPy
import numpy as np                     # 기존 NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # 학습 상태 보존에 유용한 데이터 클래스

import optax                           # Optimizers
import tensorflow_datasets as tfds     # Fashion MNIST Dataset 가져오기 위한 TFDS 패키지
from matplotlib import pyplot as plt   # 시각화
```

### 데이터셋 가져오기

10개의 범주(category)와 70,000개의 흑백 이미지로 구성된 [패션 MNIST 데이터셋](https://github.com/zalandoresearch/fashion-mnist)을 사용하겠습니다. 이미지는 해상도(28x28 픽셀)가 낮고 다음처럼 개별 옷 품목을 나타냅니다:

![Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

- 그림 1. 패션-MNIST 샘플

패션 MNIST는 컴퓨터 비전 분야의 "Hello, World" 프로그램격인 고전 MNIST 데이터셋을 대신해서 자주 사용됩니다. MNIST 데이터셋은 손글씨 숫자(0, 1, 2 등)의 이미지로 이루어져 있습니다. 여기서 사용하려는 옷 이미지와 동일한 포맷입니다.

패션 MNIST는 일반적인 MNIST 보다 조금 더 어려운 문제이고 다양한 예제를 만들기 위해 선택했습니다. 두 데이터셋은 비교적 작기 때문에 알고리즘의 작동 여부를 확인하기 위해 사용되곤 합니다. 코드를 테스트하고 디버깅하는 용도로 좋습니다.

여기에서 60,000개의 이미지를 사용하여 네트워크를 훈련하고 10,000개의 이미지를 사용하여 네트워크에서 이미지 분류를 학습한 정도를 평가합니다. Fashion MNIST 데이터셋은 TFDS (Tensorflow Dataset) 패키지에 포함되어 있습니다. TFDS에서 Fashion MNIST 데이터를 가져오고 로드합니다.

```python
ds_builder = tfds.builder('fashion_mnist')
ds_builder.download_and_prepare()
train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
train_ds['image'] = jnp.float32(train_ds['image']) / 255.
test_ds['image'] = jnp.float32(test_ds['image']) / 255.
```

TFDS Fashion MNIST 데이터세트를 `train`과 `test` 데이터세트로 분리하였습니다. 각 데이터세트는 `as_numpy()` 함수를 사용하여 Numpy 타입으로 변환하였습니다.

이미지 데이터는 픽셀 값의 범위가 0\~255 사이이므로 0\~1 사이의 부동 소수점 타입으로 데이터를 변환하였습니다.

<table>
  <tr>
    <th>레이블</th>
    <th>클래스</th>
  </tr>
  <tr>
    <td>0</td>
    <td>T-shirt/top</td>
  </tr>
  <tr>
    <td>1</td>
    <td>Trouser</td>
  </tr>
    <tr>
    <td>2</td>
    <td>Pullover</td>
  </tr>
    <tr>
    <td>3</td>
    <td>Dress</td>
  </tr>
    <tr>
    <td>4</td>
    <td>Coat</td>
  </tr>
    <tr>
    <td>5</td>
    <td>Sandal</td>
  </tr>
    <tr>
    <td>6</td>
    <td>Shirt</td>
  </tr>
    <tr>
    <td>7</td>
    <td>Sneaker</td>
  </tr>
    <tr>
    <td>8</td>
    <td>Bag</td>
  </tr>
    <tr>
    <td>9</td>
    <td>Ankle boot</td>
  </tr>
</table>

각 이미지는 하나의 레이블에 매핑되어 있습니다. 데이터셋에 *클래스 이름*이 들어있지 않기 때문에 나중에 이미지를 출력할 때 사용하기 위해 별도의 변수를 만들어 저장합니다:

```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

### 데이터 탐색

모델을 훈련하기 전에 데이터셋 구조를 살펴보죠. 다음 코드는 훈련 세트에 60,000개의 이미지가 있다는 것을 보여줍니다. 각 이미지는 28x28 픽셀로 표현됩니다:

```python
train_ds['image'].shape
```

> (60000, 28, 28, 1)

비슷하게 훈련 세트에는 60,000개의 레이블이 있습니다:

```python
len(train_ds['label'])
```

> 60000

각 레이블은 0과 9사이의 정수입니다:

```python
train_ds['label']
```

> array([2, 1, 8, ..., 6, 9, 9])

테스트 세트에는 10,000개의 이미지가 있습니다. 이 이미지도 28x28 픽셀로 표현됩니다:

```python
test_ds['image'].shape
```

> (10000, 28, 28, 1)

테스트 세트는 10,000개의 이미지에 대한 레이블을 가지고 있습니다:

```python
len(test_ds['label'])
```

> 10000

훈련 세트에서 처음 25개 이미지와 그 아래 클래스 이름을 출력해 보죠. 데이터 포맷이 올바른지 확인하고 네트워크 구성과 훈련할 준비를 마칩니다.

```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(jnp.squeeze(train_ds['image'])[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_ds['label'][i]])
plt.show()
```

![Train data](https://github.com/Taehun/taehun.github.io/blob/main/imgs/train_result.jpg?raw=true)

### 네트워크 정의

Flax Linen API의 Module의 서브 클래스로 CNN 네트워크를 정의 합니다. 이 예제의 모델 아키텍처는 비교적 단순하기 때문에 `__call__` 메서드 내에서 직접 인라인 서브 모듈을 정의하고 `@compact` 데코레이터로 래핑할 수 있습니다.

```python
class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x
```

### 손실 함수 정의

간단하게 `optax` 패키지의 `softmax_cross_entropy()`를 사용합니다. 이 함수는 _`[batch, num_classes]`_ shape을 가진 `logits`과 `labels` 파라메터를 받습니다. `labels`는 TFDS에서 정수 값으로 읽히므로 먼저 One-Hot 인코딩으로 변환해야 합니다.

```python
def cross_entropy_loss(*, logits, labels):
  labels_onehot = jax.nn.one_hot(labels, num_classes=10) # Ont-Hot 인코딩으로 [batch, num_classes] shape으로 변환
  return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
```

### 매트릭 계산

손실 (loss) 및 정확도 (accuracy) 매트릭 계산 함수를 정의합니다.

```python
def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics
```

### 학습 상태 생성

Flax의 일반적인 패턴은 step 번호, 파라메터 및 옵티마이저 상태를 포함하여 전체 학습 상태를 나타내는 단일 데이터 클래스를 만드는 것입니다.

또한 옵티마이저와 모델을 이 상태에 추가하면 `train_step()`과 같은 함수로 단일 인수만 전달하면 된다는 장점이 있습니다 (아래 참조).

이것은 매우 일반적인 패턴이기 때문에 Flax는 대부분의 기본 사용 사례를 제공하는 [flax.training.train_state.TrainState](https://flax.readthedocs.io/en/latest/flax.training.html#train-state) 클래스를 제공합니다. 일반적으로 추적할 데이터를 더 추가하기 위해 서브 클래스를 사용하지만, 이 예제에서는 수정 없이 사용할 수 있습니다.

```python
def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  cnn = CNN()
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)
```

### 학습 단계

이 함수는:

- [Module.apply](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply) 함수를 사용하여 주어진 파라미터와 입력 이미지 배치에서 신경망을 평가합니다.
- `cross_entropy_loss` 손실 함수를 계산합니다.
- [jax.value_and_grad](https://jax.readthedocs.io/en/latest/jax.html#jax.value_and_grad)를 사용하여 손실 함수와 그레디언트를 평가합니다.
- 옵티마이저에 그레이디언트의 [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions)를 적용하여 모델의 파라메터를 업데이트 합니다.
- 앞서 정의한 `compute_metrics` 함수를 사용하여 메트릭을 계산합니다.

JAX의 [@jit](https://jax.readthedocs.io/en/latest/jax.html#jax.jit) 데코레이터를 사용하여 전체 `train_step` 함수를 추적하고 이를 [XLA](https://www.tensorflow.org/xla)를 사용하여 하드웨어 가속기에서 더 빠르고 효율적으로 실행되는 융합 장치 작업으로 JIT 컴파일 합니다.

```python
@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=batch['label'])
  return state, metrics
```

### 평가 단계

[Module.apply](https://flax.readthedocs.io/en/latest/flax.linen.html#flax.linen.Module.apply)를 사용하여 테스트 세트에서 모델을 평가하는 함수를 만듭니다.

```python
@jax.jit
def eval_step(params, batch):
  logits = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits=logits, labels=batch['label'])
```

### 학습 함수

다음과 같은 학습 함수를 정의합니다:

- PRNGKey를 매개 변수로 사용하는 [jax.random.permutation](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.permutation.html)을 사용하여 각 epoch 전에 학습 데이터를 섞습니다.
- 각 배치에 대해 최적화 단계를 실행합니다.
- `jax.device_get`을 사용하여 장치에서 학습 메트릭을 검색하고 epoch의 각 배치에서 평균을 계산합니다.
- 업데이트된 파라메터와 학습 손실 및 정확도 메트릭이 포함된 옵티마이저를 반환합니다.

```python
def train_epoch(state, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  return state, epoch_metrics_np['loss'], epoch_metrics_np['accuracy']
```

### 평가 함수

다음과 같은 모델 평가 함수를 만듭니다:

- `jax.device_get`에 있는 장치에서 평가 메트릭을 검색합니다.
- JAX [pytree](https://jax.readthedocs.io/en/latest/pytrees.html#pytrees-and-jax-functions)에 저장된 메트릭 데이터를 복사합니다.

```python
def eval_model(params, test_ds):
  metrics = eval_step(params, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']
```

### 학습 상태 초기화

하나의 PRNGKey를 가져와서 그것을 분리하여 파라메터 초기화에 사용할 두 번째 키를 가져옵니다. (자세한 내용은 [PRNG chains](https://flax.readthedocs.io/en/latest/design_notes/linen_design_principles.html#how-are-parameters-represented-and-how-do-we-handle-general-differentiable-algorithms-that-update-stateful-variables)와 [JAX PRNG Design](https://jax.readthedocs.io/en/latest/design_notes/prng.html) 문서를 참조하세요.)

```python
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
```

`create_train_state` 함수는 모델 파라메터와 옵티마이저 모두 초기화하고 둘 다 반환되는 훈련 상태 데이터 클래스에 넣습니다.

```python
learning_rate = 0.1
momentum = 0.9

state = create_train_state(init_rng, learning_rate, momentum)
del init_rng  # 초기화 이후에는 사용하지 말아야 합니다.
```

### 모델 학습 및 평가

10 epoch이 완료되면 학습 데이터셋에서 모델 정확도는 약 93%, 테스트 데이터셋에서 모델 정확도는 약 89%를 달성할 수 있습니다.

```python
num_epochs = 10
batch_size = 32

for epoch in range(1, num_epochs + 1):
  # 셔플링 중에 별도의 PRNG 키를 사용하여 이미지 데이터 정렬
  rng, input_rng = jax.random.split(rng)
  # 학습 배치에 대해 최적화 단계 실행
  state, train_loss, train_accuracy = train_epoch(state, train_ds, batch_size, epoch, input_rng)
  # 각 학습 epoch 후 테스트 데이터 세트에서 모델 평가
  test_loss, test_accuracy = eval_model(state.params, test_ds)
  print(f"Epoch [{epoch}] - Train loss: {train_loss:.2f}, accuracy: {train_accuracy * 100:.2f}% / " \
            f"Test loss: {test_loss:.2f}, accuracy: {test_accuracy * 100:.2f}%")
```

> Epoch [1] - Train loss: 0.47, accuracy: 82.56% / Test loss: 0.40, accuracy: 84.49% <br>
> Epoch [2] - Train loss: 0.32, accuracy: 87.99% / Test loss: 0.33, accuracy: 87.39% <br>
> Epoch [3] - Train loss: 0.28, accuracy: 89.27% / Test loss: 0.31, accuracy: 88.69% <br>
> Epoch [4] - Train loss: 0.26, accuracy: 90.10% / Test loss: 0.30, accuracy: 89.17% <br>
> Epoch [5] - Train loss: 0.25, accuracy: 90.73% / Test loss: 0.32, accuracy: 88.98% <br>
> Epoch [6] - Train loss: 0.23, accuracy: 91.34% / Test loss: 0.32, accuracy: 89.65% <br>
> Epoch [7] - Train loss: 0.21, accuracy: 91.95% / Test loss: 0.31, accuracy: 89.96% <br>
> Epoch [8] - Train loss: 0.20, accuracy: 92.24% / Test loss: 0.30, accuracy: 90.08% <br>
> Epoch [9] - Train loss: 0.19, accuracy: 92.75% / Test loss: 0.32, accuracy: 89.88% <br>
> Epoch [10] - Train loss: 0.18, accuracy: 93.37% / Test loss: 0.34, accuracy: 89.45% <br>

### 결과 확인

학습된 모델을 사용하여 테스트 데이터셋의 첫 25개 이미지 데이터의 추론 결과를 시각화 해보겠습니다.

```python
logits = CNN().apply({'params': state.params}, test_ds['image'])
predict_results = logits.argmax(axis=1)  # 부동 소수점 값을 정수형으로 변환 합니다.

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(jnp.squeeze(test_ds['image'])[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[predict_results[i]])
plt.show()

```

![Test result](https://github.com/Taehun/taehun.github.io/blob/main/imgs/test_result.jpg?raw=true)

## 결론

위 예제에서 확인 할 수 있듯이 Flax 사용법은 기존 Tensorflow와 PyTorch 예제에서 많이 접했던 것들이 입니다. PRNGKey와 같이 새로운 것도 있지만, 손실 함수 및 옵티마이저 설정 하는 것은 이미 익숙한 코드 입니다. 이 기사 첫 부분에도 언급 하였지만 JAX는 이미 상용 프로젝트에 적용 할 수 있을만큼의 완성도가 많이 높아져서 지금부터라도 준비해서 JAX의 강점을 직접 경험해 보시기 바랍니다. Flax/JAX 모델의 최적화나 배포 관련 부분은 아직 부족하지만, 다른 배포 포맷으로 변환해서 사용하면 해결할 수 있습니다. 좀 더 많은 내용은 아래 JAX와 Flax 공식 문서를 참조하시기 바랍니다.

- [Flax 공식 문서](https://flax.readthedocs.io/en/latest/)
- [JAX 공식 문서](https://jax.readthedocs.io/en/latest/)
