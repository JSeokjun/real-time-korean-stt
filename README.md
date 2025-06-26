---
license: apache-2.0
language:
- ko
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- whisper
---

# 실시간 한국어 음성 인식 (STT) 모듈

이 프로젝트는 Hugging Face의 `SungBeom/whisper-small-ko` 모델을 사용하여 마이크로부터 실시간 음성 입력을 받아 한국어 텍스트로 변환하는 기능을 제공합니다.

## 주요 기능

-   실시간 음성 감지 및 텍스트 변환
-   일정 시간 (10초) 동안 음성 입력이 없으면, 인식된 전체 텍스트를 자동으로 `.txt` 파일로 저장
-   간단한 설정값 변경으로 음성 감지 민감도 조절 가능

## 요구사항

-   Python 3.8+
-   macOS (PyAudio 호환 환경)
-   `portaudio` 라이브러리

## 설치 방법

1.  **PortAudio 설치 (macOS의 경우)**

    Homebrew를 사용하여 `portaudio`를 설치합니다. 이 라이브러리는 마이크 입력을 처리하는 `pyaudio`에 필요합니다.
    ```bash
    brew install portaudio
    ```

2.  **종속성 설치**

    ```bash
    pip install -r requirements.txt
    ```

3.  **모델 다운로드**

    이 프로젝트는 `SungBeom/whisper-small-ko` 모델을 사용합니다. 아래 명령어를 실행하여 Hugging Face Hub로부터 모델 파일을 다운로드하세요. 모델 파일은 `models/` 디렉토리 안에 저장됩니다.

    ```bash
    # Hugging Face Hub에 로그인해야 할 수 있습니다.
    # huggingface-cli login

    # 모델 다운로드 (huggingface-cli 필요)
    huggingface-cli download --repo-type model SungBeom/whisper-small-ko --local-dir models
    ```

## 사용 방법

아래 명령어를 실행하여 실시간 음성 인식을 시작합니다.

```bash
python stt_module.py
```

-   프로그램이 실행되면 마이크 입력을 대기합니다.
-   음성이 감지되면 녹음이 시작되고, 발화가 끝나면 인식된 텍스트가 콘솔에 출력됩니다.
-   마지막 발화 후 10초가 지나면 전체 대화 내용이 `transcript_YYYYMMDD_HHMMSS.txt` 형식의 파일로 저장됩니다.
-   `Ctrl+C`를 눌러 프로그램을 종료할 수 있습니다.

## 파일 구성

-   `stt_module.py`: 메인 실행 스크립트. 실시간 음성 인식 로직을 포함합니다.
-   `requirements.txt`: 프로젝트 실행에 필요한 Python 라이브러리 목록입니다.
-   `pytorch_model.bin`, `config.json` 등: `whisper-small-ko` 모델 관련 파일들입니다.

---
<br>

# 원본 모델 정보 (`whisper-small-ko`)

> 아래 내용은 이 프로젝트에서 사용하는 `whisper-small-ko` 모델의 원본 Hugging Face 저장소에 있던 README 내용입니다. 모델의 학습 과정과 성능에 대한 자세한 정보를 담고 있습니다.

### 라이선스 및 태그
- **license:** apache-2.0
- **language:** ko
- **library_name:** transformers
- **pipeline_tag:** automatic-speech-recognition
- **tags:** whisper

### 모델 설명
해당 모델은 Whisper Small을 아래의 AI hub dataset에 대해 파인튜닝을 진행했습니다. <br>
데이터셋의 크기가 큰 관계로 데이터셋을 랜덤하게 섞은 후 5개로 나누어 학습을 진행했습니다. <br>

### 학습 결과

|    Dataset    | Training Loss | Epoch | Validation Loss | Wer     |
|:-------------:|:-------------:|:-----:|:---------------:|:-------:|
| Dataset part1 | 0.1943        |  0.2  | 0.0853          | 9.48    |

### 사용된 데이터셋
해당 모델은 AI hub의 많은 데이터셋을 한번에 학습시킨 것이 특징입니다. <br>
ASR은 domain에 대한 의존도가 매우 큽니다. 이 때문에 하나의 데이터셋에 학습을 시키더라도 다른 데이터셋에 대해서 테스트를 진행하면 성능이 크게 떨어지게 됩니다. <br>
이런 부분을 막기 위해 최대한 많은 데이터셋을 한 번에 학습시켰습니다. <br>
추후 사투리나 어린아이, 노인의 음성은 adapter를 활용하면 좋은 성능을 얻을 수 있을 것입니다.

| 데이터셋 이름 | 데이터 샘플 수(train/test) |
| --- | --- |
| 고객응대음성 | 2067668/21092 |
| 한국어 음성 | 620000/3000 |
| 한국인 대화 음성 | 2483570/142399 |
| 자유대화음성(일반남녀) | 1886882/263371 |
| 복지 분야 콜센터 상담데이터 | 1096704/206470 |
| 차량내 대화 데이터 | 2624132/332787 |
| 명령어 음성(노인남여) | 137467/237469 |
| 전체 | 10916423(13946시간)/1206588(1474시간) |


### 학습 절차

#### 학습 하이퍼파라미터

학습에 사용된 하이퍼파라미터는 다음과 같습니다:
- learning_rate: 1e-05
- train_batch_size: 32
- eval_batch_size: 16
- gradient_accumulation_steps: 2
- warmup_ratio: 0.01,
- num_train_epoch: 1