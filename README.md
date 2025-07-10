# 실시간 한국어 음성 인식 (STT) 모듈

이 프로젝트는 OpenAI의 `Whisper` 를 사용하여 마이크로부터 실시간 음성 입력을 받아 한국어 텍스트로 변환하는 기능을 제공합니다.

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

    이 프로젝트는 OpenAI의 `Whisper`를 사용합니다. 
    Hugging Face Hub로부터 원하는 모델 파일을 다운로드하여 `models/` 디렉토리 안에 저장하세요.

    프로젝트에 사용한 모델의 예시는 다음과 같습니다.

    - openai/whisper-large-v3-turbo
    - o0dimplz0o/Whisper-Large-v3-turbo-STT-Zeroth-KO-v2
    - openai/whisper-small
    - SungBeom/whisper-small-ko
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