import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pyaudio
import numpy as np
import time
from datetime import datetime
import os

# --- 모델 path 모음 ---
models = [
    "whisper-large-v3-turbo",
    "Whisper-Large-v3-turbo-STT-Zeroth-KO-v2",
    "whisper-small",
    "whisper-small-ko"
]

# --- 경로 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, f"models/{models[0]}")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- 설정 ---
CHUNK = 1024  # 한 번에 읽어올 오디오 데이터의 크기
FORMAT = pyaudio.paInt16  # 오디오 포맷 (16비트)
CHANNELS = 1  # 모노 채널
RATE = 16000  # 샘플링 레이트 (Whisper는 16000Hz를 사용)
SILENCE_THRESHOLD = 1000  # 정적을 음성으로 인식하지 않기 위한 임계값
SPEECH_END_THRESHOLD = 0.5  # 발화가 끝났다고 판단하는 침묵 시간 (초)
PAUSE_DURATION = 10 # 대화가 끝났다고 판단하는 침묵 시간 (초)

def save_transcript(text):
    """인식된 텍스트를 타임스탬프 파일로 저장"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    if not text.strip():
        print("저장할 텍스트가 없습니다.")
        return
    
    filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[저장 완료] 파일명: {filepath}")

def main():
    """실시간 음성 인식 메인 함수"""
    # 모델 및 프로세서 로드
    print(f"모델을 로딩합니다... ({MODEL_PATH})")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"사용 디바이스: {device}")

    try:
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
        model.config.forced_decoder_ids = None
    except Exception as e:
        print(f"모델 로딩 중 오류가 발생했습니다: {e}")
        print(f"{MODEL_PATH} 디렉토리에 모델 파일들이 모두 있는지 확인해주세요.")
        return

    print("모델 로딩 완료.")

    # PyAudio 초기화
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("\n실시간 음성 인식을 시작합니다. (Ctrl+C로 종료)")

    full_transcript = ""
    last_speech_time = time.time()

    try:
        while True:
            frames = []
            is_speaking = False
            last_active_time = time.time()
            
            # 음성이 감지될 때까지 대기
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # 현재 시간에 따라 PAUSE_DURATION 체크
                if time.time() - last_speech_time > PAUSE_DURATION and full_transcript:
                    save_transcript(full_transcript)
                    full_transcript = "" # 저장 후 초기화
                
                if np.max(audio_data) > SILENCE_THRESHOLD:
                    print("음성 감지됨, 녹음 시작...")
                    is_speaking = True
                    frames.append(data)
                    last_speech_time = time.time()
                    last_active_time = time.time()
                    break

            # 음성이 진행되는 동안 녹음
            while is_speaking:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                if np.max(audio_data) > SILENCE_THRESHOLD:
                    last_active_time = time.time()
                    last_speech_time = time.time()
                
                # 발화가 끝났는지 확인
                if time.time() - last_active_time > SPEECH_END_THRESHOLD:
                    print("발화 종료, 음성 처리 중...")
                    
                    audio_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if len(audio_np) > 0:
                        input_features = processor(audio_np, sampling_rate=RATE, return_tensors="pt").input_features
                        
                        predicted_ids = model.generate(input_features.to(device))
                        
                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                        
                        if transcription:
                            print(f"인식된 텍스트: {transcription}")
                            full_transcript += transcription + " "
                        
                    frames = []
                    is_speaking = False
                    break

    except (KeyboardInterrupt, OSError) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n프로그램을 종료합니다.")
        else:
            print(f"\n프로그램 오류로 종료합니다: {e}")
            
        if full_transcript:
            save_transcript(full_transcript)
            
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    main() 