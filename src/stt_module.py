import torch
import pyaudio
import numpy as np
import time
import os
import noisereduce as nr
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
from llm_module import GemmaRK3588, GemmaCPU

# --- 모델 path 모음 ---
models = [
    "whisper-large-v3-turbo",
    "whisper-small",
    "whisper-small-ko"
]

# --- 경로 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, f"models/{models[0]}")
NPU_LLM_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/unsloth-gemma-3-4b-it-rk3588-1.2.1")
CPU_LLM_MODEL_DIR = os.path.join(PROJECT_ROOT, "models/gemma-3-4b-it")
RKLLM_LIB_PATH = os.path.join(PROJECT_ROOT, "lib/librkllmrt.so")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- 설정 ---
CHUNK = 512  # 한 번에 읽어올 오디오 데이터의 크기
FORMAT = pyaudio.paInt16  # 오디오 포맷 (16비트)
CHANNELS = 1  # 모노 채널
RATE = 16000  # 샘플링 레이트 (Whisper는 16000Hz를 사용)
VAD_THRESHOLD = 0.3  # 음성으로 판단할 확률 임계값
SPEECH_END_THRESHOLD = 3  # 발화가 끝났다고 판단하는 침묵 시간 (초)
PAUSE_DURATION = 10 # 대화가 끝났다고 판단하는 침묵 시간 (초)
NOISE_LEARNING_DURATION = 5 # 시작 시, 노이즈를 학습할 시간 (초)

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
        
        # 'whisper-small-ko' 모델의 generation_config가 오래되어 'language' 인자를 지원하지 않는 문제를 해결하기 위해
        # 'openai/whisper-small' 모델의 generation_config를 명시적으로 로드하여 설정합니다.
        if models == "whisper-small-ko" :
            generation_config = GenerationConfig.from_pretrained("models/whisper-small")
            model.generation_config = generation_config
        
        model.config.forced_decoder_ids = None
    except Exception as e:
        print(f"모델 로딩 중 오류가 발생했습니다: {e}")
        print(f"{MODEL_PATH} 디렉토리에 모델 파일들이 모두 있는지 확인해주세요.")
        return

    print("모델 로딩 완료.")

    # --- LLM 모델 로드 ---
    llm = None
    # 1. NPU(RKLLM) 모델 우선 시도
    try:
        if os.path.exists(NPU_LLM_MODEL_DIR) and os.path.exists(RKLLM_LIB_PATH):
            print("\nNPU 가속 LLM 모델(GemmaRK3588)을 로딩합니다...")
            llm = GemmaRK3588(model_dir_path=NPU_LLM_MODEL_DIR, rkllm_lib_path=RKLLM_LIB_PATH)
            print("NPU LLM 모델 로딩 완료.")
        else:
            print("\nNPU LLM 모델/라이브러리를 찾을 수 없습니다. CPU/MPS 모델로 대체를 시도합니다.")
            raise FileNotFoundError("NPU 모델 경로 또는 라이브러리 없음")

    except Exception as e:
        print(f"NPU LLM 모델 로딩 실패: {e}")
        # 2. CPU/MPS 모델로 대체 시도
        try:
            if os.path.exists(CPU_LLM_MODEL_DIR):
                print(f"\nCPU/MPS LLM 모델(GemmaCPU)을 로딩합니다... (장치: {device})")
                llm = GemmaCPU(model_dir_path=CPU_LLM_MODEL_DIR, device=device)
                print("CPU/MPS LLM 모델 로딩 완료.")
            else:
                 print("\n[경고] CPU 대체 LLM 모델 경로를 찾을 수 없어 LLM 기능을 비활성화합니다.")
                 print(f"  - 모델 경로 확인: {CPU_LLM_MODEL_DIR}")
        except Exception as cpu_e:
            print(f"CPU/MPS LLM 모델 로딩 중 오류 발생: {cpu_e}")


    # --- Silero VAD 모델 로드 ---
    try:
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          onnx=False) # mps device는 onnx=False 필요
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils
        print("Silero VAD 모델 로딩 완료.")
    except Exception as e:
        print(f"Silero VAD 모델 로딩 중 오류가 발생했습니다: {e}")
        return

    # PyAudio 초기화
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"\n{NOISE_LEARNING_DURATION}초간 주변 소음을 학습합니다. 조용히 해주세요...")
    
    # 노이즈 프로필 생성
    noise_frames = []
    for _ in range(0, int(RATE / CHUNK * NOISE_LEARNING_DURATION)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        noise_frames.append(data)
    
    noise_sample = np.frombuffer(b''.join(noise_frames), dtype=np.int16)
    
    print("소음 학습 완료.")
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

                # 잡음 제거
                denoised_data = nr.reduce_noise(y=audio_data, sr=RATE, y_noise=noise_sample, n_fft=CHUNK)
                
                # VAD를 위한 데이터 준비 (float32 텐서)
                audio_float32 = denoised_data.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)

                # 음성 확률 확인
                speech_prob = vad_model(audio_tensor, RATE).item()

                # 대화가 끝났는지 확인 (PAUSE_DURATION)
                if time.time() - last_speech_time > PAUSE_DURATION and full_transcript:
                    print("\n--- 긴 침묵 감지 ---")
                    save_transcript(full_transcript)
                    
                    if llm:
                        print(f"\n[LLM 요청] 다음 텍스트를 분석합니다:\n{full_transcript}")
                        try:
                            # LLM에 분석 요청
                            llm_response = llm.generate(
                                full_transcript, 
                                system_prompt="당신은 한국어 대화의 핵심 내용을 파악하고, 대화의 주제와 주요 결론을 간결하게 요약하는 전문 분석가입니다. "
                            )
                            print("\n--- LLM 응답 ---")
                            print(llm_response)
                            print("------------------")
                        except Exception as e:
                            print(f"\n[LLM 오류] 응답 생성 중 오류가 발생했습니다: {e}")

                    full_transcript = "" # 분석 및 저장 후 초기화
                    last_speech_time = time.time() # 중복 분석 방지를 위해 시간 초기화
                
                if speech_prob > VAD_THRESHOLD:
                    print(f"음성 감지됨 (확률: {speech_prob:.2f}), 녹음 시작...")
                    is_speaking = True
                    frames.append(denoised_data.tobytes())
                    last_speech_time = time.time()
                    last_active_time = time.time()
                    break

            # 음성이 진행되는 동안 녹음
            while is_speaking:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # 잡음 제거
                denoised_data = nr.reduce_noise(y=audio_data, sr=RATE, y_noise=noise_sample, n_fft=CHUNK)
                frames.append(denoised_data.tobytes())

                # VAD를 위한 데이터 준비
                audio_float32 = denoised_data.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)

                # 음성 확률 확인
                speech_prob = vad_model(audio_tensor, RATE).item()
                
                if speech_prob > VAD_THRESHOLD:
                    last_active_time = time.time()
                    last_speech_time = time.time()
                
                # 발화가 끝났는지 확인
                if time.time() - last_active_time > SPEECH_END_THRESHOLD:
                    print("발화 종료, 음성 처리 중...")
                    
                    audio_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if len(audio_np) > 0:
                        input_features = processor(audio_np, sampling_rate=RATE, return_tensors="pt").input_features
                        
                        predicted_ids = model.generate(
                            input_features.to(device),
                            language="ko", 
                            task="transcribe"
                        )
                        
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