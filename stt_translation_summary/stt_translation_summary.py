
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import os
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI
import torch

# --------- Settings ---------
LANGUAGE_TO_TRANSLATE = 'ko'  # Target language (Korean)
SAMPLE_RATE = 16000           # 16kHz sampling rate
CHUNK_DURATION = 2            # seconds to record before processing
MODEL_SIZE = "tiny"           # Model size ("tiny" for lightweight)

# --------- Initialize ---------
print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")  # CPU + int8 for M2 Mac efficiency
translator = GoogleTranslator(source='auto', target=LANGUAGE_TO_TRANSLATE)

# --------- Functions ---------

def record_audio_chunk(duration=CHUNK_DURATION, samplerate=SAMPLE_RATE):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Block until recording is done
    return recording.flatten()

def transcribe(audio_np):
    segments, _ = model.transcribe(audio_np, language ="en")
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
    return full_text.strip()

def translate(text):
    if text.strip() == "":
        return ""
    translated = translator.translate(text)
    return translated

# --------- Main Loop ---------

print("Starting real-time captioning...\n")

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import torch
import os

# --------- Settings ---------
LANGUAGE_TO_TRANSLATE = 'ko'  # Target language (Korean)
SAMPLE_RATE = 16000           # 16kHz sampling rate
CHUNK_DURATION = 2            # seconds to record before processing
MODEL_SIZE = "tiny"           # Model size ("tiny" for lightweight)

# --------- Initialize ---------
print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")  # CPU + int8 for M2 Mac efficiency
translator = GoogleTranslator(source='auto', target=LANGUAGE_TO_TRANSLATE)

# --------- Functions ---------

def record_audio_chunk(duration=CHUNK_DURATION, samplerate=SAMPLE_RATE):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Block until recording is done
    return recording.flatten()

def transcribe(audio_np):
    segments, _ = model.transcribe(audio_np, language ="en")
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
    return full_text.strip()

def translate(text):
    if text.strip() == "":
        return ""
    translated = translator.translate(text)
    return translated

# --------- Main Loop ---------
print("Starting real-time captioning...\n")
meeting_transcript = "" 

try:
    while True:
        audio_chunk = record_audio_chunk()
        recognized_text = transcribe(audio_chunk)
        translated_text = translate(recognized_text)

        if recognized_text.strip() != "":
            print(f"\n[Recognized] {recognized_text}")
            print(f"[Translated] {translated_text}\n")
            meeting_transcript += translated_text + " " 

except KeyboardInterrupt:
    print("\n회의 종료됨. 번역 저장 중...")
    with open("meeting_transcript.txt", "w", encoding="utf-8") as f:
        f.write(meeting_transcript)
        print("번역 내용이 'meeting_transcript.txt'에 저장되었습니다.")

## --------- summarization ---------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))

def build_prompt(meeting_transcript: str) -> str:
    try:
        template = {
            'system_role': "당신은 회의 전체 내용을 카테고리별로 요약해주는 요약 AI입니다.",
            'output_format': [
                "다음 회의 날짜와 시간: OOO",
                "아이템: OOO",
                "해야 할 일: OOO",
                "팀원들의 의견: OOO",
                "멘토의 피드백: OOO"
            ]
        }

        prompt = f""" {template['system_role']}
아래 회의 전체 내용을 다음 형식으로 요약하세요:

[출력 예시]
{chr(10).join(template['output_format'])}

[회의 전체 내용]
{meeting_transcript}

[정리된 결과] """
        return prompt

    except Exception as e:
        logging.error(f"Failed to build prompt: {str(e)}")
        raise

def summarize_meeting(transcript: str):
    prompt = build_prompt(transcript)


    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 회의 전체 내용을 카테고리별로 요약해주는 요약 AI입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )


    return response.choices[0].message.content.strip()

# --------- Main Loop ---------
if __name__ == "__main__":
    if not os.path.exists("meeting_transcript.txt"):
        print("meeting_transcript.txt doesn't exist.")
        exit()

    with open("meeting_transcript.txt", "r", encoding="utf-8") as f:
        meeting_transcript = f.read()

    try:
        summary = summarize_meeting(meeting_transcript)
        print("\n 회의 요약 결과:\n")
        print(summary)
    except Exception as e:
        logging.error(f"요약 실패: {str(e)}")