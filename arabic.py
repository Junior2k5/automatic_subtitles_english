import os
import ffmpeg
import torch
import librosa
from datetime import timedelta
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import srt
import urllib.error


def extract_audio_from_video(video_path, audio_path, sample_rate=16000):
    print(f"Extracting audio from {video_path} to {audio_path} with sample rate {sample_rate} Hz...")
    ffmpeg.input(video_path).output(audio_path, ar=sample_rate).run(overwrite_output=True)
    print("Audio extraction and resampling completed.")


def transcribe_audio(audio_path, processor, model, device):
    print(f"Transcribing audio from {audio_path} using Wav2Vec2...")

    # Load the audio file using librosa
    speech, sampling_rate = librosa.load(audio_path, sr=16000)

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]

    # Splitting transcription into segments (this is a simplistic approach; ideally, you should use a VAD or similar tool)
    segments = []
    words = transcription.split()
    segment_duration = len(speech) / 16000  # 16000 Hz is the sampling rate
    interval = segment_duration / len(words)
    for i, word in enumerate(words):
        start_time = i * interval
        end_time = (i + 1) * interval
        segments.append({'start': start_time, 'end': end_time, 'text': word})

    print("Audio transcription completed.")
    return segments


def format_srt(transcriptions):
    print("Formatting transcription to SRT...")
    subtitles = []
    for i, segment in enumerate(transcriptions):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        subtitles.append(srt.Subtitle(index=i + 1, start=start_time, end=end_time, content=segment['text'].strip()))
    srt_content = srt.compose(subtitles)
    return srt_content


def save_srt(content, srt_path):
    print(f"Saving SRT to {srt_path}...")
    with open(srt_path, "w", encoding='utf-8') as srt_file:
        srt_file.write(content)
    print("SRT file saved.")


def main():
    file_name = "arabic"
    video_path = fr"mp4/{file_name}.mp4"  # Replace with your local video file path
    audio_path = "extracted_audio.wav"
    srt_path = fr"mp4/{file_name}.srt"
    model_path = r"models/speech_ar"  # Directory where the model files are stored locally

    # Step 1: Extract and resample audio from video
    extract_audio_from_video(video_path, audio_path, sample_rate=16000)

    # Step 2: Load the processor and model from the local directory
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Step 3: Transcribe the audio
    transcriptions = transcribe_audio(audio_path, processor, model, device)

    # Step 4: Format the transcription to SRT content
    srt_content = format_srt(transcriptions)

    # Step 5: Save the transcription in SRT format
    save_srt(srt_content, srt_path)


if __name__ == "__main__":
    main()