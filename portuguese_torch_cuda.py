import os
import torch
import ffmpeg
import whisper
import srt
from datetime import timedelta


def extract_audio_from_video(video_path, audio_path):
    print(f"Extracting audio from {video_path} to {audio_path}...")
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
    print("Audio extraction completed.")


def transcribe_audio(audio_path, model):
    print(f"Transcribing audio from {audio_path} using Whisper...")
    result = model.transcribe(audio_path, language='pt')  # Alterado para português
    print("Audio transcription completed.")
    return result['segments']


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


def load_whisper_model_local(model_path, device):
    print(f"Loading Whisper model from {model_path}...")
    model = whisper.load_model("large", download_root=model_path)
    model.to(device)  # Mova o modelo para o dispositivo (GPU/CPU)
    print("Whisper model loaded successfully.")
    return model


def main():
    # Detectar GPU ou CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU está sendo utilizada.")
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU não está sendo utilizada.")
        print("Utilizando CPU.")

    file_name = "portuguese"
    video_path = fr"mp4/{file_name}.mp4"
    audio_path = "extracted_audio.wav"
    srt_path = fr"mp4/{file_name}.srt"
    model_path = r"models/whisper-large-v2"

    # Etapa 1: Extraia o áudio do vídeo
    extract_audio_from_video(video_path, audio_path)

    # Etapa 2: Carregue o modelo Whisper do armazenamento local
    model = load_whisper_model_local(model_path, device)

    # Etapa 3: Transcreva o áudio
    transcriptions = transcribe_audio(audio_path, model)

    # Etapa 4: Formate a transcrição para conteúdo SRT
    srt_content = format_srt(transcriptions)

    # Etapa 5: Salve a transcrição em formato SRT
    save_srt(srt_content, srt_path)


if __name__ == "__main__":
    main()