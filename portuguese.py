import os
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


def load_whisper_model_local(model_path):
    print(f"Loading Whisper model from {model_path}...")
    model = whisper.load_model("large", download_root=model_path)  # Alterado para "large"
    print("Whisper model loaded successfully.")
    return model


def main():
    file_name = "portuguese"  # Não esqueça de atualizar o nome do arquivo, se necessário
    video_path = fr"mp4/{file_name}.mp4"  # Caminho para o arquivo de vídeo
    audio_path = "extracted_audio.wav"
    srt_path = fr"mp4/{file_name}.srt"
    model_path = r"models/whisper-small"  # Diretório onde o modelo "large" está armazenado

    # Step 1: Extract audio from video
    extract_audio_from_video(video_path, audio_path)

    # Step 2: Load the Whisper model from local storage
    model = load_whisper_model_local(model_path)

    # Step 3: Transcribe the audio
    transcriptions = transcribe_audio(audio_path, model)

    # Step 4: Format the transcription to SRT content
    srt_content = format_srt(transcriptions)

    # Step 5: Save the transcription in SRT format
    save_srt(srt_content, srt_path)


if __name__ == "__main__":
    main()