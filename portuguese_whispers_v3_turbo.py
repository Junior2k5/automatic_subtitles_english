import whisper

# Carregar o modelo Whisper (pode ser 'base', 'small', 'medium', ou 'large')
model = whisper.load_model("small")  # 'small' é uma opção balanceada para transcrições em português

# Caminho do arquivo de áudio
audio_path = "extracted_audio.wav"

# Transcrever o áudio
result = model.transcribe(audio_path, language="pt")
print(result["text"])  # A transcrição estará aqui
