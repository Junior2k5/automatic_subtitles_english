import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def sample_speakers():
    # Lista de falantes e seus arquivos de áudio de referência
    speakers = {
        'Claribel Dervla': 'claribel_dervla_reference.wav',
        'Ana Florence': 'ana_florence_reference.wav'
    }

    for speaker, audio_file in speakers.items():
        print(f"{speaker}:")
        text = f"This is a text-to-speech model voice. My name is {speaker}."

        # Gerar áudio usando o arquivo de referência
        tts.tts_to_file(
            text=text,
            file_path=f"sample_{speaker}.wav",
            speaker_wav=audio_file,  # Arquivo de referência de áudio
            language="en"
        )
        print(f"Generated sample_{speaker}.wav")


# Executar função
sample_speakers()