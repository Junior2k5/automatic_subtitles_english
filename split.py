import os
import subprocess

def split_video_with_ffmpeg(input_path, output_dir, max_duration):
    """
    Divide um vídeo em várias partes com duração máxima de max_duration minutos usando ffmpeg.

    :param input_path: Caminho do arquivo de entrada.
    :param output_dir: Diretório onde os clipes divididos serão salvos.
    :param max_duration: Duração máxima de cada clipe em minutos.
    """
    # Converter max_duration para segundos
    max_duration_seconds = max_duration * 60

    # Criar o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Obter o nome base do arquivo
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Comando ffmpeg para dividir o vídeo
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c", "copy",  # Não recodifica o vídeo
        "-map", "0",
        "-segment_time", str(max_duration_seconds),
        "-f", "segment",
        "-reset_timestamps", "1",
        os.path.join(output_dir, f"{base_name}_part_%03d.mp4")
    ]

    # Executar o comando
    try:
        subprocess.run(command, check=True)
        print(f"Divisão concluída. Partes salvas em {output_dir}.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao dividir o vídeo: {e}")

# Exemplo de uso
input_file = "mp4/english.mp4"
output_directory = "output_parts"
split_video_with_ffmpeg(input_file, output_directory, max_duration=15)
