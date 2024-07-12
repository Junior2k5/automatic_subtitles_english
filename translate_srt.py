import os
import pysrt
from transformers import MarianMTModel, MarianTokenizer


def translate_text(text, model, tokenizer, src_lang="en", tgt_lang="pt"):
    # Preprocess the text
    text = f">>{tgt_lang}<< {text}"
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    # Perform translation and decode the output
    translated_tokens = model.generate(inputs, max_length=512)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text


def translate_subtitle_file(file_path, dest_folder, model, tokenizer, src_lang="en", tgt_lang="pt"):
    subs = pysrt.open(file_path, encoding='utf-8')

    for sub in subs:
        translated_text = translate_text(sub.text, model, tokenizer, src_lang, tgt_lang)
        sub.text = translated_text

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    translated_file_path = os.path.join(dest_folder, os.path.basename(file_path))
    subs.save(translated_file_path, encoding='utf-8')
    print(f"Translated subtitles saved to {translated_file_path}")


def translate_all_subtitles_in_folder(folder_path, dest_folder, model, tokenizer, src_lang="en", tgt_lang="pt"):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.srt'):
                file_path = os.path.join(root, file)
                print(f"Translating {file_path}...")
                translate_subtitle_file(file_path, dest_folder, model, tokenizer, src_lang, tgt_lang)
    print("All subtitles translated.")


if __name__ == "__main__":
    from transformers import MarianMTModel, MarianTokenizer

    # Load model and tokenizer from a local path
    local_model_path = './local_model/opus-mt-en-pt' #download it from https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt
    model = MarianMTModel.from_pretrained(local_model_path)
    tokenizer = MarianTokenizer.from_pretrained(local_model_path)

    folder_path = './mp4'
    dest_folder = './translated_srts'
    translate_all_subtitles_in_folder(folder_path, dest_folder, model, tokenizer, src_lang="en", tgt_lang="pt")
