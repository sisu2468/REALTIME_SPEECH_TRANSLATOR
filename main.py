import asyncio
import gradio as gr
import torchaudio
from vosk import Model, KaldiRecognizer
import math
import json
import numpy as np
from google.cloud import translate_v2 as translate
import os
from scipy import signal
import time
import threading
import queue
import html
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from collections import deque
import pandas as pd

# Google Cloud credentialsの設定
if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    raise EnvironmentError("Please set the GOOGLE_APPLICATION_CREDENTIALS environment variable")

# Voskモデルの初期化（日本語と英語）
current_dir = os.path.dirname(os.path.abspath(__file__))
ja_model_path = os.path.join(current_dir, "models", "ja")
en_model_path = os.path.join(current_dir, "models", "en")

try:
    ja_model = Model(model_path=ja_model_path)
    en_model = Model(model_path=en_model_path)
except Exception as e:
    raise RuntimeError("モデルの読み込みに失敗しました。setup_models.py を実行してモデルをダウンロードしてください。")

SAMPLE_RATE = 16000  # サンプリングレート
BUFFER_SIZE = SAMPLE_RATE * 5  # 5秒分のバッファ

ja_recognizer = KaldiRecognizer(ja_model, SAMPLE_RATE)
ja_recognizer.SetMaxAlternatives(0)
ja_recognizer.SetWords(True)
en_recognizer = KaldiRecognizer(en_model, SAMPLE_RATE)
en_recognizer.SetMaxAlternatives(0)
en_recognizer.SetWords(True)

# Google Cloud Translation clientの初期化
translate_client = translate.Client()

# SpeechBrain言語識別モデルの初期化
language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmpdir")

class Word:
    def __init__(self, word, start, end, conf, fixed=True):
        self.word = word
        self.start = start
        self.end = end
        self.conf = conf
        self.fixed = fixed

    def __repr__(self):
        return f"Word('{self.word}'({self.conf:.2f}){self.start:.2f}-{self.end:.2f})"

    def __str__(self):
        return f"{self.word}({self.conf:.2f})"

class LanguageDetectionResult:
    def __init__(self, lang_with_probs: dict[str, float], start_time, end_time):
        self.lang_with_probs = lang_with_probs
        self.start_time = start_time
        self.end_time = end_time

    def language(self):
        return max(self.lang_with_probs, key=self.lang_with_probs.get)

    def prob(self, lang):
        return self.lang_with_probs.get(lang, 0)

    def __repr__(self):
        return f"LanguageDetectionResult({self.lang_with_probs}, {self.start_time:.2f}-{self.end_time:.2f})"

    def __str__(self):
        return ", ".join([f"{lang}({prob:.2f})" for lang, prob in self.lang_with_probs.items()]) + f"({self.start_time:.2f}-{self.end_time:.2f})"

class RecognitionResult:
    def __init__(self):
        self.words = []
        self.start_time = float('inf')
        self.end_time = 0

    def add_word(self, word, start, end, conf):
        self.words.append(Word(word, start, end, conf))
        self.start_time = min(self.start_time, start)
        self.end_time = max(self.end_time, end)

    @property
    def text(self):
        return " ".join(word.word for word in self.words)

    @property
    def confidence(self):
        if not self.words:
            return 0
        return sum(word.conf for word in self.words) / len(self.words)

    def __repr__(self):
        return f"RecognitionResult({self.start_time:.2f}-{self.end_time:.2f}, {[str(word) for word in self.words]}, conf={self.confidence:.2f})"

    def __str__(self):
        return f"{self.text} ({self.confidence:.2f})"

    def debug_str(self):
        return f"{self.text} ({self.confidence:.2f}) [{self.start_time:.2f}-{self.end_time:.2f}]"

# グローバル変数の定義
accumulated_time = 0
accumulated_lang: list[LanguageDetectionResult] = []
accumulated_words_ja: list[Word] = []
accumulated_words_en: list[Word] = []
language_detection_results = []
accumulated_all_ja = ""
accumulated_all_en = ""
current_language = "unknown"
audio_buffer = deque(maxlen=BUFFER_SIZE)
total_audio_duration = 0  # 累積音声データの長さを追跡
buffer_start_time = 0

# 履歴用データフレーム
history_df = pd.DataFrame(columns=["Time", "Language Detection", "Japanese ASR", "English ASR"])

# 翻訳キュー
translation_queue = queue.Queue()

def translate_text():
    global accumulated_all_ja, accumulated_all_en
    while True:
        try:
            text, src_lang = translation_queue.get(timeout=1)
            if src_lang == 'ja':
                en_translation = translate_client.translate(text, target_language='en')
                accumulated_all_en += html.unescape(en_translation['translatedText']) + " "
                accumulated_all_ja += text + " "
            else:
                ja_translation = translate_client.translate(text, target_language='ja')
                accumulated_all_ja += html.unescape(ja_translation['translatedText']) + " "
                accumulated_all_en += text + " "
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"翻訳エラー: {e}")

# 翻訳スレッドの開始
translation_thread = threading.Thread(target=translate_text, daemon=True)
translation_thread.start()

ja_ind = language_id.hparams.label_encoder.lab2ind["ja: Japanese"]
en_ind = language_id.hparams.label_encoder.lab2ind["en: English"]
def get_language_probabilities(out_prob):
    ja_prob = math.exp(out_prob[0][ja_ind].item())
    en_prob = math.exp(out_prob[0][en_ind].item())
    return ja_prob, en_prob

def detect_language_from_audio(audio_data, chunk_end_time) -> LanguageDetectionResult:
    global audio_buffer, total_audio_duration

    # バッファにデータを追加
    audio_buffer.extend(audio_data)

    # 現在の音声チャンクの開始時間と終了時間を計算
    chunk_duration = len(audio_buffer) / SAMPLE_RATE
    chunk_start_time = chunk_end_time - chunk_duration

    buffered_audio = np.array(audio_buffer)
    buffered_audio = buffered_audio / np.max(np.abs(buffered_audio))  # 正規化
    audio_tensor = torch.tensor(buffered_audio).float().unsqueeze(0)

    out_prob, score, index, text_lab = language_id.classify_batch(audio_tensor)

    ja_prob, en_prob = get_language_probabilities(out_prob)

    # 結果の表示
    print(f"Japanese: {ja_prob*100:.2f}%")
    print(f"English: {en_prob*100:.2f}%")

    if ja_prob > en_prob:
        lang = 'ja'
        confidence = ja_prob
    else:
        lang = 'en'
        confidence = en_prob

    detection_res = LanguageDetectionResult({"ja": ja_prob, "en": en_prob}, chunk_start_time, chunk_end_time)

    # バッファをスライディング
    if len(buffered_audio) > BUFFER_SIZE:
        audio_buffer = deque(list(audio_buffer)[SAMPLE_RATE:], maxlen=BUFFER_SIZE)

    return detection_res

def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr != target_sr:
        print(f"Resampling audio from {orig_sr} to {target_sr}")
        audio = audio.astype(np.float32)
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        audio = resampler(torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
    return audio

def process_audio(audio):
    global ja_recognizer, en_recognizer, accumulated_words_ja, accumulated_words_en
    global current_language
    global accumulated_time, accumulated_lang, accumulated_all_ja, accumulated_all_en, language_detection_results, audio_buffer, history_df

    if audio is None:
        return (format_results(accumulated_words_ja, "日本語文字起こし"),
                format_results(accumulated_words_en, "英語文字起こし"),
                "No audio input",
                combine_best_results(),
                accumulated_all_ja,
                accumulated_all_en,
                history_df)

    sample_rate, audio_data = audio
    accumulated_time += len(audio_data) / sample_rate

    print(f"Received audio data of length {len(audio_data)} with sample rate {sample_rate}. Sec = {len(audio_data) / sample_rate}")

    # サンプルレートの変換 (16kHzへ)
    audio_data = resample_audio(audio_data, sample_rate, target_sr=16000)

    # 言語検出
    last_language = current_language
    lang_detection_res = detect_language_from_audio(audio_data, accumulated_time)
    current_language = lang_detection_res.language()
    ja_prob = lang_detection_res.lang_with_probs.get('ja', 0)
    en_prob = lang_detection_res.lang_with_probs.get('en', 0)

    # 音声データの正規化
    audio_data = audio_data.astype(np.float32)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data = np.int16(audio_data * 32767) # int16に変換

    if current_language == 'unknown':
        # バッファリング中は一つ前の言語を使い続ける
        current_language = last_language

    accumulated_lang.append(lang_detection_res)

    print("Current language:", current_language)

    # 日本語の文字起こし
    ja_confidence = 0
    if ja_recognizer.AcceptWaveform(audio_data.tobytes()):
        result = json.loads(ja_recognizer.Result())
        if result.get("result"):
            rec_result = RecognitionResult()
            # -1の途中結果を削除してから追加
            accumulated_words_ja = [word for word in accumulated_words_ja if word.fixed]

            for word_info in result["result"]:
                word = Word(word_info["word"], word_info["start"], word_info["end"], word_info.get("conf", 0), fixed=True)
                rec_result.add_word(word.word, word.start, word.end, word.conf)
                accumulated_words_ja.append(word)
            ja_confidence = rec_result.confidence
            translation_queue.put((rec_result.text, 'ja'))
    else:
        partial = json.loads(ja_recognizer.PartialResult())
        if partial.get("partial", ""):
            # -1の途中結果を削除してから追加
            accumulated_words_ja = [word for word in accumulated_words_ja if word.fixed]

            for word in partial["partial"].split():
                accumulated_words_ja.append(Word(word, accumulated_time, accumulated_time, -1, fixed=False))

    # 英語の文字起こし
    en_confidence = 0
    if en_recognizer.AcceptWaveform(audio_data.tobytes()):
        result = json.loads(en_recognizer.Result())
        if result.get("result"):
            rec_result = RecognitionResult()
            # -1の途中結果を削除してから追加
            accumulated_words_en = [word for word in accumulated_words_en if word.fixed]

            for word_info in result["result"]:
                word = Word(word_info["word"], word_info["start"], word_info["end"], word_info.get("conf", 0), fixed=True)
                rec_result.add_word(word.word, word.start, word.end, word.conf)
                accumulated_words_en.append(word)
            en_confidence = rec_result.confidence
            translation_queue.put((rec_result.text, 'en'))
    else:
        partial = json.loads(en_recognizer.PartialResult())
        if partial.get("partial", ""):
            # -1の途中結果を削除してから追加
            accumulated_words_en = [word for word in accumulated_words_en if word.fixed]

            for word in partial["partial"].split():
                accumulated_words_en.append(Word(word, accumulated_time, accumulated_time, -1, fixed=False))

    history_df = generate_accumulated_df(accumulated_lang, accumulated_words_ja, accumulated_words_en)

    language_info = (f"検出言語: {current_language.upper()}\n"
                     f"日本語確率: {ja_prob*100:.1f}%, 英語確率: {en_prob*100:.1f}%\n"
                     f"日本語信頼度: {ja_confidence:.2f}, 英語信頼度: {en_confidence:.2f}")

    return (format_results(accumulated_words_ja, "日本語文字起こし"),
            format_results(accumulated_words_en, "英語文字起こし"),
            combine_best_results(),
            accumulated_all_ja,
            accumulated_all_en,
            history_df)

def generate_accumulated_df(accumulated_lang, accumulated_words_ja, accumulated_words_en):
    # 言語検出の各時間を基準にデータを揃える
    time_stamps = [lang.start_time for lang in accumulated_lang]
    lang_list = [str(lang) for lang in accumulated_lang]

    # 各タイムスタンプに対応する日本語ASRと英語ASRのリストを作成
    ja_list = [[] for _ in range(len(time_stamps))]
    en_list = [[] for _ in range(len(time_stamps))]

    # 日本語ASRの結果を時間スタンプに対応させる
    ja_index = 0
    time_stamp_index = 0
    while ja_index < len(accumulated_words_ja) and time_stamp_index < len(time_stamps):
        if accumulated_words_ja[ja_index].end <= time_stamps[time_stamp_index]:
            ja_list[time_stamp_index].append(accumulated_words_ja[ja_index])
            ja_index += 1
        else:
            time_stamp_index += 1

    # 英語ASRの結果を時間スタンプに対応させる
    en_index = 0
    time_stamp_index = 0
    while en_index < len(accumulated_words_en) and time_stamp_index < len(time_stamps):
        if accumulated_words_en[en_index].end <= time_stamps[time_stamp_index]:
            en_list[time_stamp_index].append(accumulated_words_en[en_index])
            en_index += 1
        else:
            time_stamp_index += 1
    ja_list = [", ".join([ja.__repr__() for ja in ja_words]) for ja_words in ja_list]
    en_list = [", ".join([en.__repr__() for en in en_words]) for en_words in en_list]

    # データフレームの作成
    df = pd.DataFrame({
        "Time": [f"{ts:.2f}s" for ts in time_stamps],
        "Language Detection": lang_list,
        "Japanese ASR": ja_list,
        "English ASR": en_list,
    })
    return df

def combine_best_results():
    best_results = []
    ja_index = 0
    en_index = 0
    current_time = 0

    while ja_index < len(accumulated_words_ja) and en_index < len(accumulated_words_en):
        ja_word = accumulated_words_ja[ja_index]
        en_word = accumulated_words_en[en_index]

        while ja_word.end <= current_time and ja_index < len(accumulated_words_ja):
            lang, ja_prob, en_prob = get_language_at_time(ja_word.start)
            if lang == 'ja' or (lang == 'unknown' and ja_prob >= en_prob):
                best_results.append(ja_word)
                current_time = ja_word.end
            ja_index += 1
            if ja_index < len(accumulated_words_ja):
                ja_word = accumulated_words_ja[ja_index]

        while en_word.end <= current_time and en_index < len(accumulated_words_en):
            lang, ja_prob, en_prob = get_language_at_time(en_word.start)
            if lang == 'en' or (lang == 'unknown' and en_prob >= ja_prob):
                best_results.append(en_word)
                current_time = en_word.end
            en_index += 1
            if en_index < len(accumulated_words_en):
                en_word = accumulated_words_en[en_index]

        word = ja_word if ja_word.start < en_word.start else en_word

        lang, ja_prob, en_prob = get_language_at_time(word.start)
        if lang == 'ja' or (lang == 'unknown' and ja_prob >= en_prob):
            best_results.append(ja_word)
            current_time = ja_word.end
            ja_index += 1

            if en_word.end < current_time:
                en_index += 1
        else:
            best_results.append(en_word)
            current_time = en_word.end
            en_index += 1

            if ja_word.end < current_time:
                ja_index += 1

    return format_results(best_results, "ベスト文字起こし")

def get_language_at_time(time):
    ja_probs = []
    en_probs = []

    # 各時刻に対して重なるすべての言語検出結果を収集
    for lang_res in accumulated_lang:
        if lang_res.start_time <= time and time < lang_res.end_time:
            ja_probs.append(lang_res.prob('ja'))
            en_probs.append(lang_res.prob('en'))
        elif time < lang_res.start_time:
            break

    if len(ja_probs) == 0 and len(en_probs) == 0:
        return current_language, 1, 1

    # 確率を統合して最終的な確率を計算
    total_ja_prob = sum(ja_probs) / len(ja_probs) if ja_probs else 0
    total_en_prob = sum(en_probs) / len(en_probs) if en_probs else 0
    total_en_prob += 0.25  # 英語の確率を少し上げる

    if total_ja_prob > total_en_prob:
        lang = 'ja'
    elif total_en_prob > total_ja_prob:
        lang = 'en'
    else:
        lang = 'unknown'

    return lang, total_ja_prob, total_en_prob

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def format_results(results, title=""):
    formatted_text = ""
    for word in results:
        confidence = word.conf
        # 緑から赤へのグラデーションを計算
        if confidence > 0.5:
            green_intensity = int(200 + 55 * ((confidence - 0.5) / 0.5))
            red_intensity = 200
            blue_intensity = 200
        else:
            green_intensity = 200
            blue_intensity = 200
            red_intensity = int(200 + 55 * (confidence / 0.5))
        color = f"rgb({red_intensity},{green_intensity},{blue_intensity})"
        if is_ascii(word.word):
            formatted_text += f'<span style="background-color: {color}; display: inline-block;">{word.word}&nbsp;</span>'  # スペースを追加
        else:
            formatted_text += f'<span style="background-color: {color}; display: inline-block;">{word.word}</span>'
    if title:
        formatted_text = f"<h2>{title}</h2>" + formatted_text
    return formatted_text

def reset_recognizer():
    global ja_recognizer, en_recognizer, accumulated_words_ja, accumulated_words_en, language_detection_results
    global accumulated_all_ja, accumulated_all_en, accumulated_lang, current_language
    global audio_buffer, total_audio_duration, history_df

    ja_recognizer = KaldiRecognizer(ja_model, SAMPLE_RATE)
    ja_recognizer.SetMaxAlternatives(0)
    ja_recognizer.SetWords(True)
    en_recognizer = KaldiRecognizer(en_model, SAMPLE_RATE)
    en_recognizer.SetMaxAlternatives(0)
    en_recognizer.SetWords(True)
    accumulated_words_ja = []
    accumulated_words_en = []
    accumulated_lang = []
    language_detection_results = []
    accumulated_all_ja = ""
    accumulated_all_en = ""
    current_language = "unknown"
    audio_buffer = deque(maxlen=BUFFER_SIZE)
    total_audio_duration = 0
    history_df = pd.DataFrame(columns=["Time", "Language Detection", "Japanese ASR", "English ASR"])
    return "リセットしました。新しい音声認識セッションを開始します。", "", "", "", "", "", history_df

async def process_file(audio_file):
    print("Processing file...")
    print(audio_file)
    global accumulated_time
    sample_rate, audio_data = audio_file
    chunk_size = SAMPLE_RATE  # 1秒ごとのチャンクサイズ
    num_chunks = int(np.ceil(len(audio_data) / chunk_size))

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(audio_data))
        chunk = (sample_rate, audio_data[start:end])
        ja, en, best, all_ja, all_en, history = process_audio(chunk)

        yield (ja, en, best, all_ja, all_en, history)
        await asyncio.sleep(0.05)  # 小さな遅延を追加してリアルタイム性をシミュレート

def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("# 多言語対応リアルタイム音声翻訳・文字起こしアプリ (音声ベース言語判定機能付き)")
        with gr.TabItem("Microphone"):
            with gr.Row():
                audio_input_mic = gr.Audio(sources=["microphone"], streaming=True, waveform_options={"sample_rate": SAMPLE_RATE})
                reset_button_mic = gr.Button("リセット")
            with gr.Row():
                japanese_output_mic = gr.HTML(label="日本語文字起こし")
                english_output_mic = gr.HTML(label="英語文字起こし")
            with gr.Row():
                best_output_mic = gr.HTML(label="ベスト文字起こし結果")
            with gr.Row():
                all_japanese_output_mic = gr.Textbox(label="全文（日本語）", lines=5)
                all_english_output_mic = gr.Textbox(label="全文（英語）", lines=5)
            with gr.Row():
                history_output_mic = gr.Dataframe(label="履歴", headers=["Time", "Language Detection", "Japanese ASR", "English ASR"], datatype=["str", "str", "str", "str"])

            audio_input_mic.stream(process_audio, inputs=[audio_input_mic],
                                   outputs=[japanese_output_mic, english_output_mic, best_output_mic, all_japanese_output_mic, all_english_output_mic, history_output_mic])
            reset_button_mic.click(reset_recognizer,
                                   outputs=[japanese_output_mic, english_output_mic, best_output_mic, all_japanese_output_mic, all_english_output_mic, history_output_mic])

        with gr.TabItem("File"):
            with gr.Row():
                audio_input_file = gr.Audio(sources=["upload"], waveform_options={"sample_rate": SAMPLE_RATE})
                reset_button_file = gr.Button("リセット")
            with gr.Row():
                japanese_output_file = gr.HTML(label="日本語文字起こし")
                english_output_file = gr.HTML(label="英語文字起こし")
            with gr.Row():
                best_output_file = gr.HTML(label="ベスト文字起こし結果")
            with gr.Row():
                all_japanese_output_file = gr.Textbox(label="全文（日本語）", lines=5)
                all_english_output_file = gr.Textbox(label="全文（英語）", lines=5)
            with gr.Row():
                history_output_file = gr.Dataframe(label="履歴", headers=["Time", "Language Detection", "Japanese ASR", "English ASR"], datatype=["str", "str", "str", "str"])

            audio_input_file.change(process_file, inputs=[audio_input_file],
                                    outputs=[japanese_output_file, english_output_file, best_output_file, all_japanese_output_file, all_english_output_file, history_output_file])
            reset_button_file.click(reset_recognizer,
                                    outputs=[japanese_output_file, english_output_file, best_output_file, all_japanese_output_file, all_english_output_file, history_output_file])

    demo.queue()
    demo.launch()

if __name__ == "__main__":
    launch_app()
