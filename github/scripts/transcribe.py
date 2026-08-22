"""
林森苑會議辨識字幕 — GitHub Actions 辨識腳本 (v2 — 準確度優化版)

相較前一版的改動：
  1. 重新啟用 VAD（語音活動偵測），並調參數 — 這是準確度/幻覺問題最大的關鍵
  2. 加入音訊前處理（ffmpeg 正規化音量、轉 16kHz mono、去除低頻噪音）
  3. initial_prompt 改為可自訂的社區專有名詞（請依實際狀況修改 MEETING_VOCABULARY）
  4. 加入 hotwords 參數，針對專有名詞做解碼偏置（不影響整體語氣，只加強特定詞彙辨識率）
  5. 加入 hallucination_silence_threshold，降低長靜音後亂講話的機率

需要的環境變數（由 GitHub Actions workflow 傳入）：
  TASK_ID                      任務 ID
  DRIVE_FILE_ID                音訊檔在 Drive 上的檔案 ID
  FILE_NAME                    原始檔名（可選，用於逐字稿命名）
  GOOGLE_SERVICE_ACCOUNT_JSON  Service Account 金鑰 JSON 內容（整段字串）
  TRANSCRIPT_FOLDER_ID         逐字稿要上傳到的 Drive 資料夾 ID
  APPS_SCRIPT_URL              Apps Script Web App 的網址
"""

import os
import io
import json
import sys
import subprocess
import requests
from datetime import timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

TASK_ID = os.environ["TASK_ID"]
DRIVE_FILE_ID = os.environ["DRIVE_FILE_ID"]
FILE_NAME = os.environ.get("FILE_NAME") or "meeting_audio"
TRANSCRIPT_FOLDER_ID = os.environ["TRANSCRIPT_FOLDER_ID"]
APPS_SCRIPT_URL = os.environ["APPS_SCRIPT_URL"]

# ---------------------------------------------------------------------------
# 請依實際會議內容修改：常出現的人名、職稱、大樓/社區用語
# 放在這裡可以提升 Whisper 對這些詞的辨識準確度（尤其是同音異字容易錯的專有名詞）
# ---------------------------------------------------------------------------
MEETING_VOCABULARY = [
    "林森苑", "管理委員會", "主任委員", "總幹事", "財務委員",
    "區分所有權人", "管理費", "公共基金", "臨時動議",
    # ← 在這裡加入實際會用到的人名、廠商名稱等
]

HALLUCINATION_KEYWORDS = [
    "Amara", "amara", "字幕組", "社群提供", "Bilibili", "bilibili",
    "訂閱", "按讚", "頻道", "詞曲", "作詞", "作曲", "編曲", "演唱", "提供字幕"
]


def get_drive_service():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    if isinstance(info, dict) and "private_key" in info:
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def download_audio(drive, file_id, dest_path):
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"下載進度: {int(status.progress() * 100)}%")


def preprocess_audio(input_path, output_path):
    """
    音訊前處理：
      - 轉成 16kHz 單聲道（Whisper 訓練時使用的規格）
      - loudnorm 音量正規化（解決忽大忽小的問題）
      - highpass 過濾掉 80Hz 以下的低頻噪音（空調、麥克風悶音）
    """
    print("音訊前處理中（正規化音量 / 去噪 / 轉換取樣率）…")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1",
        "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("音訊前處理失敗，改用原始檔案繼續：", result.stderr[-500:])
        return input_path
    return output_path


def format_timestamp(seconds):
    td = timedelta(seconds=int(seconds))
    return str(td) if td >= timedelta(hours=1) else "0:" + str(td)


def run_transcription(audio_path):
    from faster_whisper import WhisperModel
    from opencc import OpenCC

    print("載入 Whisper 模型 (large-v3) 中…")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    initial_prompt = (
        "以下是社區管理委員會會議的繁體中文逐字稿，包含正確標點符號與台灣慣用語。"
        "常見詞彙：" + "、".join(MEETING_VOCABULARY)
    )

    print("開始辨識…")
    segments, info = model.transcribe(
        audio_path,
        language="zh",
        initial_prompt=initial_prompt,
        hotwords="、".join(MEETING_VOCABULARY),  # 針對專有名詞做解碼偏置
        beam_size=5,
        best_of=5,
        condition_on_previous_text=False,  # 避免錯誤在片段間連鎖擴散
        vad_filter=True,                    # ★ 關鍵修正：重新啟用語音活動偵測
        vad_parameters=dict(
            min_silence_duration_ms=500,    # 靜音超過 0.5 秒才切段，避免把正常停頓切碎
            speech_pad_ms=200                # 語音前後各保留 200ms，避免斷字
        ),
        hallucination_silence_threshold=2.0  # 靜音超過 2 秒的區間，抑制幻覺文字
    )

    cc = OpenCC("s2twp")  # 簡體 → 台灣正體（含慣用詞轉換）

    lines = []
    for seg in segments:
        ts = format_timestamp(seg.start)
        raw_text = seg.text.strip()
        if not raw_text:
            continue
        text_tc = cc.convert(raw_text)

        if any(keyword in text_tc for keyword in HALLUCINATION_KEYWORDS):
            continue

        lines.append(f"{ts} {text_tc}")
        print(f"{ts} {text_tc}")

    return "\n".join(lines)


def upload_transcript(drive, text, base_name):
    file_metadata = {
        "name": f"{base_name}_逐字稿.txt",
        "parents": [TRANSCRIPT_FOLDER_ID]
    }
    media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")), mimetype="text/plain")
    file = drive.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return file["id"]


def delete_audio(drive, file_id):
    try:
        drive.files().delete(fileId=file_id).execute()
        print("已刪除原始音訊檔")
    except Exception as e:
        print(f"刪除音訊檔失敗（可忽略，不影響逐字稿結果）: {e}")


def notify_callback(status, transcript_text=None):
    payload = {"taskId": TASK_ID, "status": status}
    if transcript_text:
        payload["transcriptText"] = transcript_text
    resp = requests.post(APPS_SCRIPT_URL, json=payload, timeout=120)
    print("回報狀態:", resp.status_code, resp.text[:200])


def main():
    drive = get_drive_service()
    raw_path = "/tmp/audio_input"
    processed_path = "/tmp/audio_processed.wav"

    try:
        print("下載音訊中…")
        download_audio(drive, DRIVE_FILE_ID, raw_path)

        audio_for_transcribe = preprocess_audio(raw_path, processed_path)

        transcript_text = run_transcription(audio_for_transcribe)

        print("刪除原始音訊檔…")
        delete_audio(drive, DRIVE_FILE_ID)

        print("回報逐字稿結果給 Apps Script…")
        notify_callback("done", transcript_text=transcript_text)
        print("完成！")

    except Exception as e:
        print(f"發生錯誤: {e}", file=sys.stderr)
        notify_callback("error")
        sys.exit(1)


if __name__ == "__main__":
    main()
