"""
林森苑會議辨識字幕 — GitHub Actions 辨識腳本 (Large-v3 高精準度版)

流程：
  1. 用 Service Account 從 Google Drive 下載音訊檔
  2. 用 OpenAI large-v3 模型做高精準度語音辨識（繁體中文）
  3. 使用 OpenCC 自動轉為臺灣正體中文
  4. 刪除原始音訊檔（隱私考量）
  5. 呼叫 Apps Script Web App 回報完成狀態與字幕內容
"""

import os
import io
import json
import sys
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


def format_timestamp(seconds):
    td = timedelta(seconds=int(seconds))
    return str(td) if td >= timedelta(hours=1) else "0:" + str(td)


def run_transcription(audio_path):
    from faster_whisper import WhisperModel
    from opencc import OpenCC

    print("載入頂級 Whisper 模型 (large-v3) 中…")
    # 使用 OpenAI 最高精準度的 large-v3 模型
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    print("開始高精準度辨識…")
    # 提示模型優先輸出繁體中文，啟用集束搜尋 (beam_size=5) 提高正確率
    segments, info = model.transcribe(
        audio_path,
        language="zh",
        initial_prompt="以下是繁體中文語音辨識字幕逐字稿，包含精確的標點符號與台灣習慣用語：",
        beam_size=5,
        best_of=5,
        vad_filter=False,
        condition_on_previous_text=False
    )

    # 使用 OpenCC 將簡體中文無縫轉為臺灣正體中文 (含常用詞彙轉換)
    cc = OpenCC("s2twp")

    lines = []
    for seg in segments:
        ts = format_timestamp(seg.start)
        raw_text = seg.text.strip()
        text_tc = cc.convert(raw_text)
        lines.append(f"{ts} {text_tc}")
        print(f"{ts} {text_tc}")  # 順便印在 Actions log 方便除錯

    return "\n".join(lines)


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
    audio_path = "/tmp/audio_input"

    try:
        print("下載音訊中…")
        download_audio(drive, DRIVE_FILE_ID, audio_path)

        transcript_text = run_transcription(audio_path)

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
