"""
林森苑會議辨識字幕 — GitHub Actions 辨識腳本

流程：
  1. 用 Service Account 從 Google Drive 下載音訊檔
  2. 用 faster-whisper 做語音辨識（中文）
  3. 把逐字稿（含時間戳記）存成 txt，上傳回 Google Drive
  4. 刪除原始音訊檔（隱私考量）
  5. 呼叫 Apps Script Web App 回報完成狀態

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

    print("載入模型中…")
    model = WhisperModel("medium", device="cpu", compute_type="int8")

    print("開始辨識…")
    segments, info = model.transcribe(audio_path, language="zh", vad_filter=True)

    lines = []
    for seg in segments:
        ts = format_timestamp(seg.start)
        lines.append(f"{ts} {seg.text.strip()}")
        print(f"{ts} {seg.text.strip()}")  # 順便印在 Actions log 方便除錯

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
