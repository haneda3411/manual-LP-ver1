import os
import json
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import base64
import requests as http_requests
import anthropic
from notion_client import Client as NotionClient

load_dotenv()

app = Flask(__name__)
CORS(app)

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
notion_client = NotionClient(auth=os.environ.get("NOTION_TOKEN"))

NOTION_DB_MAP = {
    "刺し場調理マニュアル": os.environ.get("NOTION_DB_SASHIBA_CHORI"),
    "刺し場仕込みマニュアル": os.environ.get("NOTION_DB_SASHIBA_SHIKOMI"),
    "焼き場調理マニュアル": os.environ.get("NOTION_DB_YAKIBA_CHORI"),
    "焼き場仕込みマニュアル": os.environ.get("NOTION_DB_YAKIBA_SHIKOMI"),
    "ドリンク作成マニュアル": os.environ.get("NOTION_DB_DRINK"),
}


def download_audio(youtube_url: str, output_dir: str) -> str:
    """yt-dlpを使ってYouTube動画の音声をmp3でダウンロードする"""
    output_path = os.path.join(output_dir, "audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--output", output_path,
        "--no-playlist",
        youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")

    mp3_file = os.path.join(output_dir, "audio.mp3")
    if not os.path.exists(mp3_file):
        # 拡張子が変わる場合を探す
        files = list(Path(output_dir).glob("audio.*"))
        if not files:
            raise RuntimeError("音声ファイルの取得に失敗しました")
        mp3_file = str(files[0])
    return mp3_file


def transcribe_audio(audio_path: str) -> str:
    """Gemini REST APIで音声をテキストに変換する"""
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"text": "この音声を日本語でそのまま文字起こししてください。話されている内容を忠実にテキストにしてください。"},
                {"inline_data": {"mime_type": "audio/mpeg", "data": audio_b64}},
            ]
        }]
    }
    resp = http_requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def extract_recipe_info(transcript: str) -> dict:
    """Claude APIを使って文字起こしテキストからレシピ情報を構造化抽出する"""
    system_prompt = """あなたは料理レシピ抽出の専門AIです。
提供された文字起こしテキストから、以下のJSON形式でレシピ情報を抽出してください。

{
  "recipe_name": "レシピ名",
  "ingredients": [
    {"name": "材料名", "amount": "分量"}
  ],
  "steps": [
    "手順1",
    "手順2"
  ],
  "memo": "ポイントや注意事項（任意）",
  "category": "料理ジャンル（例：揚げ物、煮物、炒め物など）"
}

情報が不明な場合は空文字または空配列を使用してください。
必ずJSON形式のみで返答してください。余分なテキストは含めないでください。"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"以下の料理動画の文字起こしテキストからレシピ情報を抽出してください:\n\n{transcript}",
            }
        ],
        system=system_prompt,
    )

    content = message.content[0].text.strip()
    # JSONブロックがある場合は抽出
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        content = json_match.group(1).strip()

    return json.loads(content)


def create_notion_page(recipe: dict, youtube_url: str, database_id: str) -> dict:
    """Notion APIを使ってレシピページを作成する"""
    children = []

    # 動画リンク
    children.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": "動画リンク: ", "link": None},
                    "annotations": {"bold": True},
                },
                {
                    "type": "text",
                    "text": {"content": youtube_url, "link": {"url": youtube_url}},
                },
            ]
        },
    })

    # 材料セクション
    children.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "材料・分量"}}]
        },
    })
    for ingredient in recipe.get("ingredients", []):
        name = ingredient.get("name", "")
        amount = ingredient.get("amount", "")
        text = f"{name}　{amount}".strip()
        children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            },
        })

    # 調理手順セクション
    children.append({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [{"type": "text", "text": {"content": "調理手順"}}]
        },
    })
    for step in recipe.get("steps", []):
        children.append({
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [{"type": "text", "text": {"content": step}}]
            },
        })

    # メモセクション
    if recipe.get("memo"):
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "メモ・ポイント"}}]
            },
        })
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": recipe["memo"]}}]
            },
        })

    # ページ作成
    page_data = {
        "parent": {"database_id": database_id},
        "properties": {
            "Name": {
                "title": [
                    {"text": {"content": recipe.get("recipe_name", "レシピ")}}
                ]
            },
        },
        "children": children,
    }

    response = notion_client.pages.create(**page_data)
    return response


@app.route("/api/process", methods=["POST"])
def process_video():
    """YouTube URLから音声取得→文字起こし→レシピ抽出を行う"""
    data = request.get_json()
    youtube_url = data.get("url", "").strip()

    if not youtube_url:
        return jsonify({"error": "URLが入力されていません"}), 400

    # 簡易URL検証
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        return jsonify({"error": "有効なYouTube URLを入力してください"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Step 1: 音声ダウンロード
            audio_path = download_audio(youtube_url, tmpdir)
        except Exception as e:
            return jsonify({"error": f"音声取得エラー: {str(e)}"}), 500

        try:
            # Step 2: 文字起こし
            transcript = transcribe_audio(audio_path)
        except Exception as e:
            return jsonify({"error": f"文字起こしエラー: {str(e)}"}), 500

    try:
        # Step 3: レシピ抽出
        recipe = extract_recipe_info(transcript)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"レシピ抽出エラー（JSON解析失敗）: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"レシピ抽出エラー: {str(e)}"}), 500

    return jsonify({
        "transcript": transcript,
        "recipe": recipe,
    })


@app.route("/api/create-notion", methods=["POST"])
def create_notion():
    """抽出済みレシピデータをNotionページとして作成する"""
    data = request.get_json()
    recipe = data.get("recipe")
    youtube_url = data.get("url", "").strip()
    category = data.get("category", "").strip()

    if not recipe:
        return jsonify({"error": "レシピデータがありません"}), 400

    database_id = NOTION_DB_MAP.get(category)
    if not database_id:
        return jsonify({"error": f"カテゴリ「{category}」のデータベースIDが設定されていません"}), 500

    try:
        page = create_notion_page(recipe, youtube_url, database_id)
        page_url = page.get("url", "")
        return jsonify({"notion_url": page_url})
    except Exception as e:
        return jsonify({"error": f"Notionページ作成エラー: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
