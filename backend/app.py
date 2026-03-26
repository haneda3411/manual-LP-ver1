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
import anthropic
from openai import OpenAI
from notion_client import Client as NotionClient

load_dotenv()

app = Flask(__name__)
CORS(app)

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
    """OpenAI Whisper APIで音声をテキストに変換する"""
    with open(audio_path, "rb") as f:
        response = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ja",
        )
    return response.text


def extract_recipe_info(transcript: str) -> dict:
    """Claude APIを使って文字起こしテキストから構造化マニュアル情報を抽出する"""
    system_prompt = """あなたは調理マニュアル作成の専門AIです。
提供された動画の文字起こしテキストから、以下のJSON形式で情報を抽出してください。

{
  "recipe_name": "メニュー名",
  "materials": {
    "main": "主要食材の概要（例：エリンギ140g／にんにく1ヶ）",
    "seasoning": "調味料の概要（例：サラダ油・塩・バター・醤油）",
    "garnish": "薬味の概要（例：万能ねぎ）。なければ空文字",
    "details": ["食材名：分量", "食材名：分量（補足）"]
  },
  "steps": [
    {
      "title": "工程の短い説明（一文、動詞で終わる）",
      "point": "ポイント・コツのテキスト。なければnull",
      "caution": "注意点のテキスト。なければnull"
    }
  ],
  "video_title": "動画タイトル（例：【調理】メニュー名）"
}

ルール：
- stepsは工程ごとに分割する（目安3〜8工程）
- pointとcautionは現場で役立つ具体的な内容のみ。なければnull
- 情報が不明な場合は空文字またはnullを使用
- 必ずJSON形式のみで返答。余分なテキスト不可"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"以下の調理動画の文字起こしテキストからマニュアル情報を抽出してください:\n\n{transcript}",
            }
        ],
        system=system_prompt,
    )

    content = message.content[0].text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        content = json_match.group(1).strip()

    return json.loads(content)


def get_title_property_name(database_id: str) -> str:
    """データベースのタイトルプロパティ名を自動取得する"""
    db = notion_client.databases.retrieve(database_id=database_id)
    for prop_name, prop_data in db["properties"].items():
        if prop_data["type"] == "title":
            return prop_name
    return "名前"


def _para(text, bold=False):
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": text}, "annotations": {"bold": bold}}]
        },
    }


def _callout(heading, body, color="yellow_background", icon="💡"):
    rich = [{"type": "text", "text": {"content": heading}, "annotations": {"bold": True}}]
    if body:
        rich.append({"type": "text", "text": {"content": f"\n\n{body}"}})
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": rich,
            "color": color,
            "icon": {"type": "emoji", "emoji": icon},
        },
    }


def _bullet(parts):
    """parts: [(text, bold), ...]"""
    rich = [{"type": "text", "text": {"content": t}, "annotations": {"bold": b}} for t, b in parts]
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rich}}


def _toggle(title, children):
    return {
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": [{"type": "text", "text": {"content": title}}],
            "children": children,
        },
    }


def create_notion_page(recipe: dict, youtube_url: str, database_id: str) -> dict:
    """Notion APIを使って構造化マニュアルページを作成する"""
    children = []
    materials = recipe.get("materials", {})
    steps = recipe.get("steps", [])
    video_title = recipe.get("video_title") or recipe.get("recipe_name", "")

    # ① 完成盛り付け
    children.append(_para("[完成盛り付け]", bold=True))
    children.append(_callout("📷 完成写真", "ここに完成写真を追加してください\n盛り付けポイント：（写真追加後に記入）", color="yellow_background", icon="📷"))

    # ② 材料
    children.append(_para("[材料]", bold=True))
    if materials.get("main"):
        children.append(_bullet([("主要食材", True), ("：" + materials["main"], False)]))
    if materials.get("seasoning"):
        children.append(_bullet([("調味料", True), ("：" + materials["seasoning"], False)]))
    if materials.get("garnish"):
        children.append(_bullet([("薬味", True), ("：" + materials["garnish"], False)]))

    detail_list = materials.get("details", [])
    if detail_list:
        detail_blocks = [
            {"object": "block", "type": "bulleted_list_item",
             "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": d}}]}}
            for d in detail_list
        ]
        children.append(_toggle("材料（詳細量）", detail_blocks))

    # ③ 手順
    children.append(_para("[手順]", bold=True))
    for i, step in enumerate(steps, 1):
        step_children = []
        if step.get("caution"):
            step_children.append(_callout("⚠️ 注意点", step["caution"], color="yellow_background", icon="⚠️"))
        if step.get("point"):
            step_children.append(_callout("💡 ポイント", step["point"], color="yellow_background", icon="💡"))
        step_children.append(_para("（ここに写真を追加）"))
        children.append(_toggle(f"工程{i}: {step.get('title', '')}", step_children))

    # ④ 動画マニュアルフッター
    children.append(_para("動画マニュアルはこちら", bold=True))
    footer_body = f"{video_title}\n▶︎ {youtube_url}"
    children.append(_callout("▶️", footer_body, color="gray_background", icon="▶️"))

    title_prop = get_title_property_name(database_id)
    page_data = {
        "parent": {"database_id": database_id},
        "properties": {
            title_prop: {"title": [{"text": {"content": recipe.get("recipe_name", "マニュアル")}}]},
        },
        "children": children,
    }
    return notion_client.pages.create(**page_data)


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
