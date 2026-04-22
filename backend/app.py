import os
import json
import re
import subprocess
import tempfile
import uuid
import base64
from pathlib import Path

import requests
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
    "ランチ調理マニュアル": os.environ.get("NOTION_DB_LUNCH_CHORI"),
    "ランチ仕込みマニュアル": os.environ.get("NOTION_DB_LUNCH_SHIKOMI"),
}


def download_video(youtube_url: str, output_dir: str) -> str:
    """yt-dlpでYouTube動画をダウンロードする（720p以下）"""
    output_path = os.path.join(output_dir, "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--no-playlist",
        youtube_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {result.stderr}")
    files = list(Path(output_dir).glob("video.*"))
    if not files:
        raise RuntimeError("動画ファイルの取得に失敗しました")
    return str(files[0])


def extract_audio_from_video(video_path: str, output_dir: str) -> str:
    """ffmpegで動画から音声をmp3として抽出する"""
    audio_path = os.path.join(output_dir, "audio.mp3")
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "4", "-y", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"音声抽出エラー: {result.stderr}")
    return audio_path


def transcribe_audio_with_timestamps(audio_path: str) -> tuple:
    """OpenAI Whisper APIで音声をテキスト+セグメントタイムスタンプに変換する"""
    with open(audio_path, "rb") as f:
        response = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ja",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    transcript = response.text
    segments = [{"start": float(s.start), "end": float(s.end), "text": s.text}
                for s in (response.segments or [])]
    return transcript, segments


def map_steps_to_timestamps(steps: list, segments: list) -> list:
    """Claudeを使って各工程のタイムスタンプ範囲を特定する"""
    if not segments:
        return [{"step": i + 1, "start": None, "end": None} for i in range(len(steps))]

    segments_text = "\n".join([f"[{s['start']:.1f}s-{s['end']:.1f}s]: {s['text']}" for s in segments])
    steps_text = "\n".join([f"{i+1}. {s['title']}" for i, s in enumerate(steps)])

    prompt = f"""調理動画の文字起こしセグメント（タイムスタンプ付き）と工程リストを照合し、
各工程が動画の何秒〜何秒に対応するかをJSON配列で返してください。

工程リスト:
{steps_text}

文字起こしセグメント:
{segments_text}

JSON形式のみで返答（対応不明な場合はnull）:
[{{"step": 1, "start": 10.5, "end": 45.2}}, ...]"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    content = message.content[0].text.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        content = json_match.group(1).strip()
    return json.loads(content)


def select_best_frames_with_vision(step_title: str, frame_paths: list, n: int = 3) -> list:
    """Claude Visionで最も工程を表すフレームをn枚選択する"""
    if len(frame_paths) <= n:
        return frame_paths

    content = [{
        "type": "text",
        "text": (f"以下は調理動画の「{step_title}」という工程を時系列順に撮影したフレームです（番号1〜{len(frame_paths)}）。"
                 f"この工程の内容を最もよく表している上位{n}枚のフレーム番号を、カンマ区切りで返してください。例: 2,5,8")
    }]
    for i, path in enumerate(frame_paths):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "text", "text": f"フレーム{i + 1}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=20,
        messages=[{"role": "user", "content": content}],
    )
    result = message.content[0].text.strip()
    selected = []
    seen = set()
    for part in re.split(r"[,、\s]+", result):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(frame_paths) and idx not in seen:
                seen.add(idx)
                selected.append(frame_paths[idx])
        except ValueError:
            pass
    # 足りなければ未選択から補完
    for i, path in enumerate(frame_paths):
        if len(selected) >= n:
            break
        if i not in seen:
            selected.append(path)
    return selected[:n]


def get_video_duration(video_path: str) -> float:
    """ffprobeで動画の長さ（秒）を取得する"""
    cmd = ["ffprobe", "-v", "quiet", "-of", "csv=p=0",
           "-show_entries", "format=duration", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def extract_frames_as_base64(video_path: str, step_timestamps: list, output_dir: str, steps: list = None) -> list:
    """各工程からフレームを抽出しbase64データURLで返す（外部サービス不要）"""
    # タイムスタンプがnullの工程のためにフォールバック用の動画長を取得
    duration_total = get_video_duration(video_path)
    n_steps = len(step_timestamps)

    results = []
    for idx, item in enumerate(step_timestamps):
        step_num = item.get("step", idx + 1)
        start = item.get("start")
        end = item.get("end")

        # タイムスタンプがnullの場合は動画を均等分割してフォールバック
        if (start is None or end is None or end <= start) and duration_total > 0:
            seg = duration_total / max(n_steps, 1)
            start = seg * idx
            end = seg * (idx + 1)

        if start is None or end is None or end <= start:
            results.append({"step": step_num, "frame_candidates": []})
            continue

        duration = end - start
        interval = max(2.0, duration / 10)
        timestamps = []
        t = start + interval * 0.5
        while t < end and len(timestamps) < 10:
            timestamps.append(t)
            t += interval

        frame_paths = []
        for j, ts in enumerate(timestamps):
            frame_path = os.path.join(output_dir, f"step{step_num}_frame{j}.jpg")
            # 480x270サムネイルとして抽出（軽量・高速）
            cmd = ["ffmpeg", "-ss", f"{ts:.2f}", "-i", video_path,
                   "-frames:v", "1", "-vf", "scale=480:270", "-q:v", "5", "-y", frame_path]
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode == 0 and os.path.exists(frame_path):
                frame_paths.append(frame_path)

        # Claude Visionで上位3枚を選択
        step_title = ""
        if steps and step_num <= len(steps):
            step_title = steps[step_num - 1].get("title", "")
        try:
            best_paths = select_best_frames_with_vision(step_title, frame_paths, n=3)
        except Exception:
            best_paths = frame_paths[:3]

        # base64データURLに変換（外部アップロード不要）
        candidates = []
        for path in best_paths:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                candidates.append(f"data:image/jpeg;base64,{b64}")
            except Exception:
                pass

        results.append({"step": step_num, "frame_candidates": candidates})
    return results


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


def upload_image_to_public(image_bytes: bytes, content_type: str) -> str:
    """画像を公開URLにアップロード。catbox.moe → litterbox → imgbbの順で試みる"""
    import time
    ext = content_type.split("/")[-1].replace("jpeg", "jpg")
    filename = f"manual_{uuid.uuid4().hex[:8]}.{ext}"

    # 1. catbox.moe（リトライ付き）
    for attempt in range(3):
        try:
            r = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": (filename, image_bytes, content_type)},
                timeout=30,
            )
            url = r.text.strip()
            if r.ok and url.startswith("http"):
                return url
        except Exception:
            pass
        if attempt < 2:
            time.sleep(2 ** attempt)

    # 2. litterbox.catbox.moe（1時間保持）
    try:
        r = requests.post(
            "https://litterbox.catbox.moe/resources/internals/api.php",
            data={"reqtype": "fileupload", "time": "1h"},
            files={"fileToUpload": (filename, image_bytes, content_type)},
            timeout=30,
        )
        url = r.text.strip()
        if r.ok and url.startswith("http"):
            return url
    except Exception:
        pass

    # 3. tmpfiles.org
    try:
        r = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": (filename, image_bytes, content_type)},
            timeout=30,
        )
        if r.ok:
            data = r.json()
            raw_url = data.get("data", {}).get("url", "")
            # tmpfiles.org のURLを直接ダウンロード用に変換
            url = raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
            if url.startswith("http"):
                return url
    except Exception:
        pass

    raise RuntimeError("全ての画像アップロードサービスが利用できませんでした")


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


def _callout_with_bullets(heading, items, color="yellow_background", icon="💡"):
    """items: [(label, text), ...] として箇条書きをcallout内に表示"""
    bullet_children = [
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [
                    {"type": "text", "text": {"content": label}, "annotations": {"bold": True}},
                    {"type": "text", "text": {"content": f"：{text}"}},
                ]
            },
        }
        for label, text in items
    ]
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": heading}, "annotations": {"bold": True}}],
            "color": color,
            "icon": {"type": "emoji", "emoji": icon},
        },
        "children": bullet_children,
    }


def _bullet(parts):
    """parts: [(text, bold), ...]"""
    rich = [{"type": "text", "text": {"content": t}, "annotations": {"bold": b}} for t, b in parts]
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rich}}


def _image(url):
    return {
        "object": "block",
        "type": "image",
        "image": {"type": "external", "external": {"url": url}},
    }


def _toggle(title, children):
    return {
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": [{"type": "text", "text": {"content": title}}],
            "children": children,
        },
    }


def get_select_properties(database_id: str) -> list:
    """DBのselect/multi_select/statusプロパティ一覧を返す [{name, type, options:[{name,color}]}]"""
    db = notion_client.databases.retrieve(database_id=database_id)
    result = []
    for prop_name, prop_data in db["properties"].items():
        if prop_data["type"] == "select":
            options = [{"name": o["name"], "color": o.get("color", "")}
                       for o in prop_data["select"].get("options", [])]
            result.append({"name": prop_name, "type": "select", "options": options})
        elif prop_data["type"] == "multi_select":
            options = [{"name": o["name"], "color": o.get("color", "")}
                       for o in prop_data["multi_select"].get("options", [])]
            result.append({"name": prop_name, "type": "multi_select", "options": options})
        elif prop_data["type"] == "status":
            options = [{"name": o["name"], "color": o.get("color", "")}
                       for o in prop_data["status"].get("options", [])]
            result.append({"name": prop_name, "type": "status", "options": options})
    return result


def create_notion_page(recipe: dict, youtube_url: str, database_id: str, image_url: str = None, extra_props: dict = None) -> dict:
    """Notion APIを使って構造化マニュアルページを作成する"""
    children = []
    materials = recipe.get("materials", {})
    steps = recipe.get("steps", [])
    video_title = recipe.get("video_title") or recipe.get("recipe_name", "")

    # ① 完成盛り付け
    children.append(_para("[完成盛り付け]", bold=True))
    if image_url:
        children.append(_image(image_url))
    else:
        children.append(_callout("📷 完成写真", "ここに完成写真を追加してください\n盛り付けポイント：（写真追加後に記入）", color="yellow_background", icon="📷"))

    # ② ポイント・注意点まとめ（まとめに含める選択がある場合はそちらを優先）
    all_points   = [s["point"]   for s in steps if s.get("point")   and s.get("point_in_summary", True)]
    all_cautions = [s["caution"] for s in steps if s.get("caution") and s.get("caution_in_summary", True)]
    if all_points or all_cautions:
        children.append(_para("[ポイント・注意点まとめ]", bold=True))
        if all_points:
            body = "\n\n".join([f"• {text}" for text in all_points])
            children.append(_callout("💡 ポイントまとめ", body, color="yellow_background", icon="💡"))
        if all_cautions:
            body = "\n\n".join([f"• {text}" for text in all_cautions])
            children.append(_callout("⚠️ 注意点まとめ", body, color="yellow_background", icon="⚠️"))

    # ③ 材料
    children.append(_para("[材料]", bold=True))
    for d in materials.get("details", []):
        children.append({"object": "block", "type": "bulleted_list_item",
                         "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": d}}]}})

    # ③ 手順
    children.append(_para("[手順]", bold=True))
    for i, step in enumerate(steps, 1):
        children.append(_para(f"工程{i}: {step.get('title', '')}", bold=True))
        if step.get("caution"):
            children.append(_callout("⚠️ 注意点", step["caution"], color="yellow_background", icon="⚠️"))
        if step.get("point"):
            children.append(_callout("💡 ポイント", step["point"], color="yellow_background", icon="💡"))
        if step.get("selected_frame_url"):
            children.append(_image(step["selected_frame_url"]))
        else:
            children.append(_para("（ここに写真を追加）"))

    # ④ 動画マニュアルフッター
    children.append(_para("動画マニュアルはこちら", bold=True))
    if youtube_url:
        link_text = f"▶ {video_title}" if video_title else f"▶ {youtube_url}"
        children.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": link_text, "link": {"url": youtube_url}}}],
                "color": "gray_background",
                "icon": {"type": "emoji", "emoji": "▶️"},
            },
        })

    title_prop = get_title_property_name(database_id)
    properties = {
        title_prop: {"title": [{"text": {"content": recipe.get("recipe_name", "マニュアル")}}]},
    }
    # 追加プロパティ（種類など）をセット
    if extra_props:
        for prop_name, prop_value in extra_props.items():
            ptype = prop_value.get("type")
            if ptype == "select":
                properties[prop_name] = {"select": {"name": prop_value["value"]}}
            elif ptype == "multi_select":
                properties[prop_name] = {"multi_select": [{"name": v} for v in prop_value["value"]]}
            elif ptype == "status":
                properties[prop_name] = {"status": {"name": prop_value["value"]}}

    page_data = {
        "parent": {"database_id": database_id},
        "properties": properties,
        "children": children,
    }
    return notion_client.pages.create(**page_data)


@app.route("/api/process", methods=["POST"])
def process_video():
    """YouTube URLから動画取得→文字起こし→レシピ抽出→フレーム候補を行う"""
    data = request.get_json()
    youtube_url = data.get("url", "").strip()

    if not youtube_url:
        return jsonify({"error": "URLが入力されていません"}), 400

    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        return jsonify({"error": "有効なYouTube URLを入力してください"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            video_path = download_video(youtube_url, tmpdir)
        except Exception as e:
            return jsonify({"error": f"動画取得エラー: {str(e)}"}), 500

        try:
            audio_path = extract_audio_from_video(video_path, tmpdir)
        except Exception as e:
            return jsonify({"error": f"音声抽出エラー: {str(e)}"}), 500

        try:
            transcript, segments = transcribe_audio_with_timestamps(audio_path)
        except Exception as e:
            return jsonify({"error": f"文字起こしエラー: {str(e)}"}), 500

        try:
            recipe = extract_recipe_info(transcript)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"レシピ抽出エラー（JSON解析失敗）: {str(e)}"}), 500
        except Exception as e:
            return jsonify({"error": f"レシピ抽出エラー: {str(e)}"}), 500

        # フレーム候補抽出（失敗しても処理継続）
        try:
            step_timestamps = map_steps_to_timestamps(recipe.get("steps", []), segments)
            frame_results = extract_frames_as_base64(video_path, step_timestamps, tmpdir, steps=recipe.get("steps", []))
            frame_map = {f["step"]: f["frame_candidates"] for f in frame_results}
            for i, step in enumerate(recipe.get("steps", []), 1):
                step["frame_candidates"] = frame_map.get(i, [])
        except Exception as e:
            print(f"[WARN] フレーム抽出エラー: {e}")
            for step in recipe.get("steps", []):
                step.setdefault("frame_candidates", [])

    return jsonify({
        "transcript": transcript,
        "recipe": recipe,
    })


@app.route("/api/process-text", methods=["POST"])
def process_text():
    """テキスト入力からレシピ抽出（動画なし）"""
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "テキストが入力されていません"}), 400

    try:
        recipe = extract_recipe_info(text)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"レシピ抽出エラー（JSON解析失敗）: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"レシピ抽出エラー: {str(e)}"}), 500

    for step in recipe.get("steps", []):
        step.setdefault("frame_candidates", [])

    return jsonify({"transcript": text, "recipe": recipe})


@app.route("/api/db-properties", methods=["GET"])
def db_properties():
    """カテゴリのDBが持つselectプロパティと選択肢を返す"""
    category = request.args.get("category", "").strip()
    database_id = NOTION_DB_MAP.get(category)
    if not database_id:
        return jsonify({"error": f"カテゴリ「{category}」が見つかりません"}), 404
    try:
        props = get_select_properties(database_id)
        return jsonify({"properties": props})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/create-notion", methods=["POST"])
def create_notion():
    """抽出済みレシピデータをNotionページとして作成する"""
    data = request.get_json()
    recipe = data.get("recipe")
    youtube_url = data.get("url", "").strip()
    category = data.get("category", "").strip()
    extra_props = data.get("extra_props")  # {prop_name: {type, value}}
    image_b64 = data.get("image")       # base64文字列（任意）
    image_type = data.get("image_type") # MIMEタイプ（例: image/jpeg）

    if not recipe:
        return jsonify({"error": "レシピデータがありません"}), 400

    database_id = NOTION_DB_MAP.get(category)
    if not database_id:
        return jsonify({"error": f"カテゴリ「{category}」のデータベースIDが設定されていません"}), 500

    # 完成写真アップロード（あれば）
    image_url = None
    if image_b64 and image_type:
        try:
            image_bytes = base64.b64decode(image_b64)
            image_url = upload_image_to_public(image_bytes, image_type)
        except Exception as e:
            return jsonify({"error": f"画像アップロードエラー: {str(e)}"}), 500

    # 各工程の手動アップロード画像を処理
    for step in recipe.get("steps", []):
        manual_b64 = step.pop("manual_image_data", None)
        manual_type = step.pop("manual_image_type", None)
        if manual_b64 and manual_type and not step.get("selected_frame_url"):
            try:
                step["selected_frame_url"] = upload_image_to_public(
                    base64.b64decode(manual_b64), manual_type
                )
            except Exception:
                pass

    try:
        page = create_notion_page(recipe, youtube_url, database_id, image_url=image_url, extra_props=extra_props)
        page_url = page.get("url", "")
        return jsonify({"notion_url": page_url})
    except Exception as e:
        return jsonify({"error": f"Notionページ作成エラー: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
