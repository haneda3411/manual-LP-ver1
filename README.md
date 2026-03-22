# 動画マニュアル自動生成ツール

YouTube限定公開URLを入力するだけで、NotionのレシピマニュアルページをAIが自動生成するツール。

## 処理フロー

```
YouTube URL → yt-dlp (音声取得) → Whisper API (文字起こし)
→ Claude API (レシピ抽出) → プレビュー・編集 → Notion API (ページ作成)
```

## セットアップ

### 1. 必要なシステムツールをインストール

```bash
# yt-dlp
pip install yt-dlp

# ffmpeg (yt-dlpの音声変換に必要)
# Ubuntu/Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

### 2. Python依存パッケージをインストール

```bash
cd backend
pip install -r requirements.txt
```

### 3. 環境変数を設定

```bash
cp backend/.env.example backend/.env
```

`backend/.env` を開いて以下を設定：

| 変数名 | 説明 |
|--------|------|
| `ANTHROPIC_API_KEY` | Anthropic APIキー |
| `OPENAI_API_KEY` | OpenAI APIキー（Whisper用） |
| `NOTION_TOKEN` | Notion インテグレーショントークン |
| `NOTION_DATABASE_ID` | NotionデータベースのID |

#### Notionの準備

1. [Notion Integrations](https://www.notion.so/my-integrations) でインテグレーションを作成
2. 対象のNotionデータベースページを開き、右上の「…」→「コネクト」からインテグレーションを追加
3. データベースURLの末尾のID（32文字の英数字）を `NOTION_DATABASE_ID` に設定

データベースには以下のプロパティを作成しておくと完全に活用できます：
- `Name`（タイトル）：レシピ名
- `カテゴリ`（マルチセレクト）：料理ジャンル

### 4. バックエンドサーバーを起動

```bash
cd backend
python app.py
```

サーバーが `http://localhost:5000` で起動します。

### 5. フロントエンドを開く

ブラウザで `tool.html` を開く（ダブルクリック or `file://` でOK）。

## 使い方

1. `tool.html` をブラウザで開く
2. YouTube限定公開URLをフォームに貼り付け
3. 「解析開始」ボタンをクリック（1〜2分かかります）
4. 抽出されたレシピ内容を確認・修正
5. 「Notionページを作成」をクリック
6. 表示されたNotionページURLを確認

## ファイル構成

```
/
├── tool.html          # フロントエンド（ブラウザで開く）
├── backend/
│   ├── app.py         # Flask APIサーバー
│   ├── requirements.txt
│   └── .env.example   # 環境変数テンプレート
└── README.md
```

## APIコスト目安（10分動画の場合）

| API | コスト目安 |
|-----|-----------|
| OpenAI Whisper | ~$0.06 / 10分 |
| Claude API | ~$0.01〜$0.05 |
| Notion API | 無料 |

## 注意事項

- YouTube動画は「限定公開」設定が必要（完全非公開は音声取得不可）
- 動画は日本語を前提
- 1動画10分以内を推奨（長時間はAPIコスト増）
