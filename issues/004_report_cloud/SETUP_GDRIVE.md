# Google Drive API セットアップガイド

このドキュメントでは、GitHub ActionsからGoogle Driveにアクセスするための設定手順を説明します。

**想定読者**: Google Cloud Platform初心者

**所要時間**: 約30-60分

---

## 概要

GitHub ActionsでGoogle Driveのファイルを読み取るために、以下を設定します:

1. Google Cloudプロジェクト作成
2. Google Drive API有効化
3. サービスアカウント作成
4. 認証キー（JSON）取得
5. Driveフォルダへのアクセス権限付与
6. GitHub Secretsへの認証情報保存

---

## Step 1: Google Cloud プロジェクト作成

### 1.1 Google Cloud Consoleにアクセス

1. ブラウザで https://console.cloud.google.com にアクセス
2. Googleアカウントでログイン（Mind Monitor用と同じアカウント推奨）

### 1.2 新規プロジェクト作成

1. 画面上部のプロジェクト選択ドロップダウンをクリック
2. 「新しいプロジェクト」ボタンをクリック
3. プロジェクト情報を入力:
   - **プロジェクト名**: `satoru-eeg-analysis` (例)
   - **組織**: なし（個人利用）
   - **場所**: 組織なし
4. 「作成」ボタンをクリック

**確認事項:**
- プロジェクトIDが自動生成される（例: `satoru-eeg-analysis-123456`）
- このIDは後で使わないが、念のためメモ

---

## Step 2: Google Drive API 有効化

### 2.1 APIライブラリに移動

1. 左上のハンバーガーメニュー（≡）をクリック
2. 「APIとサービス」→「ライブラリ」を選択

### 2.2 Drive API を検索・有効化

1. 検索ボックスに「Google Drive API」と入力
2. 「Google Drive API」をクリック
3. 「有効にする」ボタンをクリック

**確認事項:**
- ボタンが「管理」に変わる（有効化完了）

---

## Step 3: サービスアカウント作成

### サービスアカウントとは？

> ユーザーの代わりにプログラムがAPIを呼び出すための特別なアカウント。
> メールアドレス形式（例: `xxx@project.iam.gserviceaccount.com`）を持つ。

### 3.1 サービスアカウント作成画面へ

1. 左上のハンバーガーメニュー（≡）をクリック
2. 「APIとサービス」→「認証情報」を選択
3. 上部の「認証情報を作成」→「サービスアカウント」を選択

### 3.2 サービスアカウント詳細を入力

**ステップ1: サービスアカウントの詳細**

- **サービスアカウント名**: `github-actions-drive-reader` (例)
- **サービスアカウントID**: 自動生成される（例: `github-actions-drive-reader`）
- **説明**: `GitHub ActionsからGoogle Driveを読み取るアカウント` (任意)

「作成して続行」をクリック

**ステップ2: このサービスアカウントにプロジェクトへのアクセス権を付与する**

- **ロール**: なし（後でDriveフォルダレベルで権限付与）
- 「続行」をクリック

**ステップ3: ユーザーにこのサービスアカウントへのアクセス権を付与する**

- スキップ（入力なし）
- 「完了」をクリック

**確認事項:**
- サービスアカウントが作成され、メールアドレスが表示される
- 例: `github-actions-drive-reader@satoru-eeg-analysis-123456.iam.gserviceaccount.com`
- **このメールアドレスをコピーして保存**（後で使用）

---

## Step 4: 認証キー（JSON）のダウンロード

### 4.1 キー作成

1. 作成したサービスアカウントの行をクリック（サービスアカウント詳細画面へ）
2. 上部タブの「キー」を選択
3. 「鍵を追加」→「新しい鍵を作成」を選択
4. キーのタイプ: **JSON** を選択
5. 「作成」ボタンをクリック

**結果:**
- JSONファイルが自動ダウンロードされる
- ファイル名例: `satoru-eeg-analysis-123456-a1b2c3d4e5f6.json`

### 4.2 JSONファイルの内容確認

ダウンロードしたJSONファイルをテキストエディタで開くと、以下のような内容:

```json
{
  "type": "service_account",
  "project_id": "satoru-eeg-analysis-123456",
  "private_key_id": "a1b2c3d4e5f6...",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----\n",
  "client_email": "github-actions-drive-reader@satoru-eeg-analysis-123456.iam.gserviceaccount.com",
  "client_id": "123456789012345678901",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
```

**重要:**
- このファイルは秘密鍵を含むため、**絶対にGitにコミットしない**
- 安全な場所に保管（パスワードマネージャー等）

---

## Step 5: Google Driveフォルダへのアクセス権限付与

### 5.1 Mind Monitor用フォルダの確認

1. ブラウザでGoogle Drive (https://drive.google.com) にアクセス
2. Mind MonitorのCSVファイルが保存されているフォルダを開く
3. フォルダ名を確認（例: `MindMonitor`）

### 5.2 フォルダIDの取得

1. フォルダを開いた状態でURLを確認
2. URLの形式:
   ```
   https://drive.google.com/drive/folders/1aB2cD3eF4gH5iJ6kL7mN8oP9qR0sT1u
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                            これがフォルダID
   ```
3. **フォルダIDをコピーして保存**（後で使用）

### 5.3 サービスアカウントに共有権限を付与

1. フォルダを右クリック →「共有」を選択
2. 「ユーザーやグループを追加」欄に、Step 3.2でコピーした**サービスアカウントのメールアドレス**を貼り付け
   - 例: `github-actions-drive-reader@satoru-eeg-analysis-123456.iam.gserviceaccount.com`
3. 権限を「閲覧者」に設定（編集権限は不要）
4. 「通知を送信」のチェックを外す（サービスアカウントはメール受信しない）
5. 「送信」ボタンをクリック

**確認事項:**
- フォルダの共有設定にサービスアカウントが表示される
- 権限: 閲覧者

---

## Step 6: GitHub Secrets への認証情報保存

### 6.1 GitHub リポジトリのSettings へ

1. ブラウザで GitHubリポジトリ (`satoru`) にアクセス
2. 上部タブの「Settings」をクリック
3. 左サイドバーの「Secrets and variables」→「Actions」を選択

### 6.2 Secret 1: Google Drive 認証情報

1. 「New repository secret」ボタンをクリック
2. 以下を入力:
   - **Name**: `GDRIVE_CREDENTIALS`
   - **Secret**: Step 4でダウンロードしたJSONファイルの**全文**をコピー&ペースト
     - ファイルをテキストエディタで開く
     - 全選択（Ctrl+A / Cmd+A）→ コピー（Ctrl+C / Cmd+C）
     - Secretフィールドに貼り付け
3. 「Add secret」ボタンをクリック

### 6.3 Secret 2: Google Drive フォルダID

1. 再び「New repository secret」ボタンをクリック
2. 以下を入力:
   - **Name**: `GDRIVE_FOLDER_ID`
   - **Secret**: Step 5.2で取得した**フォルダID**を貼り付け
     - 例: `1aB2cD3eF4gH5iJ6kL7mN8oP9qR0sT1u`
3. 「Add secret」ボタンをクリック

**確認事項:**
- Secrets一覧に以下2つが表示される:
  - `GDRIVE_CREDENTIALS`
  - `GDRIVE_FOLDER_ID`
- Secretの内容は表示されない（セキュリティ保護）

---

## Step 7: 動作確認（オプション）

実装完了後、以下のPythonスクリプトでローカルテストが可能です:

### 7.1 ローカル環境でのテスト

```bash
# 仮想環境有効化
source venv/bin/activate

# 依存関係インストール
pip install google-auth google-api-python-client

# JSONキーを環境変数に設定（一時的）
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/downloaded-key.json"

# テストスクリプト実行
python scripts/fetch_from_gdrive.py --folder-id <YOUR_FOLDER_ID> --list
```

**期待される出力:**

```
Found 3 CSV files in folder:
1. 2025-11-04_morning.csv (Modified: 2025-11-04 09:30:15)
2. 2025-11-03_evening.csv (Modified: 2025-11-03 19:45:22)
3. 2025-11-02_morning.csv (Modified: 2025-11-02 08:12:33)
```

---

## トラブルシューティング

### エラー: "Access Not Configured"

**原因**: Drive APIが有効化されていない

**対処**:
- Step 2を再確認
- GCPコンソールで「APIとサービス」→「有効なAPI」にDrive APIが表示されるか確認

### エラー: "Insufficient Permission"

**原因**: サービスアカウントにフォルダ共有権限が付与されていない

**対処**:
- Step 5.3を再確認
- Driveフォルダの共有設定にサービスアカウントのメールアドレスがあるか確認

### エラー: "Invalid Credentials"

**原因**: JSONキーが正しくコピーされていない

**対処**:
- GitHub SecretsのGDRIVE_CREDENTIALSを削除して再作成
- JSONファイル全文（`{` から `}` まで）がコピーされているか確認
- 改行やスペースが欠けていないか確認

### エラー: "File Not Found" / "Folder Not Found"

**原因**: フォルダIDが間違っている

**対処**:
- DriveのURLを再確認
- フォルダIDは32文字程度の英数字文字列
- URLの`/folders/`以降の部分をコピー

---

## セキュリティのベストプラクティス

### 1. 認証情報の管理

- ✅ JSONキーはGitHub Secretsに保存
- ✅ ローカルのJSONファイルは安全な場所に保管
- ❌ JSONキーをGitにコミットしない
- ❌ JSONキーをSlack等で共有しない

### 2. 最小権限の原則

- ✅ サービスアカウントは「閲覧者」権限のみ
- ✅ 特定フォルダのみアクセス可能
- ❌ Drive全体へのアクセス権は付与しない

### 3. キーのローテーション

- 定期的（年1回）にサービスアカウントキーを再生成
- 古いキーは削除

### 4. アクセスログの確認

- GCPコンソールで定期的にAPIアクセスログを確認
- 不審なアクセスがないかチェック

---

## 次のステップ

セットアップ完了後:

1. [Phase 2](ISSUE.md#phase-2-google-drive連携スクリプト作成1-15時間) に進む
2. `scripts/fetch_from_gdrive.py` を実装
3. GitHub Actions ワークフローで実際に動作確認

---

## 参考リンク

- [Google Cloud Console](https://console.cloud.google.com)
- [Google Drive API Documentation](https://developers.google.com/drive/api/v3/about-sdk)
- [Python用Drive APIクイックスタート](https://developers.google.com/drive/api/v3/quickstart/python)
- [GitHub Secrets ドキュメント](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

---

**作成日**: 2025-11-04
**最終更新**: 2025-11-04
**関連ドキュメント**: [ISSUE.md](ISSUE.md)
