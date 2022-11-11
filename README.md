MoeGoe の Webアプリバージョン<br>
<br>
TTS のモデルデータのみ対応<br>
<br>
Clener の選択機能追加<br>
marine でのアクセント補正追加<br>
<br>
# CREDITS
- [MoeGoe](https://github.com/CjangCjengh/MoeGoe)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- [marine](https://github.com/6gsn/marine)
- [Flask](https://palletsprojects.com/p/flask/)

# 必要環境
windows 10<br>
ブラウザは chrome でしか動作確認してません<br>
<br>
# インストール
画面右の Releases から WebApp_MoeGoe.zip ダウンロード<br>
パスに日本語が含まれないフォルダで解凍<br>
ファイル数が多いので時間がかかります、解凍後サイズは 1GB ちょっと。torch がでかくて削れません。<br>
<br>
モデルデータ( .pth )と設定ファイル( .json )をセット毎にを好きな名前のフォルダに入れて<br>
models フォルダに入れる<br>
例<br>
./models/AAAAA/model.pth and config.json<br>
./models/BBBBB/model.pth and config.json<br>
./models/CCCCC/model.pth and config.json<br>
<br>
WebApp_MoeGoe.exe 実行<br>
コンソールに<br>
```
* Serving Flask app 'WebApp_MoeGoe'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:15000
Press CTRL+C to quit
```
が出れば起動成功<br>
※　WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.<br>
この赤字メッセージは Flask のデフォルトメッセージなので大丈夫です<br>
<br>
chrome で http://127.0.0.1:15000 にアクセス<br>
<br>

# モデルデータ
このリポジトリにモデルデータは含まれていません<br>
リンク先の利用規約を読んでください<br>
<br>
CjangCjengh/TTSModels<br>
https://github.com/CjangCjengh/TTSModels<br>
<br>

