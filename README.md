# MoeGoe の Webアプリバージョン

- TTS のモデルデータのみ対応
- Clener の選択機能追加
- marine でのアクセント補正追加

![イラスト](https://user-images.githubusercontent.com/109923659/201433573-47ca6c32-855d-4e9a-b63f-3ba42bc3c46d.jpg)

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
./models/AAAAA/model.pth と config.json<br>
./models/BBBBB/model.pth と config.json<br>
./models/CCCCC/model.pth と config.json<br>
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
この赤字メッセージは使ってるフレームワーク Flask のデフォルトメッセージです<br>

<br>
chrome で http://127.0.0.1:15000 にアクセス<br>
<br>

# モデルデータ
このリポジトリにモデルデータは含まれていません<br>
リンク先の利用規約を読んだ上、ダウンロードしてください<br>
<br>
CjangCjengh/TTSModels<br>
https://github.com/CjangCjengh/TTSModels<br>
<br>

# 説明
- モデルデータを入れたフォルダを選択、次にスピーカーを選んで生成を押してください<br>

- 設定項目の CLENER<br>
CLENER は入力文字列を音素表現の文字列に変換する物です<br>
これの種類によって [JA] が必要になったりします<br>

- [JA]自動処理<br>
チェックしておけば必要な CLENER の時は内部で勝手に [JA] で囲みます<br>
もし日本語以外を喋らせたいときは、チェックを外して<br>
自分で各言語のラベルをテキストに入力してください<br>
モデルデータや CLENER によっては元々多言語対応していません<br>
例<br>
[ZH]你好。[ZH]<br>

- marine を使う<br>
日本語変換時のアクセントを補正<br>
詳細は<br>
marine (https://github.com/6gsn/marine)<br>

- その他<br>
エラーチェックが甘いので<br>
なんかエラー出たら、サーバー再起動、ブラウザ再読み込みしてください<br>





