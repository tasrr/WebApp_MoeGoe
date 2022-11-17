# MoeGoe の ローカルWebアプリバージョン

- TTS, VTS, HuBERT-VITS, W2V2-VITS に対応

- marine でのアクセント補正追加

![イラスト2](https://user-images.githubusercontent.com/109923659/202389969-7c8cd4c7-1df1-48c8-992e-464f3cec2be6.jpg)

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
モデルデータ( .pth )と設定ファイル( .json )を１セットで好きな名前のフォルダに入れて<br>
models フォルダに入れる<br>
例<br>
./models/AAAAA/model.pth と config.json<br>
./models/BBBBB/model.pth と config.json<br>
./models/CCCCC/model.pth と config.json<br>
<br>
<br>
HuBERT-VITS 推論用データダウンロード ( 360 MB )<br>
https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt<br>
これが無いと HuBERT 対応のモデルデータが使えません。<br>
models の HuBERT フォルダに入れる<br>
./models/HuBERT/hubert-soft-0d54a1f4.pt<br>
<br>
<br>
W2V2-VITS 推論用データダウンロード ( 582 MB たまにサイトがかなり重いです )<br>
https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip?download=1<br>
これが無いと W2V2 対応のモデルデータが使えません。<br>
解凍してできた model.onnx と model.yaml を<br>
models の W2V2 フォルダに入れる<br>
./models/W2V2/model.onnx<br>
./models/W2V2/model.yaml<br>
<br>
<br>
WebApp_MoeGoe.exe 実行<br>
コンソールに<br>
```
2 0 2 2 - 1 1 - 1 7   1 6 : 1 3 : 0 9 . 3 5 7 6 0 1 9   [ W : o n n x r u n t i m e : D e f a u l t ,   o n n x r u n t i m e _ p y b i n d _ s t a t e . c c : 1 6 4 1   o n n x r u n t i m e : : p y t h o n : : C r e a t e I n f e r e n c e P y b i n d S t a t e M o d u l e ]   I n i t   p r o v i d e r   b r i d g e   f a i l e d .
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





