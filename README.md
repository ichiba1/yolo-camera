# yolo-camera

Raspberry Pi のカメラ映像に YOLO で物体検出をかけて、結果をブラウザでリアルタイムに見るプログラムです。

- 動作確認: Raspberry Pi 5 + カメラモジュール (IMX219)
- モデル: YOLO11n（NCNN 形式に変換して約 12 FPS）

## セットアップ

```bash
# picamera2 を使うため --system-site-packages を付ける
python3 -m venv --system-site-packages ~/yolo-env
~/yolo-env/bin/pip install ultralytics

git clone https://github.com/ichiba1/yolo-camera.git ~/yolo-camera
cd ~/yolo-camera

# モデルをダウンロードして NCNN 形式に変換する
~/yolo-env/bin/yolo export model=yolo11n.pt format=ncnn
```

## 起動

```bash
~/yolo-env/bin/python ~/yolo-camera/yolo_stream.py
```

配信は Raspberry Pi 自身（127.0.0.1:8000）からだけ見られます。
別のパソコンで見るときは、SSH でポートをつないでからブラウザで http://localhost:8000 を開きます。

```bash
ssh -L 8000:localhost:8000 <ユーザー名>@<ラズパイのアドレス>
```

## ライセンス

[AGPL-3.0](LICENSE)（使用している [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) に合わせています）
