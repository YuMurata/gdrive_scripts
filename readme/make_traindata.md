# make_traindata
学習データの作成

## 必要なデータ
* スコア付けされた補正パラメータ
* 補正用画像

## 処理内容
1. ランダムな補正パラメータを2つ生成
1. スコア付けされた補正パラメータを用いて生成した補正パラメータのスコアを概算
1. 学習データの作成
    1. 生成した2つのパラメータを用いて補正用画像を補正、画像ペアとする
    1. 概算されたスコアを比較し、大きい方をラベルとする

## コマンドライン引数
* `-n`、`--generate_num`：学習データ作成数
* `-i`、`--image_name`：画像名
* `-u`、`--user_name`：ユーザ名

## ディレクトリ構成
```
/
└─content
    ├─drive
    │  └─My Drive
    │      ├─Image
    │      │  └─[category]
    │      ├─scored_param
    │      │  └─[user]
    │      │      └─[image]
    │      └─weight
    │          └─[user]
    │              ├─logs
    │              └─xception
    │                  └─logs
    └─tfrecords
        └─[user]
            └─[image]
```