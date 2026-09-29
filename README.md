# EMSOptimizer
![logo](./logo.svg)
EMSOptimizerは電気機器の設計に特化した数理最適化ライブラリです。  
本リポジトリはEMSOptimizerの公開ソースコード部（EMSOptFree）です。

## インストール手順
*Python 3.11.x 環境およびパッケージ管理ツールpipが必要です。

1. 本リポジトリの"Releases"から以下の2点をダウンロードします。
    - EMSOptFree最新バージョンのzipまたはtar.gzファイル
    - EMSOptFreeに付随するwhlファイル（emsopt_engine）

2. ダウンロードしたzipまたはtar.gzファイルをPCの任意の場所に展開します。

3. ダウンロードしたwhlファイルをコマンドラインから`pip install`します。
```sh
pip install emsopt_engine-(version)-(environment)-(os).whl
```
*依存パッケージを同時にインストールするため、実行完了まで数分かかる場合があります。

4. 形状最適化機能を有効化する場合、必要モジュール等をそれぞれの手順に従ってインストールします*。  
*必要モジュール等については[ドキュメントサイトのインストールガイド](https://emsolution-ssil.github.io/EMSOptimizerDoc/)をご覧ください。

### 開発者向け
zipまたはtar.gzファイルをダウンロード・展開する代わりに、リポジトリをPCにクローンすることもできます。

## ドキュメント
詳しい使用方法などについては[ドキュメントサイト](https://emsolution-ssil.github.io/EMSOptimizerDoc/)をご覧ください。

## 参考文献
\[1\] IEEJ Investigating R&D committee, “IEEJ technical report,” (in Japanese) Inst. Electr. Eng. Japan, Japan, Tech. Rep. 776, 2000.
