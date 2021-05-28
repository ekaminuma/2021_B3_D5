# FY2021 Junior D5[Practical Training]

#### Eli Kaminuma (Nagoya City University)
---
実習内容
|  |トピック|事例 |課題投稿 |
|:--- |:--- |:--- |:--- |
|1回|AI体験 | ジャンケン画像分類AIの作成 (Teachable Machine) | ポーズ分類AIの作成 (Teachable Machine)  |
|2回|Linux入門|  顔トラッキングAI,VTuber体験 (FaceMesh, FaceVTuber) | Linuxコマンド関係テスト  |
|3回|オープンデータ入門|  URLからQRコード作成 (Google Chart API) |タイムラプス衛星画像の探索 (Google Earth Engine)|
|4回|機械学習基礎|Iris品種分類(SVM, DecisionTree, RadomForest)|ワイン品質分類(SVM, RadomForest)|
|5回|深層学習1:基礎|家庭科試験問題[3大栄養素]で解答推論(Transformer)|青空文庫テキストで感情分析(Transformer)|
|6回|深層学習2:分類|||
|7回|深層学習3:回帰,生成|||
|8回|先端技術|||

---
### References

- 第1回　AI体験
   - Teachable Machine by Google
      - [Google Teachable Machine 2.0](https://teachablemachine.withgoogle.com/) 
      - [Teachable Machine 2.0の紹介映像(YouTube)](https://www.youtube.com/watch?v=T2qQGqZxkD0) 2019/11/8
   - Photoreal Roman Emperor Project by Daniel Voshart
      - [ArtBreeder](https://www.artbreeder.com/)
      - [Photoreal Roman Emperor Project. 54 Machine-learning assisted portraits | by Daniel Voshart | Medium](https://voshart.medium.com/photoreal-roman-emperor-project-236be7f06c8f) 2020/7/25
      - [AI 'resurrects' 54 Roman emperors, in stunningly lifelike images | Live Science](https://www.livescience.com/ai-roman-emperor-portraits.html) 2020/9/28
   - DALL-E Project by OpenAI
      - [DALL-E Project | OpenAI ](https://openai.com/blog/dall-e/) 
      - [DALL-E：テキストからの画像生成の日本語翻訳 | Note ](https://note.com/npaka/n/n412754686518)
   - Turing Test
      - [Computer AI passes Turing test in 'world first' | BBC news](https://www.bbc.com/news/technology-27762088)   
       
---
- 第2回 Linux入門
   - 課題投稿関係
     - [Google Colaboratory](https://colab.research.google.com/)
   - Linux参考情報
       - [Linux | Wikipedia](https://ja.wikipedia.org/wiki/Linux)
       - [リーナストーバルズ | Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%AA%E3%83%BC%E3%83%8A%E3%82%B9%E3%83%BB%E3%83%88%E3%83%BC%E3%83%90%E3%83%AB%E3%82%BA)
       - 参考書:「Linux ステップアップラーニング」沓名亮典著、技術評論社
       - [Ubuntu | Windows 10でLinuxを使う | Qiita ](https://qiita.com/whim0321/items/093fd3bb2dd287a72fba)
   - AI体験(続き)
       - [CMU OpenPose Project](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
       - [Google MediaPipe Library](https://github.com/google/mediapipe)
       - [FaceMesh demo wt webcam](https://viz.mediapipe.dev/demo/face_detection)
       - [Virtual YouTuber(VTuber) | Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%90%E3%83%BC%E3%83%81%E3%83%A3%E3%83%ABYouTuber)
       - [FaceVTuber](https://facevtuber.com/)
       - [バーチャルYoutuberになれる！FaceVTuberの定性的で定量的なUX改善 |FaceVTuber作者の解説](https://qiita.com/kotauchisunsun/items/0e667068213ad04d7164)
---
- 第3回 API・オープンデータ入門

   - [500 Cities](https://nccd.cdc.gov/500_Cities)
   - [米国政府データポータル](https://data.gov/)
   - [日本政府データポータル](https://data.go.jp/)
   - [東京都データポータル](https://catalog.data.metro.tokyo.lg.jp/dataset)
   - [日本政府統計 e-Stat](http://data.e-stat.go.jp/)
   - [UCI機械学習レポジトリ](https://archive.ics.uci.edu/ml/index.php)
   - [Githubプログラムコードレポジトリ](https://github.com/)
   - [Microsoft Azure AI API](https://azure.microsoft.com/ja-jp/services/cognitive-services/computer-vision/)
   - [Weatherbonk](http://www.weatherbonk.com/maps/)
   - [Google Dataset Search](https://datasetsearch.research.google.com/)
   - Google Chart APIを利用したQRコード作成方法
      - https://chart.googleapis.com/chart?cht=qr&chs=300x300&chl=https://www.youtube.com/watch?v=T2qQGqZxkD0

---
- 第4回 機械学習の基礎
    - [機械学習 | Wikipedia](https://ja.wikipedia.org/wiki/機械学習)
    - [Machine Learning | Book |McGraw-Hill Series in Computer Science](http://www.cs.cmu.edu/~tom/mlbook.html)
    -  [Scikit-Learn |Python Library](https://scikit-learn.org/)
    -  [ML Algorithm Flowchart | Scikit-Learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
    -  [Faces recognition example using eigenfaces and SVMs | Scikit-Learn examples](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)
    -  [PCA of Iris Dataset| Scikit-Learn examples](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py)
    -  [K-Means Clustering of Iris Dataset | Scikit-Learn examples](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py)
    - [Iris Dataset | UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Iris)
    - [Wine Quality Dataset | UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
    - [Pandas CheatSheet | Python ](https://qiita.com/s_katagiri/items/4cd7dee37aae7a1e1fc0)
    - [PythonによるAI・機械学習・深層学習アプリのつくり方TensorFlow2対応| Book](https://www.amazon.co.jp/%E3%81%99%E3%81%90%E3%81%AB%E4%BD%BF%E3%81%88%E3%82%8B-%E6%A5%AD%E5%8B%99%E3%81%A7%E5%AE%9F%E8%B7%B5%E3%81%A7%E3%81%8D%E3%82%8B-Python%E3%81%AB%E3%82%88%E3%82%8BAI%E3%83%BB%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%BB%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92%E3%82%A2%E3%83%97%E3%83%AA%E3%81%AE%E3%81%A4%E3%81%8F%E3%82%8A%E6%96%B9-TensorFlow2%E5%AF%BE%E5%BF%9C-%E3%82%AF%E3%82%B8%E3%83%A9%E9%A3%9B%E8%A1%8C%E6%9C%BA/dp/4802612796/ref=pd_lpo_14_t_0/356-3999262-8812043?_encoding=UTF8&pd_rd_i=4802612796&pd_rd_r=c94f6ad1-0be5-4001-95e7-f3da4a2a425b&pd_rd_w=mExdS&pd_rd_wg=tBWGj&pf_rd_p=dc0198fa-c371-4787-b1e2-96ed0e4d45e8&pf_rd_r=QS9AAK7KZ5XPR01V2GM0&psc=1&refRID=QS9AAK7KZ5XPR01V2GM0)
    
---
- 第5回 深層学習の基礎[1] 概要
    -  参考書
       -  [Deep Learning | The MIT Press ](https://www.amazon.co.jp/Learning-Adaptive-Computation-Machine-English-ebook/dp/B08FH8Y533/ref=sr_1_3?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&dchild=1&keywords=Deep+Learning&qid=1620985996&sr=8-3) 2016/11/10
       -  [ゼロから作るDeep Learning|オライリージャパン](https://www.amazon.co.jp/%E3%82%BC%E3%83%AD%E3%81%8B%E3%82%89%E4%BD%9C%E3%82%8BDeep-Learning-%E2%80%95Python%E3%81%A7%E5%AD%A6%E3%81%B6%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%AE%E7%90%86%E8%AB%96%E3%81%A8%E5%AE%9F%E8%A3%85-%E6%96%8E%E8%97%A4-%E5%BA%B7%E6%AF%85/dp/4873117585/ref=sr_1_1?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&dchild=1&keywords=Deep+Learning&qid=1620985996&sr=8-1) 2016/9/24
    -  [DeepLearningモデルの層数の可視化](https://josephpcohen.com/w/visualizing-cnn-architectures-side-by-side-with-mxnet/)
    - 深層学習モデルの論文
      - [CNN | LuCun et al, Nature 521:436, 2015](http://dx.doi.org/10.1038/nature14539)
      -  [GAN | Goodfellow et al, arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
      -  [DQN | Mnih et al, Nature 518:529, 2015](https://www.nature.com/articles/nature14236)
    -  [ILSVRC | ImageNet Large Scale Visual Recognition Challenge](https://www.image-net.org/challenges/LSVRC/)
       -  [ImageNet database](https://www.image-net.org/)
    -  [Tensorflow Playground](https://playground.tensorflow.org/)
    -  [青空文庫トップ](https://www.aozora.gr.jp/)
       -  [夏目漱石 三四郎 | 青空文庫](https://www.aozora.gr.jp/cards/000148/files/794_14946.html) 
    -  [Transformerの解説](https://www.acceluniverse.com/blog/developers/2019/08/attention.html)
    -  Transformer　Sentiment Analysis事例
       -  [小説「天気の子」を丸ごと一冊、感情分析してみた|Qiita](https://qiita.com/toshiyuki_tsutsui/items/10f52c30fe1504b83ba1)
    -  Transformer Masked Language Modelの事例
       -  [BERTの使い方 - 日本語pre-trained modelsをfine tuningして分類問題を解く|Qiita](https://qiita.com/kenta1984/items/7f3a5d859a15b20657f3)
       - [BERT日本語モデルを使って、クリスマスプレゼントに欲しいものを推測してみた|CRESCO Engineers' Blog](https://www.cresco.co.jp/blog/entry/11517/)
---
- 第6回 深層学習の基礎[2] 分類 (畳込みニューラルネットワーク)
    - 深層学習ライブラリ
      - [Tensorflow](https://www.tensorflow.org/) Google
      - [Keras](https://keras.io/ja/)
      - [PyTorch](https://pytorch.org/)
      - [Caffe](https://caffe.berkeleyvision.org/), Chainer, Microsoft [CNTK](https://github.com/microsoft/CNTK), AWS [MXNet](https://aws.amazon.com/jp/mxnet/), [Paddle](https://github.com/PaddlePaddle/Paddle)
    - [Top Deep Learning Libraries 2018](https://www.kdnuggets.com/2018/04/top-16-open-source-deep-learning-libraries.html)
    - [Github Topics - Deep Learning](https://github.com/topics/deep-learning)
    - [データサイエンスと機械学習のサーベイ | Kaggle](https://www.kaggle.com/kaggle-survey-2020)
    - 参考書
       - [すぐに使える業務で実践できる PythonによるAI・機械学習・深層学習アプリのつくり方](https://www.socym.co.jp/book/1279) 2020/10/21
    - 畳込みニューラルネットワーク
       - [LuCun Y et al, Deep Learning, Nature 521:436, 2015](https://pubmed.ncbi.nlm.nih.gov/26017442/)
       - [Convolutional Neurel Network | Wikipedia(https://en.wikipedia.org/wiki/Convolutional_neural_network)
       - [kernelフィルタと画像処理](https://www.clg.niigata-u.ac.jp/~medimg/practice_medical_imaging/imgproc_scion/how_to_scion_image/process.htm)
       - [PaddingとStride](https://github.com/vdumoulin/conv_arithmetic)
    - 深層学習用データセット
       - MNIST
       - Fashion MNIST
       - CIFAR10
       - ImageNet
    - [ImageNet Pretrained Models]()
    - [ImageNet 
    - [学習済モデル](https://keras.io/guides/)


---
- 第7回 深層学習の基礎[3] 回帰 (再帰型ニューラルネットワーク), 生成 (敵対的生成ネットワーク)
    - 
---
- 第8回 DX先端技術の紹介
-----
----
