今回は自然言語処理技術を用いて音楽サブスクの検索・レコメンドシステムについての成果物を作りました。
詳しくはスライドのご確認等よろしくお願いいたします。

【工夫点】

Webアプリ編
・コードについて、関数を見やすい単位でまとめました（メンテナンス性）。
・楽曲分類の際の分類基準をwebサイトに載せました。一方、裏で計算する時は距離、収束速度のことを考慮して標準化しました。

機械学習編
文法的に正しい文章になる様、BERTで文章を生成することである程度自然な文章ができていることを保証し、最後に別のライブラリでもう一度文法チェックを行ったことです。


【苦労点】

Webアプリ編
いくつか機能があるのでコード量が増えたことです。
また、データフレームなどのインデックスを照らし合わせたり順番を変えたりなどの操作が多く、整理するのが大変でした。

機械学習編
・感情分類について精度を上げるのが大変でした。回帰、分類共通してデータ拡張をすることでより確実性の高いモデルにできました。
データ拡張について、文法的な問題で自然な文章を作るためのコードを書くことが、様々な要素が絡むという点で苦労しました。日本語の学習済みモデルを適用して文章を生成することである程度自然な文章になることを保証しつつ単語の入れ替え、ライブラリを用いた文法チェックにより解決しました。文数は、感情強度の合計が3以上のもの限定で16000だけ増やし、回帰モデルの決定係数が0.67→0.76まで増えました。wordnet辞書の単語数から考え、また繰り返し処理でランダムに単語を選んで文章を生成しているのでもっと文章を増やすことができます。今回は2回の繰り返し処理を行いました。
・Google ColaboratoryのGPUの使用可能時間と並列処理を増やして処理スピードを上げることのバランスを取ることが難しかったです。メモリ不足に何度も陥り、やり直すことが多くなり大変厳しかったです。tempfile.NamedTemporaryFileを用いてメモリ消費を抑えたのは効果的でした。


【今後の展望】

画面のデザインを改善、人気不人気のON/OFFボタン追加。
自分の音楽ファイルをハイレゾに近い音質にする機能を搭載。
他のユーザーとのデータも学習して汎用性を上げる。
フィードバックや履歴データに強化学習を適用してレコメンドの質を上げる。
他は各コード解説に記載しています。


【他の業界などでの活用】

エンタメ業界：映画やゲームのセリフにより当てはまりの良い音楽を選定。
小売業：
SNSやレビューの文章の感情をより細かく分けることでレコメンドなどを詳細に行う。
宣伝の仕方などによる物の売れ行きを分析。
語学学習：教え方、教材に対する生徒のレビューを学習。
スポーツ業界：試合、トレーニングでコーチや監督の話した言葉がどれだけ選手や利用者のモチベーションに影響を与えたかを学習。
医療業界：患者の感情、治療に対するレビューを分析、心理的サポート。関連する病気を検索。
