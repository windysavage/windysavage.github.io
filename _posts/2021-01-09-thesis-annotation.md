---
layout: post
title:  "Thesis Annotation Competition in AI Cup 2019"
date:   2021-01-09 8:01:35 +0800
description: An introduction to the progress of this competition, combining with BERT finetuning
img: thesis.jpg
tags:  [nlp, BERT, deep-learning]
---


## 前言
寫這篇文章的主要目的是整理之前曾經參加過的競賽，2019時我剛接觸Deep learning不久，那時因為課程要求所以參加了AI Cup 2019的論文標註競賽。現在回頭看起來，當初的程式碼與想法實在是不夠嚴謹與細緻，連最基本得版本、環境控管都沒做哈哈。所以這篇文章內容並不會很專業，畢竟當時的我也才剛接觸不久而已。

## 競賽說明
這個競賽的目的是，給定一個論文摘要(Abstract)，你必須要建立一個模型判斷該摘要中的每一個句子是屬於哪種寫法(background, method, result...等等)。值得注意的是，每一個句子的label不一定只有一個，也就是說可能會有一些句子同時屬於background/method這兩種寫法之類的。這就是當初的我認為比較有挑戰性的地方，因為通常我們做classification problem的時候，每個$$x$$都只會對應一個label $$y$$。更詳細的競賽說明可以到[這裡](https://tbrain.trendmicro.com.tw/Competitions/Details/8)看

## 解題思考
現在開始review我當初的解題過程。
- 因為這個任務是NLP相關，故很直接的我一開始以資料清理+RNN系列模型來處理。首先我可以將不必要的雜訊去除，以及將一些可以整合成一類的詞彙代換成別的token(像是美國、中國等等都可以是<b>COUNTRY</b>這個token)。至於模型，我一開始使用的是LSTM，但其實效果並沒有很理想。

- 接下來，我覺得可以利用更強的模型去做language modeling，所以直接使用當時很強大的BERT，而且[tensorflow hub](https://tfhub.dev/)很佛心地有提供各種他們已經pre-train過的模型，我只要挑自己需要的即可。除此之外，因為BERT在處理input的時候已經有某種preprocessing的效果，所以我在這時候選擇先不做任何前處理，讓子彈飛一會兒，試試看BERT的實力

{% highlight ruby %}
# 從tensorflow hub撈我需要的模型
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
{% endhighlight %}

{% highlight ruby %}
# 開始將input data轉換成BERT看得懂的形式
def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids
{% endhighlight %}

- 果不其然，使用BERT之後我的成績在public leaderboard上有了明顯的成長，大概在前1/3內。接下來，我在BERT後面直接接上一層簡單的dense layer，用training data去finetune整個模型，這也是非常常見的BERT用法之一。另外，因為我認為每一句話在摘要中的順序應該也會影響到每一句話的風格，例如前面的句子可能偏background，後面的句子應該是偏result的寫法!所以我將順序資訊加上BERT本身的output concat在一起之後才餵進最後一層的dense layer。

{% highlight ruby %}
# 定義模型輸入
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
input_order = tf.keras.layers.Input(shape=(1), dtype=tf.int32, name="orders")

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
x_order = tf.keras.layers.Dense(1)(input_order)

# concat BERT的輸出與順序資訊
merge_x = tf.concat([pooled_output, x_order], axis=1)

# 最後一層的dense layer，拿來分類用
x = tf.keras.layers.Dropout(0.3)(merge_x)
x = tf.keras.layers.Dense(6, activation='sigmoid')(x)

model = Model(inputs=[input_word_ids, input_mask, segment_ids, input_order], outputs=x)
model.summary()
{% endhighlight %}

- 最後，我使用的loss function是binary cross entropy，因為我把問題變成一個binary classification問題。舉例來說，假設A句子同時有兩個label，分別是background和result，我就把他的label轉變為{{"[1,0,0,0,0,1]"}}這個形式，對每一維而言，模型都是看作二元分類問題(是不是background/是不是method/是不是result...)。

{% highlight ruby %}
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
{% endhighlight %}

- 後來我們團隊成績落在f1-score 0.68左右，冠軍大約是0.74。後來因為我們團隊只有我一個人在打這個比賽，加上那時已經接近期末，研究室也比較忙所以表定到12/31的比賽我大概只玩到12/10左右就不玩了......其實有點可惜，應該還有很多事情可做~

# 結語
最後，我整理一下應該還可以執行的改善方案...
1. 加回資料前處理步驟，token代換應該是有效的，畢竟他可以有效降低資料雜訊
2. 使用ensemble，根據一些後來的經驗，我發現ensemble也可以有效地提升model的robustness
3. 對同一個句子而言，label間的交互關係我沒有利用到。例如，background寫法的句子不太可能同時有method這個label...
4. 做多一點實驗，嘗試不同角度的machine learning problem formulation!
5. 對商業應用而言，其實若這些f1 score的差距並不影響實際上線表現，可以不用汲汲營營的追求分數...這也是為什麼我後來沒有很喜歡比賽


<small>image from [here](https://unsplash.com/photos/npxXWgQ33ZQ) </small>