---
layout: post
title:  如何自學python與deep learning
date:   2020-10-18 12:01:35 +0800
image:  03.jpg
tags:   
    - "resource" 
    - "deep-learning"
---

## 前言
近幾年來由於深度學習等相關技術蓬勃發展，同時產生許多新的工作機會，例如: 資料科學家、機器學習工程師、數據分析師等等。加上時下流行的深度學習模型大都以較親民的程式語言Python開發，故吸引許多人踏進機器學習領域。本篇文章將以筆者的個人經驗出發，分享自己從一個只修過一學期程式設計的新手變成可以用Python找到工作的<strong>心路歷程及推薦的學習資源</strong>。但我自認為在機器學習領域中自己還只能算是有點經驗的開發者而已，故本篇只適用於跟我一樣以前<strong>幾乎完全不會寫程式的人XD</strong>

以下我將分為幾個階段分別論述:
0. 從零開始學習程式語言的好幫手-Google Search
1. 從零開始熟悉Python
2. 利用Python開發專案
3. 從零開始熟悉深度學習(Deep learning)
4. 從零開始熟悉TensorFlow & PyTorch

## 從零開始學習程式語言的好幫手-Google搜尋
對很多新手而言，可能很難自己領悟Google搜尋還可以幫助你學習! 當初我還在這個階段時，剛好有資工系的朋友跟我說要善用Google搜尋，真的是受益良多XD
剛開始的時候，你可能會遇到以下幾種情況
- 看不懂程式的錯誤訊息
- 拿錯誤訊息去Google後看不懂大家的回答

遇到這些情況時，我給的建議就是<strong>「要有耐心與毅力」</strong>! 剛開始學習時大部分內容看不懂是正常的，看不懂得同時你還可以更深入地去Google。舉例而言，你可能看不懂<strong>「no module named 'cv2'」</strong>是什麼意思，但你可以有以下幾種做法
- 拿整句錯誤訊息餵給Google
- 如果使用第一個辦法卻看不懂大家的回覆時，也許你可以試著先了解什麼是module? 試著了解什麼是cv2?

這只是個很簡單的例子，實際上可能遇到的問題可以說是千奇百怪，我只能建議你要試著有耐心地一層一層透過搜尋關鍵字去學習你看不懂的東西，總有一天你將可以自己解決大部分錯誤!

![]({{site.baseurl}}/images/cv2.png)
*你可能看不懂裡面的一些詞彙，可以試著再深入地去Google*


當你發現Google結果中有出現下列的網站，可以試著打開來看看
- [stakeoverflow](https://stackoverflow.com/): 權威性的解答網站，給programmer的知識+
- [CSDN](https://www.csdn.net/): 知名的中文解答網站，缺點是品質參差不齊

*其實我覺得從Google搜尋學習的能力可能是本篇文章最重要的重點XDD*

## 從零開始熟悉Python
#### Syntax
Python的基礎語法其實相當簡單且直覺，對於初學者而言，學習重點大概有以下幾項
- Indentation(用縮排表現程式碼層次)
- 變數命名、類別
- 條件判斷式(if-else)
- 迴圈(for, while loop)
- list, dictionary, set, tuple等資料結構
- 函數(def)

以上的內容都不難，以我的經驗大概只要自己實際做一遍就學會了!
接下來就介紹幾個我認為適用於完全不會寫程式的新手的學習資源

###### (1) 線上資源
- [w3schools](https://www.w3schools.com/python/default.asp)，一個教你程式語言的入門網站，除了Python以外還包含了其他常用的程式語言
- [菜鳥教程](https://www.runoob.com/python3/python3-tutorial.html)，跟w3schools類似，但是是以簡體中文編寫
- [codecademy](https://www.codecademy.com/catalog/language/python)，提供互動式學習的網站，缺點是Pro等級才能解鎖大部分功能

我認為第一或第二個網站加上下面介紹的Python開發環境即可為你打造不錯的學習環境，應該不太需要付錢買課程XD

###### (2) 線下資源
- [台大資訊系統訓練班](https://train.csie.ntu.edu.tw/train/)，台大資工系開設的付費課程，可以挑選自己有興趣的課程報名

#### Environment
在這裡我會推薦幾個初學者常用的Python環境建置方式

- [Anaconda](https://www.anaconda.com/)，一個非常常用的套裝軟體，它會自動幫你安裝Python以及常用的套件(package)，最重要的是它整合了<strong>Jupyter Notebook</strong>。Jupyter Notebook提供了一個在網頁上執行Python的互動式介面，對初學者而言相當適合
- [Google Colab](https://colab.research.google.com/)，Google提供的線上版Jupyter Notebook，可連結你的Google帳號且與Google Sheet, Google Slides等產品一樣支援共用功能。Google Colab亦可連結你的Google Drive內資料夾，非常方便! 除此之外，Google還很佛心地提供GPU供大家使用(但每次最多只能跑12小時)，對於GPU資源不夠的開發者而言可省下不少成本。

![]({{site.baseurl}}/images/colab.png)
*Colab的介面，其實就是Jupyter Notebook，可以即時顯示你要的變數內容或運算結果*

以上就是我當初從零開始學習Python時接觸過且覺得不錯的資源分享，完成這個階段之後你可能也會跟我一樣產生以下幾個迷惘
- 我已經會用Python基礎語法且知道怎麼利用Jupyter Notebook寫code，但這樣<strong>真的算是學會Python了嗎?</strong>
- 我想要找一些小專案試試看自己的能力，以驗證自己是否真的熟悉Python，<strong>但我不知道可以做些什麼</strong>

如果你有類似的困擾，那麼可以參考下一個section中的建議~

## 利用Python開發專案
當初我也曾經被上面那兩個問題困擾了很久，還好後來因為碩班產學合作題目給了我一些靈感與機會實作Python專案，我會在下方列出幾個當初自己曾經嘗試過的案子以及推薦的學習資源給你參考

#### 網頁爬蟲
對於初學者而言，我認為網頁爬蟲可以很好地幫你複習基礎語法，但開始進行爬蟲前你必須要知道網頁爬蟲大約有以下三種常用方法
- 利用[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)抓取靜態HTML內容: 這個方法可以抓到的網頁內容僅限於靜態網頁內容，也就是說，若你想要爬的網頁會根據你的瀏覽行為動態產生內容改變(例如: 臉書、ig往下滑之後會更新貼文)，那你就沒轍了。我推薦初學者可以從<strong>爬取ptt文章</strong>開始，較簡單且會讓你得到成就感XD
- 利用[selenium](https://pypi.org/project/selenium/)模擬瀏覽器行為以抓取網頁動態產生的內容。換句話說，selenium就是一個可以幫你滑臉書或ig的套件，你需要做的事情就是隨著內容的載入抓取你想爬的內容! 用這個方法的話我曾經爬過ig貼文，並將貼文的文字與圖片另存成檔案。
- 利用GET/POST取得網頁動態產生的內容後再抓取你需要的資訊。這個方法需要用requests這個套件對目標網頁的host傳遞參數並接收回應，基本上使用這個方法時你必須要知道網頁如何與後端溝通，剩下的其實也就是利用第一種方法從伺服器的回應中抓取你需要的資訊。建議的研究對象為臉書，可以試試看用這個方法抓取粉絲專頁的公開內容

以上三種方法網路上都有許多人分享作法，我就不贅述了。這邊主要是想告訴你可以從這些方向著手，還有一些可幫助你搜尋資源的關鍵字XD

![]({{site.baseurl}}/images/ptt.png)
*ptt是靜態網頁，右邊的畫面是你在chrome瀏覽時按下F12會出現的東西，或是按右鍵選單內的檢查也可以看到，裡面可以觀察這個網站的HTML結構*

#### 輿情分析
我很幸運地在剛進入碩班時老師就指派了一份產學合作計畫給我，該計畫的目標是分析A公司每天在被刊登的新聞中的輿論正負向，也就是說要<strong>判斷每一則新聞對A公司而言是正面的描述, 負面的描述或中立<strong>。當初我用到的工具有以下這些，你也可以參考看看
- 利用[PyQt](https://pypi.org/project/PyQt5/)製作GUI，因為我的輿論正負向判斷引擎是用Python編寫的。你也可以試著用<strong>網頁+Python</strong>的方式開發
- 利用[Jieba](https://github.com/fxsjy/jieba)對每一個句子進行斷詞，斷詞後可依語句結構或關鍵字查詢等方法判斷該句的正負向(這其實是蠻傳統的NLP方法)
- 但其實繁體中文斷詞有更好的工具可以幫助你，非常推薦你用[ckiptagger](https://github.com/ckiplab/ckiptagger)，它可以做斷詞(WS)、詞性標註(POS)、命名實體判別(NER)。(這時候你就可以現學現賣，去Google剛剛提到的詞彙是什麼意思，甚至是去找目前流行的方法有哪些)

舉例而言，當新聞中出現「大漲」、「看好」、「獲利」等正面詞彙時，通常表示這篇文章的情緒是正面的，但也要視這些形容詞所對應的主詞而定(如果是形容A公司的競爭對手的話，對A公司就不算是正面了)。基本上當初剛進碩班時就是用類似這樣的做法去判斷新聞正負向，其實蠻粗糙的而且效果也不算太好。不過對初學者而言我覺得算是一次不錯的練習!

*當初我可是熬了好久的夜，只為了看懂PyQt的[document](https://www.riverbankcomputing.com/static/Docs/PyQt5/module_index.html#ref-module-index)...*




## 從零開始熟悉深度學習(Deep learning)

當你已經可以用Python完成開發小專案時，代表已經初步熟悉語法與開發流程，接下來就可以嘗試接觸深度學習了XD 當初我是在碩一上到碩一下的寒假間發現深度學習似乎是一門很有趣的學問，所以才會開始利用線上資源自學，如果你也有興趣的話也許可以參考我接下來分享的經驗唷!

#### 什麼是深度學習? 它跟AI、機器學習的差別是什麼?

目前已經有很多人分享過類似的議題，所以我這邊只會粗淺地說明自己的想法。
- 我認為AI就是一套可以<strong>模擬人類行為的程式</strong>，它的表現方式不限於機器學習演算法，還可以使用rule-based的作法達成，所以他才會被叫做<strong>人工智慧</strong>嘛。 
- 而機器學習(machine learning)就更narrow囉，它是指一群可以從大量資料中(也許就是大家常說的大數據)學習到資料內隱含pattern演算法，所以它最大的特點是你必須餵給它資料，巧婦難為無米之炊阿XD
- 深度學習(deep learning)則是指機器學習演算法內屬於類神經網路(neural network)架構的一群演算法，基本上我認為它就是machine learning的分支!

![]({{site.baseurl}}/images/dl.png)
*經典的AI、ML與DL的關係圖*

在開始學習深度學習前，其實有一些預備知識是需要你先準備好的，但你也可以遇到問題時再去學習相關的理論。舉例來說，也就是當你在深度學習課程中看到gradients時卻不知道它是什麼時，再去Google學習XD

#### 學習前的預備知識

下面我會列出學習深度學習所需的預備知識以及推薦的線上資源~

###### 線性代數
基本上深度學習所使用的架構就是neural network，其中的計算大都與線性代數有關，所以學好線性代數會有助於你理解一些相關知識或是paper上的內容。當然，你有可能不需要將這些計算實際實作出來，因為目前已經有發展地很好的可以幫你的API(ex: tensorflow & pytorch)

- [3Blue1Brown](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&ab_channel=3Blue1Brown)很有名的youtuber，他的線性代數課程講解得非常好，非常推薦大家使用
- [李宏毅老師的課程](https://speech.ee.ntu.edu.tw/~tlkagk/courses.html)他算是我在台灣聽過最紅的machine learning教學youtuber，同時李老師也有錄製過線性代數的影片，非常適合喜歡用中文學習的人!

###### 機率論
除了線性代數之外，機率也在深度學習中扮演非常重要的地位，因為基本上模型就是在你given一組x的情況下，讓它學會輸出正確的y，所以你常常會在論文或相關教材中發現它們通常都會用到機率的概念。至於機率論的線上資源因為我並沒有使用過，所以可能沒辦法給你很好的推薦

![]({{site.baseurl}}/images/attention.jpg)
*你會很常看到數學...截圖自attention is all you need (Vaswani et al., 2017)*

#### Deep learning的學習資源
當你大概掌握了以上兩門課程之後，就可以開始進入深度學習的世界(但其實也可以先試著踏進來，再透過Google學習法補強)。基本上我在這個階段就是大量看了網路上的教學影片以及眾多網友的部落格，這個過程就是在幫你形塑neural network的基本概念，所以你一但遇到自己覺得可能沒有完全理解的地方，建議你透過反覆思考或大量Google的方式將知識內化。或者，你也可以試著<strong>用自己的話將相關知識講解給懂得人聽</strong>，讓他們去判斷你到底是不是徹底掌握了。以下的學習資源我將會以學校或老師來區分，因為他們每年的教學內容可能會不太一樣

1. [台大李宏毅老師](https://speech.ee.ntu.edu.tw/~tlkagk/courses.html)是我認為最淺顯易懂、最容易讓你入門深度學習的老師，他的講解風格非常有趣而且會搭配例子告訴你相關知識內容，這也是我入坑的地方
2. [清大吳尚鴻老師](https://nthu-datalab.github.io/ml/index.html)比較偏理論教學，且前半部分課程內容是有關machine learning方法(如Random forest, SVM...)，他可以讓你建立良好的理論基礎但同樣地就是比較難XD 除此之外，每一堂課助教都有對應的notebook幫助你學習，看完影片後搭配助教的notebook學習效果更好
3. [台大陳縕儂老師](https://www.csie.ntu.edu.tw/~yvchen/teaching)比較偏深度學習的教學，她的教學方式給我的感覺跟李宏毅老師差不多，也講解得非常清楚且篇應用。除此之外，我也推薦她的演算法課程，有興趣自學的話可以參考
4. [台大林軒田老師](https://www.youtube.com/playlist?list=PLXVfgk9fNX2I7tB6oIINGBmW50rrmFTqf&app=desktop)的機器學習基石與機器學習技法同樣地也非常知名，只是這門課我沒有去上過所以沒有辦法跟你說心得

當你完成了任何一位老師的課程之後，你大概也具備了進入深度學習領域的基礎門票，不過理論與實作缺一不可，而且這些老師的課程也都包含了實作的部分，所以接下來我必須要讓你知道現在最有名的兩個模型開發架構。

## 從零開始熟悉TensorFlow & PyTorch

從標題上就知道現在最有名的兩個開發架構就是Tensorflow與Pytorch，其實我很難告訴你他們哪個好哪個壞，目前我的經驗是case by case，也就是說當你熟悉pytorch或是同事與你協作的架構是pytorch時你勢必就要用pytorch。但若你的專案可能需要deploy到手機或網頁上，目前我聽說TensorFlow lite或TensorFlow.js可能可以幫到你，但我自己也沒實際碰過。所以給你的建議就是<strong>兩個都要會</strong>，你也可以隨便挑一個入坑~

他們的網站上都有不錯的教學，你照著教學作的話不僅可以熟悉實作的流程，也可以順便複習基本概念，所以我非常推薦初學者利用他們的tutorial。
- [PyTorch](https://pytorch.org/tutorials/)
- [TensorFlow](https://www.tensorflow.org/tutorials?hl=zh_tw)

除此之外，你也可以去逛逛Github看看別人都怎麼寫程式，除了自己閉門造車之外看別人的code也可以幫助你快速學習。這邊的話我就很難幫你推薦了，你應該要去找<strong>有興趣的project</strong>，學習別人怎麼寫程式之外還可以結合自己的理論，是一件非常美好的事。
基本上以上這些就是我在碩班兩年的學習過程，我也因為課程或面試需要參加過一些線上競賽，我最大的體悟就是你必須要不斷地讓自己進步和認清自己的能力與興趣。當你把時間投注在自己有興趣且利用有效的資源時，實力的進步絕對是最好的投資回報。

如果你有其他推薦的學習資源或是建議我這篇文章可以改進的地方，都非常歡迎你寄信跟我說:)


***
<small>image from [here](https://unsplash.com/photos/ieic5Tq8YMk) </small>