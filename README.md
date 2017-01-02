# LKYDeepNN
=============
[LKYDeepNN](https://github.com/mosdeo/LKYDeepNN) 
- 可訓練的深度類神經網路 (Deep Neural Network) 函式庫
- 輕量，核心部份只依賴 C++11 標準函式庫，低相依性、好移植，方便在嵌入式系統上使用。



### 附有訓練視覺化 demo 程式
- 訓練視覺化程式以 OpenCV 撰寫，但 LKYDeepNN 本身不依賴 OpenCV。

![Alt text](https://github.com/mosdeo/LKYDeepNN/blob/master/classification_demo.png "Classification Demo")　　![Alt text](https://github.com/mosdeo/LKYDeepNN/blob/master/regression_demo.png "Regression Demo")




### 隱藏層的層數和節點數可以任意設定，簡單又有彈性
- vector<int>(4,7) 4個隱藏層，每層都是7個節點。
- vector<int>{5,5,6,6} 4個隱藏層，每層節點數分別是:5個、5個、6個、6個。
- LKYDeepNN(5, vector<int>{8,7}, 4)
- 代表輸入點5個、2個隱藏層分別是8節點和7節點，最後輸出層有5個節點。


### 自由設定活化函數
- SetActivation( 隱藏層 , 輸出層 )
- SetActivation(new Tanh() ,new Linear())  //for 回歸推薦使用
- SetActivation(new ReLU() ,new Softmax()) //for 分類推薦使用


### 訓練
- void Training(double learningRate, int epochs, std::vector<*vector<*double\>\> trainData)
- 每一筆資料都需要先整理成std::vector<*double\>
- std::vector<*vector<*double\>\> 就是很多筆資料


### 歷史
- 最早寫論文用了 James D. McCaffrey 在 Blog 上公開的單層倒傳遞類神經網路做出成果。雖然自己小小修bug還增加功能，但是最核心的部份依然不夠了解。
- 之後還從1層改為2層，但是改的過程中，才發現自己不懂倒傳遞的相關理論，但是自己卻能寫出應用類神經網路的論文，覺得相當慚愧。
- 看到很多論文都一些特殊的網路結構，例如 dropout、AutoEncode 等等都只能望洋興嘆，覺得應該寫出自己完全了解、完全掌握修改彈性的類神經網路。
- 看到 [FukuML](https://github.com/fukuball/fuku-ml) 和 [libDNN](https://github.com/botonchou/libdnn/)，讓我覺得相當佩服，他們都是台灣人，所以我應該也能寫得出來吧？我也想要寫出自己的機器學習函式庫。
- 真的寫下去，就要面對自己其實不那麼懂倒傳遞演算法的羞恥心。
- 

### 未來預計處理issue
- 可保存和讀取的 Weights 和 Biases
- 更豐富得測試資料集
- Copy Constructor
- 資料正規化工具
- 更多活化函數
- 訓練過程中可自適應的動態 Learning Rate
- Weights 分析工具
- Convolution Layer


### reference
- James D. McCaffrey 的類神經網路 [Coding Neural Network Back-Propagation Using C#](https://visualstudiomagazine.com/articles/2015/04/01/back-propagation-using-c.aspx)
- 這是我用 C++ 修改後的版本，有一個 branch 是 2-Hidden-Layer [mosdeo/NeuralNetwork](https://github.com/mosdeo/NeuralNetwork)
- 倒傳遞演算法是看這篇文章做的 [A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
- 測試資料的產生函式，大多是從這邊抓 js 來改 [Tensorflow Playground](https://github.com/tensorflow/playground) 
