# LKYDeepNN
---
[LKYDeepNN](https://github.com/mosdeo/LKYDeepNN) 
- 可訓練的深度類神經網路 (Deep Neural Network) 函式庫。
- 輕量，核心部份只依賴 C++11 標準函式庫，低相依性、好移植，方便在嵌入式系統上使用。

### 附有訓練視覺化 demo 程式
- 訓練視覺化程式以 OpenCV 撰寫，但 LKYDeepNN 本身不依賴 OpenCV。
- 繪圖功能僅以 function pointer 傳入物件中，在訓練過程中呼叫。
- 下面這2張圖 ↓ ↓ ↓ ↓ 是 33fps 的 GIF，如果不會動的話，請按 F5 重新整理網頁，或是單獨對圖檔另開新視窗。
![Alt text](https://github.com/mosdeo/LKYDeepNN/blob/master/training_demo_classification.gif "Classification Demo")　　![Alt text](https://github.com/mosdeo/LKYDeepNN/blob/master/training_demo_regression.gif "Cos(2*x) Regression Demo")



### 隱藏層的層數和節點數可以任意設定，簡單又有彈性
- std::vector<int&gt;(8,7) 8個隱藏層，每層都是7個節點，還可以再高，只要記憶體夠大的話。
- std::vector<int&gt;{5,5,6,6} 4個隱藏層，每層節點數分別是:5個、5個、6個、6個。
- LKYDeepNN::LKYDeepNN(9, std::vector<int&gt;{4,8}, 7) 代表輸入點9個、2個隱藏層分別是4節點和8節點，最後輸出層有7個節點。
- 目前只能 Fully-Connected，未來會考慮實作 Dropout-Connected 或 Fuzzy-Connected

### 自由設定活化函數
- LKYDeepNN::SetActivation( 隱藏層 , 輸出層 )
- LKYDeepNN::SetActivation( new Tanh() ,new Linear() )  //回歸推薦使用
- LKYDeepNN::SetActivation( new ReLU() ,new Softmax() ) //分類推薦使用
- 目前已經有: ReLU, Tanh, Softmax, Sigmoid, Linear
- 可以繼承 abstract class Activation 自由實作任意活化函數

### 訓練
- void LKYDeepNN::Training(double learningRate, int epochs, std::vector<vector<double&gt;&gt; trainData)
- 每一筆資料都需要先整理成 std::vector<double&gt;
- std::vector<vector<double&gt;&gt; 就是很多筆資料，這個才能餵給模型。

### 歷史
- 最早寫[論文](http://handle.ncl.edu.tw/11296/ndltd/22213658258720259065)用了 James D. McCaffrey 在 Blog 上公開的單隱藏層倒傳遞類神經網路做出成果。雖然自己小小修 bug 還增加功能，但是最核心的部份依然不夠了解。
- 之後還從1層改為2層，但是改的過程中，才發現自己不懂倒傳遞的相關理論，但是自己卻能寫出應用類神經網路的論文，覺得相當慚愧。
- 看到很多論文都一些特殊的網路結構，例如 Dropout、AutoEncode 等等都只能望洋興嘆，覺得應該寫出自己完全了解、完全掌握修改彈性的類神經網路。
- 看到 [FukuML](https://github.com/fukuball/fuku-ml) 和 [libDNN](https://github.com/botonchou/libdnn/)，讓我覺得相當佩服，他們都是台灣人，所以我應該也能寫得出來吧？我也想要寫出自己的機器學習函式庫。
- 真的寫下去，就要面對自己其實不懂倒傳遞演算法的羞恥心，紮實的拿起紙筆推導演算，不能再期望 code 亂打能 build 就會動。

### 未來預計處理issue
- 可保存和讀取的 Weights 和 Biases
- 更豐富得測試資料集
- Cross Entropy BP
- Early Stopping
- L1 & L2 Regularization
- Copy Constructor
- 資料正規化工具
- 更多活化函數。
- 訓練過程中可自適應的動態 Learning Rate
- Weights 分析工具
- Convolution Layer

### 編譯參數
- 在 .vscode/tasks.json 裡

### Reference:
- James D. McCaffrey 的類神經網路 [Coding Neural Network Back-Propagation Using C#](https://visualstudiomagazine.com/articles/2015/04/01/back-propagation-using-c.aspx)。有一個我用 C++ 修改後的版本，其中一個 branch 是 2-Hidden-Layer [mosdeo/NeuralNetwork](https://github.com/mosdeo/NeuralNetwork)
- 倒傳遞演算法是看這篇文章做的 [A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
- 測試資料的產生函式，大多是從這邊抓 js 來改 [Tensorflow Playground](https://github.com/tensorflow/playground) 

### 致謝
- 由 ![Microprogram](http://i.imgur.com/isNhjvl.jpg) 贊助上班時間產出XD
