# LKYDeepNN
=============
[LKYDeepNN](https://github.com/mosdeo/LKYDeepNN) 是一個輕量、只依賴 C++11 標準函式庫、方便在嵌入式系統上使用的深度類神經網路 (Deep Neural Network) 函式庫。核心部份只用 C++ 撰寫而成，目的是達到低相依性、可移植性、使函式庫容易修改。


## 隱藏層的層數和節點數可以任意設定，簡單又有彈性
-------------
- vector<int>(4,7) 4個隱藏層，每層都是7個節點。
- vector<int>{5,5,6,6} 4個隱藏層，每層節點數分別是:5個、5個、6個、6個。
- LKYDeepNN(5, vector<int>{8,7}, 4)
- 代表輸入點5個、2個隱藏層分別是8節點和7節點，最後輸出層有5個節點。


## 自由設定活化函數
-------------
- SetActivation(new Tanh(),new Linear())  //for 回歸
- SetActivation(new ReLU(),new Softmax()) //for 分類

## 訓練
-------------
- void Training(double learningRate, int epochs, vector<vector<double>> trainData)
- 每一筆資料都需要先整理成vector<double>
- vector<vector<double>> 就是很多筆資料
