# LKYDeepNN
---

[LKYDeepNN](https://github.com/mosdeo/LKYDeepNN) 是一個輕量、只依賴 C++11 標準函式庫、方便在嵌入式系統上使用的深度類神經網路 (Deep Neural Network) 函式庫。核心部份只用 C++ 撰寫而成，目的是達到低相依性、可移植性、使函式庫容易修改。


## 特色
---
- 隱藏層的層數和節點數可以任意設定。
- vector<int>(4,7) 4個隱藏層，每層都是7個節點。
- vector<int>{5,5,6,6} 4個隱藏層，每層節點數分別是:5個、5個、6個、6個。
