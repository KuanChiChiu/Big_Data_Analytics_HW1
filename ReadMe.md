#Big Data Analytics

### Homework 1 (M10502216 邱冠期)
- 參考https://github.com/ManSoSec/Microsoft-Malware-Challenge
所產生的Dataset，其中有1800+屬性

### (1) 哪些屬性對於惡意程式分類有效？

- 利用 Scikit-Learn Random forests 計算出importances值越高的feature越有效。

### (2) 哪些屬性對於惡意程式分類無效？

- 利用 Scikit-Learn Random forests 計算出importances值越低的feature越無效。

### (3) 用什麼方法可以幫助你決定上述的結論？

- 利用 Scikit-Learn Random forests 演算法。

### (4) 透過Python哪些套件以及方法可以幫助你完成上面的工作？

- pandas
- numpy
- matplotlib
- sklearn.ensemble

### (5)課程迄今有無建議？

- 內容豐富有深度，提供很多資源讓我們學習，提供資料能更統整性會更好。

**How to start**
- 安裝Docker,下載Spark-Anaconda
- 透過ipython jupyter連到Docker
- Dataset: https://github.com/ManSoSec/Microsoft-Malware-Challenge
- resource: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

#### python代? python codes

```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from into import into
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

mydata = pd.read_csv('/Microsoft-Malware-Challenge/Dataset/train/LargeTrain.csv')
X = np.array(mydata.ix[:,0:1804])
y = np.array(mydata.ix[:,1804:1805]).ravel()
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=10,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
# Choose top 10
plt.bar(range(10), importances[indices[0:10]],
       color="b", align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()
```
#### 分析結果

![](https://github.com/KuanChiChiu/Big_Data_Analytics_HW1/tree/master/img/result.png)
