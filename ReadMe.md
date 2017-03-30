#Big Data Analytics

### Homework 1 (M10502216 ���a��)
- �Ѧ�https://github.com/ManSoSec/Microsoft-Malware-Challenge
�Ҳ��ͪ�Dataset�A�䤤��1800+�ݩ�

### (1) �����ݩʹ��c�N�{���������ġH

- �Q�� Scikit-Learn Random forests �p��Ximportances�ȶV����feature�V���ġC

### (2) �����ݩʹ��c�N�{�������L�ġH

- �Q�� Scikit-Learn Random forests �p��Ximportances�ȶV�C��feature�V�L�ġC

### (3) �Τ����k�i�H���U�A�M�w�W�z�����סH

- �Q�� Scikit-Learn Random forests �t��k�C

### (4) �z�LPython���ǮM��H�Τ�k�i�H���U�A�����W�����u�@�H

- pandas
- numpy
- matplotlib
- sklearn.ensemble

### (5)�ҵ{�������L��ĳ�H

- ���e�״I���`�סA���ѫܦh�귽���ڭ̾ǲߡA���Ѹ�Ư��ξ�ʷ|��n�C

**How to start**
- �w��Docker,�U��Spark-Anaconda
- �z�Lipython jupyter�s��Docker
- Dataset: https://github.com/ManSoSec/Microsoft-Malware-Challenge
- resource: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

#### python�N? python codes

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
#### ���R���G

![](https://github.com/KuanChiChiu/Big_Data_Analytics_HW1/tree/master/img/result.png)
