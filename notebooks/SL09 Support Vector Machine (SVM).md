
# Support Vector Machine (SVM)
<br><br>
Support vector machine models can be used for classification, regression and detecting outliers.  In practice though, SVMs are mostly used for classification purposes. SVM finds the hyperplane that best separates two classes. Generally 'best separates' means that SVM finds the hyperplane that is as equi-distant from each class as possible. In the picture below from wikipedia this 'maximum margin' concept is depicted.
<br>
![svm](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Svm_max_sep_hyperplane_with_margin.png/330px-Svm_max_sep_hyperplane_with_margin.png)
<br><br>
SVMs can make use of different kernel functions (a topic unto itself) as the decision function of a particular SVM implementation; this lends flexibility to SVM. So how does SVM work?
<br>
<br>
- SVM first finds the hyperplane that best separates both classes
  - Note that at this stage SVM has not computed the 'maximum margin' yet
- Then SVM readjusts this plane slightly to arrive at the best/maximum margin hyperplane

Note that when SVM encounters a non-separable problem it employs what is knows as the 'kernel trick'. What the kernel trick does is to take a low dimensional input space and turn it into a higher dimensional input space. In the image below the kernel trick is to change the basis function to $\phi(a,b) = \{a, b, a^2+b^2\}$ so that instead of creating a hyperplane, SVM is creating a circle such that the linearly separable data in red can be correctly classified.
<br><br> 
![kernel trick](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Kernel_trick_idea.svg/1260px-Kernel_trick_idea.svg.png)
<br>


```python
'''
In this example we are reading in a house description and sale dataset. For this classification we are going to 
estimate whether a house will sell(and with what probability) within 90 days of being put on the market.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# this data has already been cleaned up, standardized, one hot encoded and vetted
df = pd.read_csv("classification_house_sale_px_data.csv", parse_dates=True, sep=',', header=0)
df_labels = pd.read_csv("classification_house_sale_px_labels.csv", parse_dates=True, sep=',', header=0)

# split data into training and test sets
train, test, y_train, y_test = train_test_split(df, df_labels, train_size=.6, test_size=.4, shuffle=True)

# run the classifier on the training data
clf = LinearSVC(random_state=0, C=1e5, penalty="l2", loss="hinge", dual=True)
clf.fit(train, list(y_train.label.values))
# make prediction on the test data
#predicted = clf.predict(test)
print("SVM: Test set accuracy (R^2) = {0:.3f}".format(clf.score(test, y_test.label.values)))
```

    SVM: Test set accuracy (R^2) = 0.440


<br>
Note that SVM is scoring lower than a simple decision tree as per the previous notebook.
<br>

# Take away
- SVM are flexible since different kernel functions may be used as the decision function.
- SVM are robust to outliers by design
- SVM classification does not provide probability measures 
