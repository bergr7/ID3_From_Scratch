# Let's try ID3 algorithm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from ID3 import DecisionTreeClassifier

# load some data
iris = load_iris()
features = iris['data']
features_ids = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
labels = iris['target']
labels_names = {index: name for index, name in enumerate(iris['target_names'])}
features_df = pd.DataFrame(data=features)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)

# instantiate ID3
clf = DecisionTreeClassifier(sample=features, attributes=features_ids, labels=list(labels))

# run algo
clf.id3_recv(sample_ids=features_ids, attribute_ids=labels_names, node=None)





