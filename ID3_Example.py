import numpy as np
import pandas as pd
from collections import deque

# generate some data
# define features and target values
data = {
    'wind_direction': ['N', 'S', 'E', 'W'],
    'tide': ['Low', 'High'],
    'swell_forecasting': ['small', 'medium', 'large'],
    'good_waves': ['Yes', 'No']
}

# create an empty dataframe
data_df = pd.DataFrame(columns=data.keys())

np.random.seed(42)
# randomnly create 1000 instances
for i in range(1000):
    data_df.loc[i, 'wind_direction'] = str(np.random.choice(data['wind_direction'], 1)[0])
    data_df.loc[i, 'tide'] = str(np.random.choice(data['tide'], 1)[0])
    data_df.loc[i, 'swell_forecasting'] = str(np.random.choice(data['swell_forecasting'], 1)[0])
    data_df.loc[i, 'good_waves'] = str(np.random.choice(data['good_waves'], 1)[0])

data_df.head()

# separate target from predictors
X = np.array(data_df.drop('good_waves', axis=1).copy())
y = np.array(data_df['good_waves'].copy())
feature_names = list(data_df.keys())[:3]

# import and instantiate our DecisionTreeClassifier class
from ID3 import DecisionTreeClassifier

# instantiate DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(X=X, feature_names=feature_names, labels=y)
print("System entropy {:.4f}".format(tree_clf.entropy))
# run algorithm id3 to build a tree
tree_clf.id3()
tree_clf.printTree()



