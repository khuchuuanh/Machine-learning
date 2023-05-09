import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz 


class TreeNode(object):
    def __init__(self, ids = None,  children = [], entropy = 0, depth = 0):
        self.ids = ids
        self.children = children
        self.entropy = entropy
        self.depth = depth
        self.split_attribute = None
        self.order = None
        self.label  = None

    
    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order


    def set_lable(self, label):
        self.label = label


def entropy(freq):
    # remove prob
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))


class DecisionTreeID3(object):
    def __init__(self, max_depth = 10, min_samples_split = 2, min_gain = 1e-4):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.Ntrain = 0
        self.min_gain = min_gain

    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data =data
        self.target = target
        self.labels = target.unique()

        ids = range(self.Ntrain)
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth= 0)
        queue = [self.root]

        while queue:
            node  = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)

    
    def _entropy(self, ids):
        if len(ids) ==0:
            return 0
        
        ids = [i+1  for i in ids]
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    



df = pd.read_csv('D:/machine learning/PlayTennis.csv')
#print(df.dtypes)

print(df.shape)
Le = LabelEncoder()
df['outlook'] = Le.fit_transform(df['outlook'])
df['temp'] = Le.fit_transform(df['temp'])
df['humidity'] = Le.fit_transform(df['humidity'])
df['windy'] = Le.fit_transform(df['windy'])
df['play'] = Le.fit_transform(df['play'])

print(df.head())
y = df['play']
X = df.drop(['play'], axis= 1)

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, y)
tree.plot_tree(clf)



dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 


X_pred = clf.predict([['2','1','0','0']])

print(X_pred)