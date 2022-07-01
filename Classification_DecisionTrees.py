#In the below code we use decision trees to predict the possibility of breast cancer using the breast_cancer dataset in sklearn.
#We use the train_test_split to split the data in 80/20
# The DecisionTreeClassifier is used from sklearn.tree to build the model
#We use the matplotlib plt to plot the tree



------------------------------------------
#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer


# In[10]:


data = load_breast_cancer()


# In[11]:


dataset =pd.DataFrame(data=data['data'], columns=data['feature_names'])


# In[12]:


dataset


# In[14]:


from sklearn.model_selection import train_test_split
X = dataset.copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)


# In[22]:


from sklearn.tree import DecisionTreeClassifier    # import the model from skl 
clf = DecisionTreeClassifier(max_depth=4)         # where we put tht parameters maximum depth of tree =4
clf = clf.fit(X_train, y_train)                    #provide training values


# In[17]:


clf.get_params()


# In[18]:


X_test


# In[23]:


predictions = clf.predict(X_test) # do a prediction 
predictions                       # 0 and 1 class predicitons


# In[24]:


clf.predict_proba(X_test) # gives a prabability

# utput shows for the 1st instance the probability of class 0 is 1 while probability of class 1 is 0


# In[25]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions) # shows the accuracy of our model


# In[27]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions, labels = [0,1])


# In[ ]:


#output shows, when class is 0 it preedicts 0 39 times and wrongly classified instances 1
#whereas when class is 1 it predicts 1 71 times and wrongly classified instances 3


# In[29]:


from sklearn.metrics import precision_score
precision_score(y_test, predictions)


# In[ ]:


#the output of precision score gives a precision of 95%


# In[30]:


from sklearn.metrics import recall_score
recall_score(y_test, predictions)


# In[31]:


#The output shows that our model has a recall score of 98%


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names = ['malignant', 'benign']))


# In[34]:


feature_names = X.columns
feature_names                       #gets the features


# In[37]:


clf.feature_importances_ #gives a list of how important the features are to make a prediction


# In[67]:


feature_importance = pd.DataFrame(clf.feature_importances_ , index = feature_names, columns =None, dtype=None, copy=None).sort_values(0 , axis=0, ascending=False, inplace=False, kind='quicksort')


# In[68]:


features = list(feature_importance[feature_importance[0]>0].index)
features


# In[69]:


feature_importance.head(10).plot(kind ='bar')


# In[70]:


from sklearn import tree
from matplotlib import pyplot as plt

fig =plt.figure(figsize =(25,20))
_=tree.plot_tree(clf,
                feature_names=feature_names,
                class_names={0:'Malignant', 1:'Benign' },
                filled = True,
                 fontsize=12)


# In[ ]:


#

