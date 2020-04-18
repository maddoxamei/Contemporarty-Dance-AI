
# coding: utf-8

# In[1]:


#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


#Loading dataset
source_dir = '/Akamai/MLDance/data/CSV/Pre-Processed/Combined/'
raw_data = pd.read_csv(source_dir+'_comprehensive_.csv')


# In[47]:


#Let's check how the data is distributed
data = raw_data.copy()
data.head()


# In[48]:


#Information about the data columns
data.info()


# In[49]:


#checking to see if there's any null variables
data.isnull().sum()


# In[50]:


#Preprocessing Data
col_list = list(data.columns)
end = [s for s in col_list if 'End' in s]
for col in end:
    data.pop(col)
data.pop('Time')

data = data[data.Sentiment != 2]


# In[51]:


# listing the unique values for the wine quality
data['Sentiment'].unique()


# In[52]:


#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


# In[53]:


data['Sentiment'] = label_quality.fit_transform(data['Sentiment'])

#Bad becomes 0 and good becomes 1 


# In[54]:


data.head()


# In[55]:


data['Sentiment'].value_counts()


# In[56]:


sns.countplot(data['Sentiment'])


# In[57]:


#Now seperate the dataset as response variable and feature variabes
X = data.drop('Sentiment', axis = 1)
y = data['Sentiment']


# In[58]:


#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[59]:


#Applying Standard scaling to get optimized result

sc = StandardScaler()


# In[60]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# # Generate Model Template

# In[61]:


from sklearn.metrics import accuracy_score

def runModel(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    #Let's see how our model performed
    print(classification_report(y_test, pred))
    
    #Confusion matrix for the random forest classification
    print("Confusion Matrix:\n"+str(confusion_matrix(y_test, pred)))
    
    cm = accuracy_score(y_test, pred)
    print("Accuracy:\t"+str(cm))


# ### Check Sample

# In[62]:


import random

def check_sample(model):
    index = random.randint(1,data.shape[0]) - 1
    sample = data.iloc[index]
    sample = sample.drop('Sentiment')

    Xnew = [sample]
    ynew = model.predict(Xnew)
    print('The sentiment of frame with given parameters is:') 
    print(ynew)

    Xnew = [sample-.35]
    ynew = model.predict(Xnew)
    print('The sentiment of frame with given parameters is:') 
    print(ynew)

    Xnew = [sample+.35]
    ynew = model.predict(Xnew)
    print('The sentiment of frame with given parameters is:') 
    print(ynew)


# # Classifiers

# ### Random Forest Classifier

# In[63]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
runModel(rfc)
check_sample(rfc)


# ### Naive Bayes

# In[64]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
runModel(gnb)
check_sample(gnb)


# ### Decision Tree

# In[65]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier() #random_state = 0
runModel(dtc)
check_sample(dtc)


# ### K Nearest Neighbor

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
runModel(knn)
check_sample(knn)


# ### Neural Network

# In[ ]:


from sklearn.neural_network import MLPClassifier

nnc = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
runModel(nnc)
check_sample(nnc)

