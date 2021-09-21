import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
from tqdm import tqdm_notebook as tqdm
import warnings
from sklearn.pipeline import Pipeline

df=pd.read_csv("./sample_data/Online_Research/Online_research.csv",delimiter=",")
temp=df["factor3"].str.split(pat=";",expand=True)

df.drop(["factor3"],axis=1,inplace=True)

df=pd.concat([df,temp],axis=1)

df.drop([3,4,5],axis=1,inplace=True)

df.rename(columns = {0: 'site_1', 1: 'site_2', 2: 'site_3'}, inplace = True)


df["site_2"].fillna("NA",inplace=True)


df["site_3"].fillna("NA",inplace=True)

print(df.head(3))
print("Attributes: \n",list(df.columns))
print("Total attributes: \n",len(list(df.columns))-1)
print("Shape of data frame: \n",df.shape)

#checking for Online and Offline stats
sns.countplot(x='Preference',data=df, palette='hls')
plt.xlabel('Preference', fontsize=12)
plt.ylabel('Preference', fontsize=13)
plt.show()
print("Total Number of Online",len(df[df['Preference']=='Online']))
print("Total Number of Online",len(df[df['Preference']=='Offline']))



#creating dummies for categorical data
columns1=['Gender', 'Age_group', 'Qualification', 'Income_group', 'Employment_status', 'factor1', 'factor2', 'factor4', 'factor5', 'factor6', 'factor7', 'factor8', 'factor9', 'factor10', 'factor11', 'factor12', 'factor13', 'factor14','site_1', 'site_2', 'site_3']

df_dummies=pd.get_dummies(df,columns=columns1)
print("Columns names: ",df_dummies.columns.tolist())
print("Total Number of Columns: ",len(df_dummies.columns.tolist()))



le=LabelEncoder()
df_dummies['Preference']=le.fit_transform(df_dummies['Preference'])

labels=df_dummies[['Preference']]
print("Check Labels",labels.head())


#Dropping the target variable
features=df_dummies.drop(['Preference'],axis=1)



#Dividing training and testing data
X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,
                                               stratify=labels)
print("Training split input;", X_train.shape)
print("Testing split input;", X_test.shape)


pipe_steps=[('sacler',StandardScaler()),('decsT', DecisionTreeClassifier())]

check_params={'decsT__criterion':['gini','entropy'],
              'decsT__max_depth': np.arange(3,15)}

pipeline=Pipeline(pipe_steps) 
print(pipeline)             

print("Start fitting data")
warnings.filterwarnings("ignore")

for cv in tqdm(range(3,6)):
  create_grid= GridSearchCV(pipeline, param_grid=check_params,cv=cv)
  create_grid.fit(X_train,y_train)
  print("Score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test,y_test)))
  print("!!best fit parameters from GridSearchCV")
  print(create_grid.best_params_)

print('out of loop')



#Fitting the data
DecsTree=DecisionTreeClassifier(criterion='gini', max_depth=6)
DecsTree.fit(X_train,y_train)
print('Decision Tree Classifier Created')
y_pred=DecsTree.predict(X_test)
print("Classification report",classification_report(y_test,y_pred))


#creating heatmap
a=df_dummies.corr()
sns.heatmap(a)



#Create confusion matrix
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5,annot=True, square=True, cmap='Blues')
plt.ylabel('Actual Value')
plt.xlabel('Predicted label')
plt.show()



#Create decision tree plot
plt.figure(figsize=(70,25))
dec_tree=plot_tree(decision_tree=DecsTree, feature_names=features.columns,
                   class_names=['Offline','Online'],filled=True)
plt.show()




