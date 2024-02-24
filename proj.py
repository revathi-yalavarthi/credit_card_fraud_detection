import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import classification_report
data=pd.read_csv('creditcard.csv')
data.head()
data.tail()
data.shape
data.isnull().sum()
sc=StandardScaler()
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))
data=data.drop(['Time'],axis=1)
dup=data.duplicated().any()
data=data.drop_duplicates()
data.shape
count=data['Class'].value_counts()
sns.countplot(data, x='Class')
plt.title('Countplot of class variable')
plt.xlabel('class')
plt.ylabel('Count')
X=data.drop('Class',axis=1)
y=data['Class']
plt.show()
#Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


#UNDERSAMPLING
normal=data[data['Class']==0]
fraud=data[data['Class']==1]
n=normal.shape
f=fraud.shape
normal_sample=normal.sample(n=473)
ns=normal_sample.shape
new_data=pd.concat([normal_sample,fraud],ignore_index=True)
new_count=new_data['Class'].value_counts()
nd1=new_data.head()
X=data.drop('Class',axis=1)
y=data['Class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
sns.countplot(new_data, x='Class')
plt.title('Countplot of class variable')
plt.xlabel('class')
plt.ylabel('Count')
plt.show()

#OverSampling
X=data.drop('Class',axis=1)
y=data['Class']
from imblearn.over_sampling import SMOTE
X_res,y_res=SMOTE().fit_resample(X,y)
sns.countplot(data, x='Class')
plt.title('Countplot of class variable')
plt.xlabel('class')
plt.ylabel('Count')
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.20,random_state=42)


#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

# Train your model
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

# Save the model to a file
import joblib
joblib.dump(model, 'trained_model.pkl')

#classification report
report = classification_report(y_test, y_pred)
print(report)

#ROC CURVE
y_scores = model.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_scores)
print(f"AUC: {auc:.2f}")

#metric score and confusion matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1score=f1_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
print("accuracy score:",accuracy)
print("precision score:",precision)
print("recall score:",recall)
print("f1 score:",f1score)