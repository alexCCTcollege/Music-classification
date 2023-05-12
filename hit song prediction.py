#this is the file for hit song prediction on genre subset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix as confucio
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA


#DF
df = pd.read_excel('C:/Users/santi/OneDrive/Desktop/df.xlsx')
df.columns
df['key'] = df['key'].astype('category',copy=False)
df['mode'] = df['mode'].astype('category',copy=False)
df['time_signature'] = df['time_signature'].astype('category',copy=False)
regex = r'^(\d{4})' 
df['year'] = df['release_date'].str.extract(regex, expand=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')

#ALBUM ROCK      
album_rock = df[df['main_genre'] == 'album rock']
album_rock = album_rock.drop(columns=['song_name', 'billboard', 
                'artists', 'explicit','main_genre',
                'release_date', 'song_id'])

    
np.percentile(album_rock['popularity'], 75)
plt.hist(album_rock['popularity'])
plt.show()
album_rock['popularity'].max()

def cat(value):
    if value >= 54:
        return 1
    elif value < 54:
        return 0
   
album_rock['pop'] = album_rock.apply(lambda row: cat(row['popularity']), axis = 1)
album_rock = album_rock.drop(columns=['year'])
album_rock = album_rock.drop(columns=['popularity'])
album_rock = album_rock.drop(columns=['song_type'])


#null and shape ckecks
x_R = album_rock
x_R.shape
print(x_R.columns)
sns.boxplot(data=x_R)
plt.show()
x_R.isna().sum()

#define IQR
IQR_list = []
Q1_list = []
Q3_list = []

for gg in x_R.columns:
    Q1 = x_R[gg].quantile(0.25)
    Q3 = x_R[gg].quantile(0.75)
    IQR = Q3 - Q1
    IQR_list.append(IQR)
    Q1_list.append(Q1)
    Q3_list.append(Q3)
    
#cancel outliers
x_R = x_R[  x_R['duration_ms'] > (213127.0 - 1.5 * 70756.5)]
x_R = x_R[  x_R['duration_ms'] < (283883.5 + 1.5 * 70756.5)]
x_R = x_R[  x_R['acousticness'] > (0.0305 - 1.5 * 0.25949999999999995)]
x_R = x_R[  x_R['acousticness'] < (0.29 + 1.5 * 0.25949999999999995)]
x_R = x_R[  x_R['instrumentalness'] > (2.1e-06 - 1.5 * 0.0032979)]
x_R = x_R[  x_R['instrumentalness'] < (0.0033 + 1.5 * 0.0032979)]
x_R = x_R[  x_R['speechiness'] > (0.0305 - 1.5 * 0.0179)]
x_R = x_R[  x_R['speechiness'] < (0.0484 + 1.5 * 0.0179)]

#define target and categorical transformation
yx_R =  x_R 
categorical = ['key','mode', 'time_signature']
x_R = pd.get_dummies(x_R, columns =categorical )
y = yx_R['pop']
x_R = x_R.drop(columns=['pop'])
con_vars = ['duration_ms', 'acousticness','energy',
           'danceability', 'instrumentalness', 'liveness',
           'loudness', 'speechiness','valence','tempo']

#scaler
scaler = StandardScaler()
x_R[con_vars]=scaler.fit_transform(x_R[con_vars])

# PCA
# Loop Function to identify number of principal components that explain at least 85% of the variance
for comp in range(x_R.shape[1]):
    pca = PCA(n_components= comp, random_state=42)
    pca.fit(x_R)
    comp_check = pca.explained_variance_ratio_
    final_comp = comp
    if comp_check.sum() > 0.85:
        break
        
Final_PCA = PCA(n_components= final_comp,random_state=42)
Final_PCA.fit(x_R)
cluster_df = Final_PCA.transform(x_R)
num_comps = comp_check.shape[0]
print("Using {} components, we can explain {}% of the variability in the original data.".format(final_comp,comp_check.sum()))



precision_ = []
recall_ = []
accuracy_ = []
f1_ = []

'''KNN 1'''

#split
X_train, X_test, y_train, y_test = train_test_split(cluster_df,y,test_size=0.30)

#undersampling
print("Before undersampling: ", Counter(y_train))
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
print("After undersampling: ", Counter(y_train_under))

#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_under, y_train_under)
print(knn.score(X_test, y_test))


#Checking for k value
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train_under, y_train_under)
	
	# Compute training and test data accuracy
	train_accuracy[i] = knn.score(X_train_under, y_train_under)
	test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()



#cv
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=2)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X_train_under, y_train_under, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))



#confusion matrix
y_pred= knn.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))


precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))


#dance
df['key'] = df['key'].astype('category',copy=False)
df['mode'] = df['mode'].astype('category',copy=False)
df['time_signature'] = df['time_signature'].astype('category',copy=False)
regex = r'^(\d{4})' 
df['year'] = df['release_date'].str.extract(regex, expand=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
dance = df[df['main_genre'] == 'dance pop']
dance = dance.drop(columns=['song_name', 'billboard', 
                'artists', 'explicit','main_genre',
                'release_date', 'song_id'])



np.percentile(dance['popularity'], 75)
plt.hist(dance['popularity'])
plt.show()
dance['popularity'].max()

def cat2(value):
    if value >= 63:
        return 1
    elif value < 63:
        return 0
   
dance['pop'] = dance.apply(lambda row: cat2(row['popularity']), axis = 1)
dance = dance.drop(columns=['year'])
dance = dance.drop(columns=['popularity'])
dance = dance.drop(columns=['song_type'])

#checks
x_D = dance
x_D.shape
print(x_D.columns)
sns.boxplot(data=x_D)
plt.show()
x_D.isna().sum()

#IQR
IQR_list = []
Q1_list = []
Q3_list = []

for gg in x_D.columns:
    Q1 = x_D[gg].quantile(0.25)
    Q3 = x_D[gg].quantile(0.75)
    IQR = Q3 - Q1
    IQR_list.append(IQR)
    Q1_list.append(Q1)
    Q3_list.append(Q3)
    
#handling outliers
x_D = x_D[  x_D['duration_ms'] > (215707.0 - 1.5 * 53353)]
x_D = x_D[  x_D['duration_ms'] < (269060.0 + 1.5 * 53353)]
x_D = x_D[  x_D['acousticness'] > (0.0244 - 1.5 * 0.21509999999999999)]
x_D = x_D[  x_D['acousticness'] < (0.2395 + 1.5 * 0.21509999999999999)]
x_D = x_D[  x_D['instrumentalness'] > (0.0 - 1.5 * 4.6649999999999996e-05)]
x_D = x_D[  x_D['instrumentalness'] < (4.6649999999999996e-05 + 1.5 * 4.6649999999999996e-05)] 
x_D = x_D[  x_D['speechiness'] > (0.036449999999999996 - 1.5 * 0.061700000000000005)]
x_D = x_D[  x_D['speechiness'] < (0.09815 + 1.5 * 0.061700000000000005)]
    
#categorical data
yx_D =  x_D 
y = yx_D['pop']
x_D = pd.get_dummies(x_D, columns =categorical )
x_D = x_D.drop(columns=['pop'])


'''KNN 2'''
#scaler
x_D[con_vars]=scaler.fit_transform(x_D[con_vars])

# PCA
# Loop Function to identify number of principal components that explain at least 85% of the variance
for comp in range(x_R.shape[1]):
    pca = PCA(n_components= comp, random_state=42)
    pca.fit(x_D)
    comp_check = pca.explained_variance_ratio_
    final_comp = comp
    if comp_check.sum() > 0.85:
        break
        
Final_PCA = PCA(n_components= final_comp,random_state=42)
Final_PCA.fit(x_D)
cluster_df2 = Final_PCA.transform(x_D)
num_comps = comp_check.shape[0]
print("Using {} components, we can explain {}% of the variability in the original data.".format(final_comp,comp_check.sum()))

#split
X_train, X_test, y_train, y_test = train_test_split(cluster_df2,y,test_size=0.30)

#undersampling
print("Before undersampling: ", Counter(y_train))
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
print("After undersampling: ", Counter(y_train_under))

#knn
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_under, y_train_under)
print(knn.score(X_test, y_test))


#Checking for k value
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train_under, y_train_under)
	
	# Compute training and test data accuracy
	train_accuracy[i] = knn.score(X_train_under, y_train_under)
	test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()



#knn
knn_cv = KNeighborsClassifier(n_neighbors=2)
cv_scores = cross_val_score(knn_cv, X_train_under, y_train_under, cv=5)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))



y_pred= knn.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))

#country 
df = pd.read_excel('C:/Users/santi/OneDrive/Desktop/df.xlsx')
df['key'] = df['key'].astype('category',copy=False)
df['mode'] = df['mode'].astype('category',copy=False)
df['time_signature'] = df['time_signature'].astype('category',copy=False)
regex = r'^(\d{4})' 
df['year'] = df['release_date'].str.extract(regex, expand=False)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
cont = df[df['main_genre'] == 'contemporary country']
cont = cont.drop(columns=['song_name', 'billboard', 
                'artists', 'explicit','main_genre',
                'release_date', 'song_id'])


np.percentile(cont['popularity'], 75)
plt.hist(cont['popularity'])
plt.show()
cont['popularity'].max()

def cat3(value):
    if value >= 59:
        return 1
    elif value < 59:
        return 0
   
cont['pop'] = cont.apply(lambda row: cat3(row['popularity']), axis = 1)
cont = cont.drop(columns=['year'])
cont = cont.drop(columns=['popularity'])
cont = cont.drop(columns=['song_type'])


x_C = cont
x_C.shape
print(x_C.columns)
sns.boxplot(data=x_C)
plt.show()
x_C.isna().sum()

#IQR
IQR_list = []
Q1_list = []
Q3_list = []

for gg in x_C.columns:
    Q1 = x_C[gg].quantile(0.25)
    Q3 = x_C[gg].quantile(0.75)
    IQR = Q3 - Q1
    IQR_list.append(IQR)
    Q1_list.append(Q1)
    Q3_list.append(Q3)

#outliers
x_C = x_C[  x_C['duration_ms'] > (198960.0 - 1.5 * 41800.0)]
x_C = x_C[  x_C['duration_ms'] < (240760.0 + 1.5 * 41800.0)]
x_C = x_C[  x_C['acousticness'] > (0.0459 - 1.5 * 0.2741)]
x_C = x_C[  x_C['acousticness'] < (0.32 + 1.5 * 0.2741)]
x_C = x_C[  x_C['instrumentalness'] > (0.0 - 1.5 * 2e-05)]
x_C = x_C[  x_C['instrumentalness'] < (2e-05 + 1.5 * 2e-05)] 
x_C = x_C[  x_C['speechiness'] > (0.0289 - 1.5 * 0.012899999999999998)]
x_C = x_C[  x_C['speechiness'] < (0.0418 + 1.5 * 0.012899999999999998)]
    

#categorical data
yx_C =  x_C 
y = yx_C['pop']
x_C = pd.get_dummies(x_C, columns =categorical )
x_C = x_C.drop(columns=['pop'])


'''KNN 3'''
x_C[con_vars] = scaler.fit_transform(x_C[con_vars])

#PCA
# Loop Function to identify number of principal components that explain at least 85% of the variance
for comp in range(x_R.shape[1]):
    pca = PCA(n_components= comp, random_state=42)
    pca.fit(x_C)
    comp_check = pca.explained_variance_ratio_
    final_comp = comp
    if comp_check.sum() > 0.85:
        break
        
Final_PCA = PCA(n_components= final_comp,random_state=42)
Final_PCA.fit(x_C)
cluster_df3 = Final_PCA.transform(x_C)
num_comps = comp_check.shape[0]
print("Using {} components, we can explain {}% of the variability in the original data.".format(final_comp,comp_check.sum()))

#split
X_train, X_test, y_train, y_test = train_test_split(cluster_df3,y,test_size=0.30)

#undersampling
print("Before undersampling: ", Counter(y_train))
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
print("After undersampling: ", Counter(y_train_under))

#knn
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_under, y_train_under)
print(knn.score(X_test, y_test))


#Checking for k value
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train_under, y_train_under)
	
	# Compute training and test data accuracy
	train_accuracy[i] = knn.score(X_train_under, y_train_under)
	test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()



#cv
knn_cv = KNeighborsClassifier(n_neighbors=4)
cv_scores = cross_val_score(knn_cv, X_train_under, y_train_under, cv=5)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))



y_pred= knn.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))


'''
SVM
'''
#album rock
y = yx_R['pop']

X_train, X_test, y_train, y_test = train_test_split(cluster_df,y,test_size=0.30)

#SVC Model
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', SVC(random_state=1) )])

grid_params = {'classification__kernel': ['linear','poly', 'rbf', 'sigmoid'],
              'classification__C': [1,10,100]}

svm_clf = GridSearchCV(estimator=model, param_grid=grid_params, scoring='precision_weighted', cv=5)
# fit the model with the transformed training set
svm_clf.fit(X_train,y_train)

svm_clf_best_parameters = svm_clf.best_params_
print("Optimal parameters:\n", svm_clf_best_parameters)

svm_clf_best_result = svm_clf.best_score_ 
print("Best mean cross-validated score:\n", svm_clf_best_result)

y_pred= svm_clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))
#dance
x = x_D
y = yx_D['pop']
feature_scaler = StandardScaler()
x_D = feature_scaler.fit_transform(x_D)
X_train, X_test, y_train, y_test = train_test_split(cluster_df2,y,test_size=0.30)

#SVC Model
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', SVC(random_state=1) )])

grid_params = {'classification__kernel': ['linear','poly', 'rbf', 'sigmoid'],
              'classification__C': [1,10,100]}

svm_clf = GridSearchCV(estimator=model, param_grid=grid_params, scoring='precision_weighted', cv=5)
# fit the model with the transformed training set
svm_clf.fit(X_train,y_train)

svm_clf_best_parameters = svm_clf.best_params_
print("Optimal parameters:\n", svm_clf_best_parameters)

svm_clf_best_result = svm_clf.best_score_ 
print("Best mean cross-validated score:\n", svm_clf_best_result)


y_pred= svm_clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))

#country
x = x_C
y = yx_C['pop']
feature_scaler = StandardScaler()
x_C = feature_scaler.fit_transform(x_C)
X_train, X_test, y_train, y_test = train_test_split(cluster_df3,y,test_size=0.30)


#SVC Model
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', SVC(random_state=1) )])

grid_params = {'classification__kernel': ['linear','poly', 'rbf', 'sigmoid'],
              'classification__C': [1,10,100]}

svm_clf = GridSearchCV(estimator=model, param_grid=grid_params, scoring='precision_weighted', cv=5)
# fit the model with the transformed training set
svm_clf.fit(X_train,y_train)

svm_clf_best_parameters = svm_clf.best_params_
print("Optimal parameters:\n", svm_clf_best_parameters)

svm_clf_best_result = svm_clf.best_score_ 
print("Best mean cross-validated score:\n", svm_clf_best_result)


y_pred= svm_clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))




'''
NB
'''
#album rock

y = yx_R['pop']

X_train, X_test, y_train, y_test = train_test_split(cluster_df,y,test_size=0.30)
nb_model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', GaussianNB())
    ])
nb_model.get_params().keys()
nb_clf = GridSearchCV(estimator=nb_model, param_grid={}, scoring='recall', cv=5)
nb_clf.fit(X_train,y_train )


y_pred = nb_clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))

#dance
y = yx_D['pop']
feature_scaler = StandardScaler()
x_D = feature_scaler.fit_transform(x_D)
X_train, X_test, y_train, y_test = train_test_split(cluster_df2,y,test_size=0.30)
nb_model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', GaussianNB())
    ])
nb_model.get_params().keys()
nb_clf = GridSearchCV(estimator=nb_model, param_grid={}, scoring='recall', cv=5)
nb_clf.fit(X_train,y_train )


y_pred= nb_clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))
# country
y = yx_C['pop']
feature_scaler = StandardScaler()
x_C = feature_scaler.fit_transform(x_C)
X_train, X_test, y_train, y_test = train_test_split(cluster_df3,y,test_size=0.30)
nb_model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', GaussianNB())
    ])
nb_model.get_params().keys()
nb_clf = GridSearchCV(estimator=nb_model, param_grid={}, scoring='recall', cv=75)
nb_clf.fit(X_train,y_train )


y_pred= nb_clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))


precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))





'''
tree
'''
#rock

y = yx_R['pop']

X_train, X_test, y_train, y_test = train_test_split(cluster_df,y,test_size=0.30)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
clf = clf.fit(X_train_under, y_train_under)

tree.plot_tree(clf)




y_pred= clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))



#dance
y = yx_D['pop']
feature_scaler = StandardScaler()
x_D = feature_scaler.fit_transform(x_D)
X_train, X_test, y_train, y_test = train_test_split(cluster_df2,y,test_size=0.30)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
clf = clf.fit(X_train_under, y_train_under)

tree.plot_tree(clf)

y_pred= clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))


precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))


#country
y = yx_C['pop']
feature_scaler = StandardScaler()
x_C = feature_scaler.fit_transform(x_C)
X_train, X_test, y_train, y_test = train_test_split(cluster_df3,y,test_size=0.30)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)


clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3)
clf = clf.fit(X_train_under, y_train_under)

tree.plot_tree(clf)



y_pred= clf.predict(X_test)
cm = confucio(y_test, y_pred)
print(cm)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))


'''
forest
'''

Smote = SMOTE()
y = yx_R['pop']

X_train, X_test, y_train, y_test = train_test_split(cluster_df,y,test_size=0.30)

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = Smote.fit_resample( X_train , y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))


dt = RandomForestClassifier(random_state=42)

params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(X_train_SMOTE, y_train_SMOTE)
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()
score_df.nlargest(5,"mean_test_score")
dt_best = grid_search.best_estimator_

def evaluate_model(dt_classifier):
    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))
    
evaluate_model(dt_best)


y_pred= grid_search.predict(X_test)


print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))
#dance
y = yx_D['pop']

X_train, X_test, y_train, y_test = train_test_split(cluster_df2,y,test_size=0.30)

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = Smote.fit_resample( X_train , y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))


dt = RandomForestClassifier(random_state=42)

params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(X_train_SMOTE, y_train_SMOTE)
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()
score_df.nlargest(5,"mean_test_score")
dt_best = grid_search.best_estimator_

evaluate_model(dt_best)

y_pred= grid_search.predict(X_test)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))

precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))

#country
y = yx_C['pop']

X_train, X_test, y_train, y_test = train_test_split(cluster_df3,y,test_size=0.30)

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = Smote.fit_resample( X_train , y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))


dt = RandomForestClassifier(random_state=42)

params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(X_train_SMOTE, y_train_SMOTE)
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()
score_df.nlargest(5,"mean_test_score")
dt_best = grid_search.best_estimator_
    
evaluate_model(dt_best)

y_pred= grid_search.predict(X_test)

print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred , average = 'binary'))


precision_.append(precision_score(y_test, y_pred, average='macro'))
recall_.append(recall_score(y_test, y_pred, average='macro'))
accuracy_.append(accuracy_score(y_test, y_pred))
f1_.append(f1_score(y_test, y_pred , average = 'binary'))


print(precision_)
print(recall_)
print(accuracy_)
print(f1_)



print((sum(precision_)/len(precision_)))
print((sum(recall_)/len(recall_)))
print((sum(accuracy_)/len(accuracy_)))
print((sum(f1_)/len(f1_)))
