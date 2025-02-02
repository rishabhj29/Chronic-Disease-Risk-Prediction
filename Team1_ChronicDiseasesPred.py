import pandas as pd
import numpy as np

#Read the dataset and keep only the required columns.
brfss = pd.read_csv("BRFSSData_Complete.csv")


########################################################################
#Exploratory Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns
# Display the first few rows
print(brfss.head())
# Check data types
print(brfss.dtypes)
# Summary statistics for numerical columns
print(brfss.describe())
# Check for missing values
missing_values = brfss.isnull().sum()
missing_values = missing_values[missing_values > 0]
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel('Number of Missing Values')
plt.xlabel('Columns')
plt.title('Missing Values in Each Column')
plt.tight_layout()
plt.show()

# Histogram
brfss['PHYSHLTH'].hist()
plt.show()

# Occurence of Chronic Diseases
columns_of_interest = ['_ASTHMS1', '_DRDXAR2', '_MICHD', 'ADDEPEV3', 'DIABETE4']
new_names = ['Asthma', 'Arthritis', 'Coronary Heart Disease', 'Depression', 'Diabetes']
selected_columns = brfss[columns_of_interest]
percentages = (selected_columns == 1).mean() * 100

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(columns_of_interest)), percentages, color='skyblue')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')
ax.set_title('Occurence of Chronic Diseases')
ax.set_ylabel('Percentage of People')
ax.set_xticks(range(len(columns_of_interest)))
ax.set_xticklabels(new_names, rotation=45)
plt.tight_layout()
plt.show()

correlation_matrix = brfss.corr()
high_corr = correlation_matrix[(correlation_matrix >= 0.5) | (correlation_matrix <= -0.7)]
sns.heatmap(high_corr, annot=True, cmap='coolwarm', cbar=True)
plt.show()


# Thresholds for strong correlations
high_threshold = 0.9
low_threshold = -0.9

upper_tri = np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_)
upper_corr_matrix = correlation_matrix.where(upper_tri)

# Find the pairs with a high correlation (above high_threshold or below low_threshold)
correlated_pairs = upper_corr_matrix.stack().reset_index()
correlated_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
high_correlated_pairs = correlated_pairs.loc[(correlated_pairs['Correlation'] >= high_threshold) | 
                                             (correlated_pairs['Correlation'] <= low_threshold)]

#Network Graph
import networkx as nx
G = nx.from_pandas_edgelist(high_correlated_pairs, 'Feature1', 'Feature2', 'Correlation')
plt.figure(figsize=(14, 14))
pos = nx.spring_layout(G, seed=42)  # For consistent layout
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, edge_color='black', linewidths=1, font_size=10)
plt.title('Network Graph of Highly Correlated Features')
plt.show()

# Pair plot for a subset of Alcohol variables
sns.pairplot(brfss[['ALCDAY4', 'AVEDRNK3', 'DRNK3GE5', 'MAXDRNKS','DROCDY4_','_RFDRHV8']])
plt.show()
# Pair plot for a subset of Smoking variables
sns.pairplot(brfss[['_SMOKGRP', '_PACKYRS', '_YRSSMOK', '_RFSMOK3','_SMOKER3']])
plt.show()

#Boxplot for Height and Weight
brfss['WTKG3_adj'] = brfss['WTKG3'] / 100
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=brfss['WTKG3_adj'])
plt.title('Weight Distribution')
plt.ylabel('Weight in Kilograms')
plt.subplot(1, 2, 2) 
sns.boxplot(y=brfss['HTM4'])
plt.title('Height Distribution')
plt.ylabel('Height in Meters')
plt.tight_layout()
plt.show()

g = sns.FacetGrid(brfss, col='GENHLTH', row='SEXVAR', margin_titles=True)
g.map(sns.scatterplot, 'PHYSHLTH', 'MENTHLTH')
g.add_legend()
plt.show()


##############################################################################
##PCA Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
# Replace special values with NaN for imputation
special_values = [77, 88, 99, 777, 888, 999, 7777, 8888, 9999]
brfss.replace(special_values, np.nan, inplace=True)

# Convert all columns to numeric type, coercing errors to NaN
df_numeric = brfss.apply(pd.to_numeric, errors='coerce')

# Create an imputer object with a mean filling strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Impute missing values
columns_to_drop = ['IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'SEQNO', 'TOLDCFS', 'HAVECFS', 'WORKCFS']
df_numeric = df_numeric.drop(columns=columns_to_drop, axis=1, errors='ignore')
imputed_data = imputer.fit_transform(df_numeric)

if imputed_data.shape[1] != len(df_numeric.columns):
    # Create a placeholder column filled with the mean of the imputed data
    placeholder_column = np.nanmean(imputed_data) * np.ones((imputed_data.shape[0], 1))
    # Append the placeholder column to the imputed data
    imputed_data = np.hstack((imputed_data, placeholder_column))

df_imputed = pd.DataFrame(imputed_data, columns=df_numeric.columns)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

# Perform PCA
pca = PCA(n_components=0.95)
principal_components = pca.fit_transform(df_scaled)

# Create a DataFrame with the principal components
principalDf = pd.DataFrame(data=principal_components,
                           columns=['PC' + str(i) for i in range(1, pca.n_components_ + 1)])

print(principalDf.head())

#Get the top 10 contributors of each PCA
pca.fit(df_scaled)
components = pca.components_
components_df = pd.DataFrame(components, columns=df_numeric.columns, index=[f'PC{i+1}' for i in range(components.shape[0])])
# For each principal component, features with the highest absolute contributions
for i in range(components.shape[0]):
    pc = f'PC{i+1}'
    print(f'{pc} top contributing features:')
    loadings = components_df.loc[pc].abs().sort_values(ascending=False)
    print(loadings.head())  # Adjust the number to see more or fewer top features
    print("\n")

top_contributors = pd.DataFrame()
for i, pc in enumerate(components_df.index):
    loadings = components_df.loc[pc].abs().sort_values(ascending=False)
    top_features = loadings.head(10)  # Get top 10 contributing features for each PC
    top_contributors[pc] = top_features.index.values

top_contributors = top_contributors.T
top_contributors.to_excel('pca_top_contributors.xlsx')

pc1_loadings = components_df.loc['PC1'].abs().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=pc1_loadings.values, y=pc1_loadings.index, palette="viridis")
plt.title('Top 10 Contribution of Features to PC1')
plt.xlabel('Absolute Contribution')
plt.ylabel('Features')
plt.show()

principalDf.to_csv("PCA.csv")

##############################################################################

# List of columns you want to keep
columns_to_keep = [
    '_STATE', 'FMONTH', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 
    'MEDCOST1', 'CHECKUP1', 'SLEPTIM1', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 
    'ASTHMA3', 'ASTHNOW', 'CHCCOPD3', 'ADDEPEV3', 'HAVARTH4', 'DIABETE4', 
    'MARITAL', 'RENTHOM1', 'EMPLOY1', 'PREGNANT', 'BLIND', 'DECIDE', 'DIFFWALK', 
    'LCSFIRST', 'LCSLAST', 'LCSNUMCG', 'LCSCTSC1', 'LCSSCNCR', 'LCSCTWHN', 
    'DRNK3GE5', 'MAXDRNKS', 'COVIDPOS', 'COVIDSMP', 'COVIDPRM', 'PDIABTS1', 
    'PREDIAB2', 'DIABTYPE', 'INSULIN1', 'CHKHEMO3', 'EYEEXAM1', 'FEETSORE', 
    'COVIDVA1', 'COVIDNU1', 'COVIDFS1', 'COVIDSE1', 'COPDCOGH', 'COPDFLEM', 
    'COPDBRTH', 'COPDBTST', 'COPDSMOK', 'CNCRDIFF', 'CNCRAGE', 'CNCRTYP2', 
    'CAREGIV1', 'CRGVHRS1', 'CRGVPRB3', 'SDHSTRE1', 'MARIJAN1', 'ASBIRDUC', 
    'QSTLANG', 'MSCODE', '_RFHLTH', '_PHYS14D', '_MENT14D', '_HLTHPLN', '_TOTINDA', 
    '_MICHD', '_ASTHMS1', '_DRDXAR2', '_RACEPR1', '_SEX', '_AGE_G', 'HTIN4', 
    'WTKG3', '_BMI5', '_BMI5CAT', '_CHLDCNT', '_EDUCAG', '_INCOMG1', '_SMOKER3', 
    '_SMOKGRP', '_LCSREC', 'DROCDY4_', '_RFDRHV8'
]

# Keeping only the specified columns
df_filtered = brfss[columns_to_keep]

cm = df_filtered.corr()
hc = cm[(cm >= 0.9) | (cm <= -0.9)]
sns.heatmap(hc, annot=True, cmap='coolwarm', cbar=True)
plt.show()

cm.to_csv("correlation.csv")

# Remove unnecessary columns as specified.
columns_to_remove = ['CRGVPRB3', 'CRGVHRS1', 'CAREGIV1', 'PREDIAB2', 'COVIDFS1', 'COVIDSE1', '_DRDXAR2']
df_filtered.drop(columns=columns_to_remove, inplace=True)

# Fill in missing data as per the given instructions.
fill_values = {
    'MSCODE': '5', 'ASBIRDUC': '7', 'SDHSTRE1': '7', 'CNCRTYP2': 0, 'ASTHNOW': 7, 'ADDEPEV3': 7,
    'LCSFIRST': 777, 'LCSLAST': 777, 'LCSNUMCG': 777, 'LCSCTSC1': 7, 'LCSSCNCR': 7, 'CHCCOPD3': 7,
    'LCSCTWHN': 7, 'DRNK3GE5': 77, 'MAXDRNKS': 77, 'COVIDPOS': 7, 'COVIDVA1': 7, 'ASTHMA3': 7, 'DROCDY4_': 0,
    'COVIDNU1': 7, 'COPDCOGH': 7, 'COPDFLEM': 7, 'COPDBRTH': 7, 'COPDBTST': 7, 'CVDSTRK3': 7, 'RENTHOM1': 7,
    'COPDSMOK': 77, 'CNCRDIFF': 7, 'PREGNANT': 7, '_SMOKGRP': '9', '_LCSREC': '9', 'CVDCRHD4': 7, 'MARITAL': 7,
    'CNCRAGE': 98 , 'POORHLTH':77, 'EMPLOY1': 0, 'BLIND': 7, 'DECIDE': 7, 'DIFFWALK': 7, 'CVDINFR4': 7,
    'COVIDSMP': 7, 'COVIDPRM': 77, '_MICHD': 2, 'GENHLTH': 9, 'PHYSHLTH': 77, 'HAVARTH4': 7, 'DIABETE4': 7, 
    'MENTHLTH': 77, 'MEDCOST1': 7, 'CHECKUP1': 7, 'SLEPTIM1': 77, # Assume manual intervention for specific conditions
}
df_filtered.fillna(value=fill_values, inplace=True)
df_filtered['SDHSTRE1'] = df_filtered['SDHSTRE1'].replace('9', '7')

# Specific conditional operations.
df_filtered['MARIJAN1'] = df_filtered['MARIJAN1'].apply(lambda x: x if x in [str(i) for i in range(1, 31)] + ['88'] else '0')
df_filtered.loc[df_filtered['_SEX'].astype(str) == '1', 'PREGNANT'] = '2'
df_filtered.loc[~df_filtered['CNCRTYP2'].between(1, 97), 'CNCRTYP2'] = 0

# Handle DIABETE4 related conditions.
diabetes_columns = ['PDIABTS1', 'DIABTYPE', 'INSULIN1', 'CHKHEMO3', 'EYEEXAM1', 'FEETSORE']
for column in diabetes_columns:
    df_filtered.loc[df_filtered['DIABETE4'] != '1', column] = 7 if column != 'CHKHEMO3' else 77

# Additional conditions for COVID-related columns.
df_filtered.loc[df_filtered['COVIDPOS'].isin(['1', '3']), 'COVIDSMP'] = df_filtered.loc[df_filtered['COVIDPOS'].isin(['1', '3']), 'COVIDSMP'].fillna('7')
df_filtered.loc[(df_filtered['COVIDPOS'].isin(['1', '3'])) & (df_filtered['COVIDSMP'] == '1'), 'COVIDPRM'] = '77'

# Replacing CNCRDIFF and CNCRTYP2 based on conditions
df_filtered.loc[df_filtered['CNCRDIFF'].isin(['7', '9']), 'CNCRAGE'] = df_filtered.loc[df_filtered['CNCRDIFF'].isin(['7', '9']), 'CNCRAGE'].fillna(98).astype(int)
df_filtered.loc[df_filtered['CNCRDIFF'].isin(['7', '9']), 'CNCRTYP2'] = df_filtered.loc[df_filtered['CNCRDIFF'].isin(['7', '9']), 'CNCRTYP2'].fillna(77).astype(int)

#Handle CVDINFR4 and _MICHD relationship
df_filtered.loc[df_filtered['CVDINFR4'].isin(['7', '9']), '_MICHD'] = df_filtered.loc[df_filtered['CVDINFR4'].isin(['7', '9']), '_MICHD'].fillna(2).astype(int)

# Function to replace outliers with the column mean
def replace_outliers_with_mean(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    column_mean = df[column].mean()
    
    # Identify outliers and replace them with the column mean
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    df.loc[outliers, column] = column_mean
    
    return df

# Replace outliers in 'WTKG3' with its mean
df_filtered = replace_outliers_with_mean(df_filtered, 'WTKG3')
# Replace outliers in 'HTIN4' with its mean
df_filtered = replace_outliers_with_mean(df_filtered, 'HTIN4')

# Calculate mean for 'HTIN4' and 'WTKG3' and fill missing values.
mean_HTIN4 = df_filtered['HTIN4'].mean()
mean_WTKG3 = df_filtered['WTKG3'].mean()
df_filtered['HTIN4'].fillna(mean_HTIN4, inplace=True)
df_filtered['WTKG3'].fillna(mean_WTKG3, inplace=True)

# Clear and recalculate BMI ('_BMI5') and BMI categories ('_BMI5CAT').
df_filtered['_BMI5'] = ((df_filtered['WTKG3'] / 1000) / ((df_filtered['HTIN4'] * 0.0254) ** 2)).round(1)
df_filtered['_BMI5CAT'] = pd.cut(df_filtered['_BMI5'], bins=[0, 5.825, 11.65, 17.475, 23.3], labels=[1,2,3,4])

# Handling the first NaN in '_CHLDCNT'.
if df_filtered['_CHLDCNT'].isna().any():
    first_nan_index = df_filtered['_CHLDCNT'].isna().idxmax()
    df_filtered.at[first_nan_index, '_CHLDCNT'] = 9
    
#Handle "_SMOKER3" and "_SMOKGRP" relationship
df_filtered.loc[df_filtered['_SMOKER3'] == '9', '_SMOKGRP'] = df_filtered.loc[df_filtered['_SMOKER3'] == '9', '_SMOKGRP'].fillna('9')    

# Export the cleaned and transformed DataFrame.
df_filtered.to_csv("Cleaned_Data.csv", index=False)


##############################################################################

##Modeling to predict Diabetes
##Logistic Regression

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


#Bringing data to the usable format
# If values in 'DIABETE4' are 1 or 2, replace them with 1; all other values will be set to 0.
df_filtered['DIABETE4'] = df_filtered['DIABETE4'].apply(lambda x: 1 if x in [1, 2] else 0)

################################################################################################################################
df_dummy=pd.get_dummies(df_filtered,drop_first=True)
X = df_dummy.drop(['DIABETE4','_STATE','FMONTH', 'ASTHNOW','MARITAL' ,'RENTHOM1', 'EMPLOY1','QSTLANG', 'MSCODE_2.0','MSCODE_3.0','MSCODE_5.0','MSCODE_5','_HLTHPLN','_TOTINDA', '_BMI5CAT_2','_BMI5CAT_3','_BMI5CAT_4', '_CHLDCNT', '_EDUCAG', '_INCOMG1'], axis=1)
Y = df_dummy['DIABETE4']

#X = df_filtered[['_STATE', 'FMONTH', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH',
#        'MEDCOST1', 'CHECKUP1', 'SLEPTIM1', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3',
#        'ASTHMA3', 'ASTHNOW', 'CHCCOPD3', 'ADDEPEV3', 'HAVARTH4', 'MARITAL',
#        'RENTHOM1', 'EMPLOY1', 'PREGNANT', 'BLIND', 'DECIDE', 'DIFFWALK',
#        'LCSFIRST', 'LCSLAST', 'LCSNUMCG', 'LCSCTSC1', 'LCSSCNCR', 'LCSCTWHN',
#        'DRNK3GE5', 'MAXDRNKS', 'COVIDPOS', 'COVIDSMP', 'COVIDPRM', 'PDIABTS1',
#        'DIABTYPE', 'INSULIN1', 'CHKHEMO3', 'EYEEXAM1', 'FEETSORE', 'COVIDVA1',
#        'COVIDNU1', 'COPDCOGH', 'COPDFLEM', 'COPDBRTH', 'COPDBTST', 'COPDSMOK',
#        'CNCRDIFF', 'CNCRAGE', 'CNCRTYP2', 'SDHSTRE1', 'MARIJAN1', 'ASBIRDUC',
#        'QSTLANG', 'MSCODE', '_RFHLTH', '_PHYS14D', '_MENT14D', '_HLTHPLN',
#        '_TOTINDA', '_MICHD', '_ASTHMS1', '_RACEPR1', '_SEX', '_AGE_G', 'HTIN4',
#        'WTKG3', '_BMI5', '_BMI5CAT', '_CHLDCNT', '_EDUCAG', '_INCOMG1',
#        '_SMOKER3', '_SMOKGRP', '_LCSREC', 'DROCDY4_', '_RFDRHV8']]
#Y = df_filtered['DIABETE4']

# Perform train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

model=LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions on the training and test data
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Binarize predictions based on a 0.5 cutoff
train_predictions_binary = [1 if p > 0.7 else 0 for p in train_predictions]
test_predictions_binary = [1 if p > 0.7 else 0 for p in test_predictions]

# Calculate accuracy and F1 score for training data
train_accuracy = accuracy_score(Y_train, train_predictions_binary)
train_f1 = f1_score(Y_train, train_predictions_binary)

# Calculate accuracy and F1 score for test data
test_accuracy = accuracy_score(Y_test, test_predictions_binary)
test_f1 = f1_score(Y_test, test_predictions_binary)

# Display the model evaluation metrics
print(f"Training Data: Accuracy = {train_accuracy:.3f}, F1 Score = {train_f1:.3f}")
print(f"Testing Data: Accuracy = {test_accuracy:.3f}, F1 Score = {test_f1:.3f}")
#85% Accuracy; 0.012 f1 score

#Feature Selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import make_scorer

f1_scorer = make_scorer(f1_score)
sfs = SFS(model, 
          k_features='best', 
          forward=True, 
          scoring=f1_scorer,
          cv=5)
sfs = sfs.fit(X_train, Y_train)

X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)

# Refit the logistic regression model with selected features
model.fit(X_train_sfs, Y_train)

y_pred_sfs = model.predict(X_test_sfs)

f1_score_sfs = f1_score(Y_test, y_pred_sfs)
print(f"F1-score after forward selection: {f1_score_sfs}")


################################################################################################################################

##Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree

 
 
# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
 
# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
 
# Fit the model to the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the training set
y_pred_train = dt_classifier.predict(X_train)
 
# Calculate accuracy and other performance metrics
accuracy = accuracy_score(y_train, y_pred_train)
report = classification_report(y_train, y_pred_train)
 
# Print the performance metrics
print(f'Accuracy: {accuracy}') #1
print(f'Classification Report:\n{report}') #f1 score 1


# Make predictions on the testing set
y_pred = dt_classifier.predict(X_test)
 
# Calculate accuracy and other performance metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
 
# Print the performance metrics
print(f'Accuracy: {accuracy}') #0.78
print(f'Classification Report:\n{report}') #f1 score 0.87

plot_tree(dt_classifier)
dt_classifier.tree_.max_depth #62 is the max depth

##tuning the hyper parameters to avoid overfitting
parameter_grid={'max_depth': range(1,10), 'min_samples_split': range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt_classifier,parameter_grid,verbose=3,scoring='f1',cv=10)
grid.fit(X_train,y_train)

#best parameters
grid.best_params_


#Build decision Tress using best parameters
dt=DecisionTreeClassifier(max_depth=9, min_samples_split=5, random_state=1) #initialize
dt.fit(X_train,y_train) #train meaning find tress


# Make predictions on the training set
y_pred_train = dt.predict(X_train)
 
# Calculate accuracy and other performance metrics
accuracy = accuracy_score(y_train, y_pred_train)
report = classification_report(y_train, y_pred_train)
 
# Print the performance metrics
print(f'Accuracy: {accuracy}') #0.86
print(f'Classification Report:\n{report}') #f1 score = 0.92


# Make predictions on the testing set
y_pred = dt.predict(X_test)
 
# Calculate accuracy and other performance metrics
accuracy = accuracy_score(y_test, y_pred) 
report = classification_report(y_test, y_pred) 
 
# Print the performance metrics
print(f'Accuracy: {accuracy}') #0.85
print(f'Classification Report:\n{report}') #f1 score = 0.92

plot_tree(dt)


################################################################################################################################
##Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
 
# Create a smaller subset of the training data for the grid search
X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)
 
# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
 
# Define a smaller grid of hyperparameters to search
param_grid = {

    'n_estimators': [100, 200],

    'max_depth': [None, 10],

    'min_samples_split': [2, 10],

    'min_samples_leaf': [1, 4]

}
 
# Use F1 score as the scoring metric, with 'weighted' average for multiclass classification
f1_scorer = make_scorer(f1_score, average='weighted')

# Setup the grid search with reduced CV folds and parallel jobs
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=f1_scorer, cv=3, n_jobs=-1)
 
# Fit the grid search model on the subset
grid_search.fit(X_train_sub, y_train_sub)
 
# Find the best parameters and use them to make predictions on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
 
# Calculate accuracy and weighted F1 score
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
 
# Print the performance metrics
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}') #0.85
print(f'Weighted F1 Score: {weighted_f1}') #0.80

################################################################################################################################
##Final Insights

feature_names = X_train.columns
feature_importances = dt.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

from sklearn.tree import export_text
tree_rules = export_text(dt, feature_names=list(feature_names))
print(tree_rules)

df = brfss
df['_AGE_G'] = df['_AGE_G'].map({
    1: '18-24',
    2: '25-34',
    3: '35-44',
    4: '45-54',
    5: '55-64',
    6: '65 or older'
})

# Rename DIABETE4 categories
df['DIABETE4'] = df['DIABETE4'].map({
    1: 'Diabetic',
    2: 'Diabetic',
    4: 'Diabetic',
    3: 'Non-Diabetic',
    7: 'Non-Diabetic',
    9: 'Non-Diabetic'
    })

# Rename _BMI5CAT categories
df['_BMI5CAT'] = df['_BMI5CAT'].map({
    1: 'Underweight',
    2: 'Normal Weight',
    3: 'Overweight',
    4: 'Obese'
})

# Rename CVDSTRK3 categories
df['CVDSTRK3'] = df['CVDSTRK3'].map({
    1: 'Stroke History',
    2: 'No Stroke History',
    7: 'No Stroke History',
    9: 'No Stroke History'
})

# Rename CHCCOPD3 categories
df['CHCCOPD3'] = df['CHCCOPD3'].map({
    1: 'Pulmonary Disease History',
    2: 'No Pulmonary Disease History',
    7: 'No Pulmonary Disease History',
    9: 'No Pulmonary Disease History'
})

# Rename _RFHLTH categories
df['_RFHLTH'] = df['_RFHLTH'].map({
    1: 'Good or Better Health',
    2: 'Fair or Pool Health',
    9: 'Good or Better Health'
})

# Rename _RACE1 categories
df['_RACE1'] = df['_RACE1'].map({
    1: 'White',
    2: 'African American',
    3: 'American Indian or Alaskan Native',
    4: 'Asian',
    5: 'Native Hawaiian',
    7: 'Multiracial',
    8: 'Hispanic',
    9: 'Refused'
    })

#Age group Distribution
age_group_counts = df.groupby(['_AGE_G', 'DIABETE4']).size().unstack(fill_value=0)
age_group_percentages = age_group_counts.div(age_group_counts.sum(axis=1), axis=0) * 100
age_group_percentages.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Percentage of Diabetics in Different Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Percentage')
plt.legend(title='Diabetes Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# BMI Category Distribution
bmi_category_counts = df.groupby(['_BMI5CAT', 'DIABETE4']).size().unstack(fill_value=0)
bmi_category_percentages = bmi_category_counts.div(bmi_category_counts.sum(axis=1), axis=0) * 100
bmi_category_percentages.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Percentage of Diabetics by BMI Category')
plt.xlabel('BMI Category')
plt.ylabel('Percentage')
plt.legend(title='Diabetes Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

stroke_history_counts = df.groupby(['CVDSTRK3', 'DIABETE4']).size().unstack(fill_value=0)
stroke_history_percentages = stroke_history_counts.div(stroke_history_counts.sum(axis=1), axis=0) * 100
stroke_history_percentages.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Percentage of Diabetics by Stroke History')
plt.xlabel('Stroke History')
plt.ylabel('Percentage')
plt.legend(title='Diabetes Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


pd_history_counts = df.groupby(['CHCCOPD3', 'DIABETE4']).size().unstack(fill_value=0)
pd_history_percentages = pd_history_counts.div(pd_history_counts.sum(axis=1), axis=0) * 100
pd_history_percentages.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Percentage of Diabetics by Pulmonary Disease History')
plt.xlabel('Pulmonary Disease History')
plt.ylabel('Percentage')
plt.legend(title='Diabetes Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Race Distribution
race_counts = df.groupby(['_RACE1', 'DIABETE4']).size().unstack(fill_value=0)
race_percentages = race_counts.div(race_counts.sum(axis=1), axis=0) * 100
sorted_race_percentages = race_percentages.sort_values('Diabetic', ascending=True)
sorted_race_percentages.plot(kind='barh', stacked=True, figsize=(10, 6))
plt.title('Percentage of Diabetics by Race')
plt.ylabel('Race Category')
plt.xlabel('Percentage')
plt.legend(title='Diabetes Status')
plt.tight_layout()
plt.show()

################################################################################################################################
##Modeling to predict Heart disorders
##Logistic Regression

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


#Bringing data to the usable format
df_filtered['CVDCRHD4'] = df_filtered['CVDCRHD4'].apply(lambda x: 1 if x == 1 else 0)
df_filtered['_MICHD'] = df_filtered['_MICHD'].apply(lambda x: 1 if x == 1 else 0)

################################################################################################################################

df_dummy=pd.get_dummies(df_filtered,drop_first=True)
##X = df_dummy.drop('CVDCRHD4', axis=1)
X = df_dummy.drop(['CVDCRHD4', '_MICHD', 'CVDINFR4','_STATE','FMONTH', 'ASTHNOW','MARITAL' ,'RENTHOM1', 'EMPLOY1','QSTLANG', 'MSCODE_2.0','MSCODE_3.0','MSCODE_5.0','MSCODE_5','_HLTHPLN','_TOTINDA', '_BMI5CAT_2','_BMI5CAT_3','_BMI5CAT_4', '_CHLDCNT', '_EDUCAG', '_INCOMG1'], axis=1)
Y = df_dummy['CVDCRHD4']

# Perform train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

model=LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions on the training and test data
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Binarize predictions based on a 0.5 cutoff
train_predictions_binary = [1 if p > 0.7 else 0 for p in train_predictions]
test_predictions_binary = [1 if p > 0.7 else 0 for p in test_predictions]

# Calculate accuracy and F1 score for training data
train_accuracy = accuracy_score(Y_train, train_predictions_binary)
train_f1 = f1_score(Y_train, train_predictions_binary)

# Calculate accuracy and F1 score for test data
test_accuracy = accuracy_score(Y_test, test_predictions_binary)
test_f1 = f1_score(Y_test, test_predictions_binary)

# Display the model evaluation metrics
print(f"Training Data: Accuracy = {train_accuracy:.3f}, F1 Score = {train_f1:.3f}")
print(f"Testing Data: Accuracy = {test_accuracy:.3f}, F1 Score = {test_f1:.3f}")
#94% Accuracy; 0.0 f1 score

#Feature Selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import make_scorer

f1_scorer = make_scorer(f1_score)
sfs = SFS(model, 
          k_features='best', 
          forward=True, 
          scoring=f1_scorer,
          cv=5)
sfs = sfs.fit(X_train, Y_train)

X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)

# Refit the logistic regression model with selected features
model.fit(X_train_sfs, Y_train)

y_pred_sfs = model.predict(X_test_sfs)

f1_score_sfs = f1_score(Y_test, y_pred_sfs)
print(f"F1-score after forward selection: {f1_score_sfs}")


################################################################################################################################

##Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
 
# Fit the model to the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the training set
y_pred_train = dt_classifier.predict(X_train)
 
# Calculate accuracy and other performance metrics
accuracy = accuracy_score(y_train, y_pred_train)
report = classification_report(y_train, y_pred_train)
 
# Print the performance metrics
print(f'Accuracy: {accuracy}') #1
print(f'Classification Report:\n{report}') #f1 score 1


# Make predictions on the testing set
y_pred = dt_classifier.predict(X_test)
 
# Calculate accuracy and other performance metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
 
# Print the performance metrics
print(f'Accuracy: {accuracy}') #0.89
print(f'Classification Report:\n{report}') #f1score = 0.94

plot_tree(dt_classifier)
dt_classifier.tree_.max_depth #52 is the max depth

# Create a smaller subset of the training data for the grid search
X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

##tuning the hyper parameters to avoid overfitting
parameter_grid={'max_depth': range(1,10), 'min_samples_split': range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt_classifier,parameter_grid,verbose=3,scoring='f1',cv=10)
grid.fit(X_train_sub,y_train_sub)

#best parameters
grid.best_params_


#Build decision Tress using best parameters
dt=DecisionTreeClassifier(max_depth=9, min_samples_split=2, random_state=1) #initialize
dt.fit(X_train,y_train) #train meaning find tress


# Make predictions on the training set
y_pred_train = dt.predict(X_train)
 
# Calculate accuracy and other performance metrics
accuracy = accuracy_score(y_train, y_pred_train)
report = classification_report(y_train, y_pred_train)
 
# Print the performance metrics
print(f'Accuracy: {accuracy}') #0.94
print(f'Classification Report:\n{report}') #f1 score = 0.97


# Make predictions on the testing set
y_pred = dt.predict(X_test)
 
# Calculate accuracy and other performance metrics
accuracy = accuracy_score(y_test, y_pred) 
report = classification_report(y_test, y_pred) 
 
# Print the performance metrics
print(f'Accuracy: {accuracy}') #0.94
print(f'Classification Report:\n{report}') #f1 score = 0.97

plot_tree(dt)

################################################################################################################################
##Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score
 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
 
# Create a smaller subset of the training data for the grid search
X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)
 
# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
 
# Define a smaller grid of hyperparameters to search
param_grid = {

    'n_estimators': [100, 200],

    'max_depth': [None, 10],

    'min_samples_split': [2, 10],

    'min_samples_leaf': [1, 4]

}
 
# Use F1 score as the scoring metric, with 'weighted' average for multiclass classification
f1_scorer = make_scorer(f1_score, average='weighted')

# Setup the grid search with reduced CV folds and parallel jobs
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=f1_scorer, cv=3, n_jobs=-1)
 
# Fit the grid search model on the subset
grid_search.fit(X_train_sub, y_train_sub)
 
# Find the best parameters and use them to make predictions on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
 
# Calculate accuracy and weighted F1 score
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
 
# Print the performance metrics
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}') #0.93
print(f'Weighted F1 Score: {weighted_f1}') #0.91

################################################################################################################################
##Final Insights

feature_names = X_train.columns
feature_importances = dt.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

from sklearn.tree import export_text
tree_rules = export_text(dt, feature_names=list(feature_names))
print(tree_rules)


# Rename Heart Disease Patient categories
df['_MICHD'] = df['_MICHD'].map({
    1: 'Heart Disease Patient',
    2: 'Healthy Patient'
    })

# Rename COVID categories
df['COVIDPOS'] = df['COVIDPOS'].map({
    1: 'Tested Positive',
    3: 'Tested Positive',
    2: 'Tested Negative',
    7: 'Not Tested',
    9: 'Not Tested'
    })

# Rename Smoking Groups categories
df['_SMOKGRP'] = df['_SMOKGRP'].map({
    1.0: 'Current Smoker: 20+ Pack Years',
    2.0: 'Former smoker: 20+ Pack Years',
    3.0: 'Other Smokers',
    4.0: 'Non Smokers',
    })



df['COVIDPRM'] = df['COVIDPRM'].map({
    1: "Tiredness or fatigue",
    2: "Memory problems",
    3: "Shortness of breath",
    4: "Muscle pain",
    5: "Fast-beating or pounding heart or chest pain",
    6: "Dizziness on standing",
    7: "Depression, anxiety, or mood changes",
    8: "Symptoms that get worse after physical or mental activities",
    9: "No long-term symptoms",
    10: "Loss of taste or smell",
    11: "Some other symptom",
    77: "Some other symptom",
    99: "Some other symptom"
})

michd_counts = df.groupby(['DIABETE4', '_MICHD']).size().unstack(fill_value=0)
michd_percentages = michd_counts.div(michd_counts.sum(axis=1), axis=0) * 100
michd_percentages.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Percentage of MI/CHD Conditions among Diabetic Status')
plt.xlabel('Diabetes Status')
plt.ylabel('Percentage')
plt.legend(title='MI/CHD Status', loc='upper right')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

df_filtered_menthlth = df[(df['MENTHLTH'] >= 1) & (df['MENTHLTH'] <= 30)]
micd_counts = df_filtered_menthlth.groupby('MENTHLTH')['_MICHD'].value_counts(normalize=True).unstack(fill_value=0) * 100
micd_percentages = micd_counts['Heart Disease Patient']
plt.figure(figsize=(12, 6))
micd_percentages.plot(kind='bar')
plt.title('Percentage of People with Heart Disease by Mental Health Days')
plt.xlabel('Number of Bad Mental Health Days')
plt.ylabel('Percentage of Heart Disease Patients')
plt.tight_layout()
plt.show()

# COVIDPOS vs _MICHD
covidpos_michd_counts = df.groupby(['COVIDPOS', '_MICHD']).size().unstack(fill_value=0)
covidpos_michd_percentages = covidpos_michd_counts.div(covidpos_michd_counts.sum(axis=1), axis=0) * 100
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
covidpos_michd_percentages.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('COVID-19 Positive Status vs. Heart Disease')
plt.xlabel('COVID-19 Positive Status')
plt.ylabel('Percentage')
plt.xticks(rotation=45)


# _SMOKGRP vs _MICHD
smokgrp_michd_counts = df.groupby(['_SMOKGRP', '_MICHD']).size().unstack(fill_value=0)
smokgrp_michd_percentages = smokgrp_michd_counts.div(smokgrp_michd_counts.sum(axis=1), axis=0) * 100
plt.figure(figsize=(14, 6))
smokgrp_michd_percentages.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Smoking Group vs. Heart Disease')
plt.xlabel('Smoking Group')
plt.ylabel('Percentage')
plt.xticks(wrap=True)
plt.tight_layout()
plt.show()

michd_counts = df.groupby(['COVIDPRM', '_MICHD']).size().unstack(fill_value=0)
michd_percentages = michd_counts.div(michd_counts.sum(axis=1), axis=0) * 100
sorted_michd_percentages = michd_percentages.sort_values(by="Heart Disease Patient", ascending=False)
sorted_michd_percentages.plot(kind='barh', stacked=True, figsize=(10, 8))
plt.title('People with Heart Disease by CPVOD-19 Symptoms')
plt.yticks(wrap=True)
plt.legend(title='Heart Disease Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#END################################################################################################################################