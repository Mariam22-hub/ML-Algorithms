# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn import tree

# %%
data = pd.read_csv("drug.csv")
print ("\ndata\n",data.head(20))

# %%
datanull= data.isnull().any(axis = 1) 
data[datanull]
count_null_rows = datanull.sum()

print("Number of rows with at least one null value:", count_null_rows)

# %%
categorical_columns = data.select_dtypes(include='object').columns
numerical_columns = data.select_dtypes(exclude='object').columns
#replace categorical missing values with the mode and the numercial missing values with the mean

data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())


# %%
# we notice that missing values have been replaced
print ("\ndata\n",data.head(20))

# %%
categorical_columns = data.select_dtypes(include='object').columns
numerical_columns = data.select_dtypes(exclude='object').columns

# %%
sex_encoder = LabelEncoder()
bp_encoder = LabelEncoder()
cholesterol_encoder = LabelEncoder()
# Fit and transform the categorical variables using the label encoders
data['Sex'] = sex_encoder.fit_transform(data['Sex'])
data['BP'] = bp_encoder.fit_transform(data['BP'])
data['Cholesterol'] = cholesterol_encoder.fit_transform(data['Cholesterol'])


# %%
print ("\ndata\n",data.head(20))

# %%
x = data.drop(columns=['Drug'])
y = data[['Drug']]

# %%
from sklearn.model_selection import train_test_split

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

bestAccuracy = 0
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=np.random.randint(100 + 2*i))
    
    clf = tree.DecisionTreeClassifier()
    
    clf = clf.fit(X_train, y_train)
    
    plt.figure(figsize=(10, 7))
    tree.plot_tree(clf, filled=True, feature_names=X_train.columns)
    plt.title(f"Iteration {i+1}: Training set size - {len(X_train)}, Testing set size - {len(X_test)}")
    plt.show()
    num_nodes = clf.tree_.node_count
    num_leaves = clf.get_n_leaves()
    tree_depth = clf.get_depth()
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)*100
    if accuracy>bestAccuracy:
        bestAccuracy = accuracy
        clfBest = tree.DecisionTreeClassifier()
        clfBest = clf
    print(f"Iteration {i+1}: Accuracy - {accuracy:.4f}, Number of nodes - {num_nodes}, Number of leaves - {num_leaves}, Tree depth - {tree_depth}")
print(f"Best Accuracy: {bestAccuracy}")
plt.figure(figsize=(10, 7))
tree.plot_tree(clfBest, filled=True, feature_names=X_train.columns)
plt.title(f"Best Decision tree")
plt.show()

# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fpdf import FPDF
from tabulate import tabulate
import matplotlib.backends.backend_pdf as pdf_backend
training_sizes = np.arange(0.3, 0.8, 0.1)

mean_accuracies = []
max_accuracies = []
min_accuracies = []
mean_nodes = []
max_nodes = []
min_nodes = []

for train_size in training_sizes:
    accuracies = []
    nodes = []
    
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_size), random_state=np.random.randint(100))
        
        clf = DecisionTreeClassifier()
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        num_nodes = clf.tree_.node_count
        
        # Append to the lists
        accuracies.append(accuracy)
        nodes.append(num_nodes)

    mean_accuracies.append(np.mean(accuracies)*100)
    max_accuracies.append(np.max(accuracies)*100)
    min_accuracies.append(np.min(accuracies)*100)
    
    mean_nodes.append(np.mean(nodes))
    max_nodes.append(np.max(nodes))
    min_nodes.append(np.min(nodes))

report_data = {
    'Training Size': training_sizes,
    'Mean Accuracy': mean_accuracies,
    'Max Accuracy': max_accuracies,
    'Min Accuracy': min_accuracies,
    'Mean Nodes': mean_nodes,
    'Max Nodes': max_nodes,
    'Min Nodes': min_nodes
}
#save report to excel sheet
report_df = pd.DataFrame(report_data)

report_df.to_excel('report.xlsx', index=False, sheet_name='report')
#save report to pdf
latex_table = data.to_latex(index=False)
with pdf_backend.PdfPages("report.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.table(cellText=report_df.values, colLabels=report_df.columns, cellLoc='center', loc='center')
    pdf.savefig(fig, bbox_inches='tight')

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(training_sizes * 100, mean_accuracies, label='Mean Accuracy')
plt.plot(training_sizes * 100, max_accuracies, label='Max Accuracy')
plt.plot(training_sizes * 100, min_accuracies, label='Min Accuracy')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Training Set Size')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_sizes * 100, mean_nodes, label='Mean Nodes')
plt.plot(training_sizes * 100, max_nodes, label='Max Nodes')
plt.plot(training_sizes * 100, min_nodes, label='Min Nodes')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Number of Nodes')
plt.title('Number of Nodes vs. Training Set Size')
plt.legend()

plt.tight_layout()
plt.show()

# %%
