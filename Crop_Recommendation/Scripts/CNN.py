#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model_name = "CNN"


# In[2]:


#df = pd.read_csv("Data_Processed/Recommendation.csv")
df = pd.read_csv("../Data/Crop_production 2.csv")
df.info()


# In[3]:


df = df.drop(["Unnamed: 0"],axis=1)
df.info()


# In[4]:


# for i in df.columns:
#     print(i,df[i].unique(),df[i].nunique())


# In[5]:


def outlier_info(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_count = outliers.shape[0]
    total_count = df.shape[0]
    outlier_percentage = (outlier_count / total_count) * 100

    return outlier_count, outlier_percentage

def outlier_remover(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df.loc[df[column] < lower_bound, column] = lower_bound
    df.loc[df[column] > upper_bound, column] = upper_bound
    return df
# for i in df.columns:
#     if df[i].dtype != "object":
#         sns.boxplot(df[i])
#         plt.show()
#         print(f"Outlier counter and percentage for {i}: {outlier_info(df, i)}")


# In[6]:


df["Area_in_hectares"] = np.log(df["Area_in_hectares"])
df["Production_in_tons"] = np.log(df["Production_in_tons"])
df["Yield_ton_per_hec"] = np.log(df["Yield_ton_per_hec"])


df = outlier_remover(df,"N")
df = outlier_remover(df,"P")
df = outlier_remover(df,"K")
df = outlier_remover(df,"pH")
df = outlier_remover(df,"rainfall")
df = outlier_remover(df,"temperature")
df = outlier_remover(df,"Area_in_hectares")
df = outlier_remover(df,"Production_in_tons")
df = outlier_remover(df,"Yield_ton_per_hec")

# for i in df.columns:
#     if df[i].dtype != "object":
#         sns.boxplot(df[i])
#         plt.show()
#         print(f"Outlier counter and percentage for {i}: {outlier_info(df, i)}")


# In[7]:


# Plotting the class distribution
# plt.figure(figsize=(10, 6))
# sns.countplot(x=df['Crop'])
# plt.title('Class Distribution of Crop')
# plt.xlabel('Crop')
# plt.ylabel('Count')
# plt.xticks(rotation=90)
# plt.show()


# In[8]:


tt = df["Crop"].value_counts()
# print(tt)
df = df[df["Crop"] != "apple"]
df = df[df["Crop"] != "coffee"]
tt = df["Crop"].value_counts()
# print(tt)


# In[9]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

C = ["State_Name","Crop_Type","Crop"]
le_D = {}
for i in C:
    le_D[i] = LabelEncoder()
    df[i] = le_D[i].fit_transform(df[i])

bulk_scaler = StandardScaler()
features_to_scale = ["State_Name","Crop_Type","N","P","K","pH","rainfall","temperature","Area_in_hectares","Production_in_tons","Yield_ton_per_hec"]
#features_to_scale = ["N","P","K","pH","rainfall","temperature","Area_in_hectares","Yield_ton_per_hec"]
df[features_to_scale] = bulk_scaler.fit_transform(df[features_to_scale])

df.info()
df.nunique()


# In[10]:


plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[11]:


from sklearn.decomposition import PCA

t = df[["Production_in_tons","Area_in_hectares"]].values
pca = PCA(n_components=1)
X_pca = pca.fit_transform(t)

df = df.drop(["Production_in_tons","Area_in_hectares","Yield_ton_per_hec"],axis=1)
df["PCA"] = X_pca

df.info()


# In[12]:


import torch
from torch.utils.data import Dataset, DataLoader, random_split,TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from imblearn.over_sampling import *
data = df


# In[13]:


class CropRecommendationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CropRecommendationModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128 * input_size, num_classes) 

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x


# In[14]:


from sklearn.datasets import make_classification
from sklearn.decomposition import PCA,IncrementalPCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve,precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.manifold import TSNE
#import shap
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.calibration import calibration_curve
from scipy.interpolate import make_interp_spline

def plot_tsne_pca(features, targets, target_names,name,sample_size=2000):
    # Subsample the data for faster computation
    if len(features) > sample_size:
        idx = np.random.choice(len(features), sample_size, replace=False)
        features_sampled = features[idx]
        targets_sampled = targets[idx]
    else:
        features_sampled = features
        targets_sampled = targets

    # TSNE
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(features_sampled)
    
    for i, target_name in enumerate(target_names):
        plt.scatter(tsne_results[targets_sampled == i, 0], tsne_results[targets_sampled == i, 1], label=le_D["Crop"].inverse_transform([target_name])[0], s=10)
    plt.title("t-SNE")
    plt.grid()
    #plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.show()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.savefig(f'../Results/{name}_tsne.png', bbox_inches='tight')
    plt.close()
    fig_legend = plt.figure(figsize=(10, 10))
    legend = plt.figlegend(handles, labels, loc='center', ncol=10)
    for label in legend.get_texts():
        label.set_ha('right')
    fig_legend.savefig(f'../Results/{name}_tsne_legend.png', bbox_inches='tight')
    plt.close()
    from IPython.display import Image, display
    display(Image(f'../Results/{name}_tsne.png'))
    display(Image(f'../Results/{name}_tsne_legend.png'))
    
    # PCA
    pca = IncrementalPCA(n_components=2)
    pca_results = pca.fit_transform(features_sampled)
    
    for i, target_name in enumerate(target_names):
        plt.scatter(pca_results[targets_sampled == i, 0], pca_results[targets_sampled == i, 1], label=le_D["Crop"].inverse_transform([target_name])[0], s=10)
    plt.title("PCA")
    plt.grid()
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.show()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.savefig(f'../Results/{name}_pca.png', bbox_inches='tight')
    plt.close()
    fig_legend = plt.figure(figsize=(10, 10))
    legend = plt.figlegend(handles, labels, loc='center', ncol=10)
    for label in legend.get_texts():
        label.set_ha('right')
    fig_legend.savefig(f'../Results/{name}_pca_legend.png', bbox_inches='tight')
    plt.close()
    from IPython.display import Image, display
    display(Image(f'../Results/{name}_pca.png'))
    display(Image(f'../Results/{name}_pca_legend.png'))

def plot_radar_chart(metrics_df,name):
    labels = metrics_df['Class']
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for metric in metrics_df.columns[1:]:
        values = metrics_df[metric].tolist()
        values += values[:1]
        ax.plot(angles, values, label=metric)
        ax.fill(angles, values, alpha=0.1)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Metrics Radar Chart')
    plt.savefig(f"../Results/{name}_radar_chart.png")
    plt.show()
    
def compute_metrics(val_target, val_predicted, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(val_target, val_predicted, average=None, labels=range(num_classes),zero_division=1)
    metrics = {
        'Class': range(num_classes),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    return pd.DataFrame(metrics)

def train_validation(train_features, test_features, train_target, test_target,name,num_epochs=100):

    input_size = train_features.shape[1]
    hidden_size = 64
    num_classes = len(data['Crop'].unique())

    model = CropRecommendationModel(input_size, hidden_size, num_classes).to("cuda")
    #model = torch.compile(model)
    # num_epochs = 1100
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []


    for epoch in range(num_epochs):
        # Training
        model.train()
        outputs = model(train_features)
        loss = criterion(outputs, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        with torch.no_grad():
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct = (train_predicted == train_target).sum().item()
            train_accuracy = train_correct / train_target.size(0)
            train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features)
            test_loss = criterion(test_outputs, test_target)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_correct = (test_predicted == test_target).sum().item()
            test_accuracy = test_correct / test_target.size(0)
            test_losses.append(test_loss.item())
            test_accuracies.append(test_accuracy)

    # Plotting all graphs in a single figure
    
    # Loss and Accuracy Graphs
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss', color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Epoch vs Loss Plot")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), test_losses, label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Epoch vs Loss Plot")
    plt.legend()
    plt.grid()
    plt.savefig(f'../Results/{name}_loss_plot.png')
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy', color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Epoch vs Accuracy Plot")
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), test_accuracies, label='Val Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Epoch vs Accuracy Plot")
    plt.legend()
    plt.grid()
    plt.savefig(f'../Results/{name}_accu_plot.png')
    plt.show()
    
    # Confusion Matrix
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_features)
        _, test_predicted = torch.max(test_outputs.data, 1)
    
    test_predicted_names = le_D["Crop"].inverse_transform(test_predicted.cpu())
    test_target_names = le_D["Crop"].inverse_transform(test_target.cpu())
    cm = confusion_matrix(test_target_names, test_predicted_names)
    colors = ["#000000", "#ff0000"]  # Black and Red
    cmap = sns.color_palette(colors)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, linewidths=0.5, linecolor='white')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'../Results/{name}_cm_plot.png')
    plt.show()
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_features)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == test_target).sum().item() / test_target.size(0)

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    all_metrics = []
    fold_metrics = compute_metrics(test_target.cpu().numpy(), test_predicted.cpu().numpy(), num_classes)
    all_metrics.append(fold_metrics)
    avg_metrics = pd.concat(all_metrics).groupby('Class').mean().reset_index()
    avg_metrics['Class'] = le_D['Crop'].inverse_transform(avg_metrics['Class'].astype(int))
    plot_radar_chart(avg_metrics,name)
    
    val_target_onehot = nn.functional.one_hot(test_target, num_classes=num_classes).cpu().numpy()
    roc_auc_dict = {}

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(val_target_onehot[:, i], test_outputs.cpu().numpy()[:, i])
        roc_auc_dict[i] = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {le_D["Crop"].inverse_transform([i])[0]} (area = {roc_auc_dict[i]:0.2f})')

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid()

    # Capture the legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Save the ROC curve plot without the legend
    plt.savefig(f'../Results/{name}_roc_curve.png', bbox_inches='tight')
    plt.close()

    # Create a new figure for the legend
    fig_legend = plt.figure(figsize=(8, 6))
    plt.axis('off')
    legend = plt.figlegend(handles, labels, loc='center', ncol=5)
    for label in legend.get_texts():
        label.set_ha('right')
    fig_legend.savefig(f'../Results/{name}_roc_curve_legend.png', bbox_inches='tight')
    plt.close()

    # Show ROC curve and legend separately
    from IPython.display import Image, display
    display(Image(f'../Results/{name}_roc_curve.png'))
    display(Image(f'../Results/{name}_roc_curve_legend.png'))


    plot_tsne_pca(train_features.cpu(), train_target.cpu(), target_names=data['Crop'].unique(),name=name)
    return model, criterion

def cross_check(features, target, name, k=10, num_epochs=100):
    learning_rate = 0.001
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        train_features, val_features = features[train_idx], features[val_idx]
        train_target, val_target = target[train_idx], target[val_idx]
        input_size = train_features.shape[1]
        hidden_size = 64
        num_classes = len(data['Crop'].unique())

        model = CropRecommendationModel(input_size, hidden_size, num_classes).to("cuda")
        #model = torch.compile(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        xt = []
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            outputs = model(train_features)
            loss = criterion(outputs, train_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            xt.append(epoch)
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_outputs = model(val_features)
                val_loss = criterion(val_outputs, val_target)
                val_losses.append(val_loss.item())

        if fold == 0:
            xt = np.array(xt)
            train_losses = np.array(train_losses)
            val_losses = np.array(val_losses)
            plt.plot(xt, train_losses, label='Train Loss')
            plt.plot(xt, val_losses, label='Validation Loss')
            plt.xlabel("Epochs")
            plt.grid()
            plt.ylabel("Loss")
            plt.title("Epoch vs Loss Plot for Fold 1")
            plt.legend()
            plt.show()
            print(f'Epoch [{xt[-1]+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

        model.eval()
        with torch.no_grad():
            train_outputs = model(train_features)
            train_loss = criterion(train_outputs, train_target)
            _, train_predicted = torch.max(train_outputs.data, 1)
            train_correct = (train_predicted == train_target).sum().item()
            train_accuracy = train_correct / train_target.size(0)

            val_outputs = model(val_features)
            val_loss = criterion(val_outputs, val_target)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_correct = (val_predicted == val_target).sum().item()
            val_accuracy = val_correct / val_target.size(0)

            # Compute additional metrics
            train_precision = precision_score(train_target.cpu(), train_predicted.cpu(), average='weighted',zero_division=0)
            train_recall = recall_score(train_target.cpu(), train_predicted.cpu(), average='weighted',zero_division=0)
            #train_recall_ng = recall_score(train_target.cpu(), train_predicted.cpu(), average='weighted',zero_division=0,pos_label=0)
            train_f1 = f1_score(train_target.cpu(), train_predicted.cpu(), average='weighted',zero_division=0)

            val_precision = precision_score(val_target.cpu(), val_predicted.cpu(), average='weighted',zero_division=0)
            val_recall = recall_score(val_target.cpu(), val_predicted.cpu(), average='weighted',zero_division=0)
            #val_recall_ng = recall_score(val_target.cpu(), val_predicted.cpu(), average='weighted',zero_division=0,pos_label=0)
            val_f1 = f1_score(val_target.cpu(), val_predicted.cpu(), average='weighted',zero_division=0)
            
            conf_mat_train = confusion_matrix(train_target.cpu(), train_predicted.cpu())
            conf_mat_val = confusion_matrix(val_target.cpu(), val_predicted.cpu())
            tnr_per_class_train = []
            tnr_per_class_val = []
            for i in range(len(conf_mat_train)):
                tn = np.sum(np.delete(np.delete(conf_mat_train, i, axis=0), i, axis=1))
                fp = np.sum(np.delete(conf_mat_train[:, i], i))
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                tnr_per_class_train.append(tnr)
            for i in range(len(conf_mat_val)):
                tn = np.sum(np.delete(np.delete(conf_mat_val, i, axis=0), i, axis=1))
                fp = np.sum(np.delete(conf_mat_val[:, i], i))
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                tnr_per_class_val.append(tnr)
            
            train_tnr = np.mean(tnr_per_class_train)
            val_tnr = np.mean(tnr_per_class_val)
            
        fold_results.append({
            'fold': fold+1,
            'train_loss': train_loss.item(),
            'train_accuracy': train_accuracy,
            'val_loss': val_loss.item(),
            'val_accuracy': val_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_tnr': train_tnr,
            'train_f1': train_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_tnr': val_tnr,
            'val_f1': val_f1
        })
        print(f'Fold {fold+1}, Train Loss: {train_loss.item()}, Train Accuracy: {train_accuracy*100:.2f}%, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy*100:.2f}%')
        print(f'Fold {fold+1}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f},Train Negative Rate: {train_tnr:.4f} ,Train F1: {train_f1:.4f}')
        print(f'Fold {fold+1}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f},Validation Negative Rate: {val_tnr:.4f} ,Validation F1: {val_f1:.4f}')

    # Visualization of k-fold cross-validation results
    train_losses = [result['train_loss'] for result in fold_results]
    val_losses = [result['val_loss'] for result in fold_results]
    train_accuracies = [result['train_accuracy'] for result in fold_results]
    val_accuracies = [result['val_accuracy'] for result in fold_results]
    train_precisions = [result['train_precision'] for result in fold_results]
    val_precisions = [result['val_precision'] for result in fold_results]
    train_recalls = [result['train_recall'] for result in fold_results]
    val_recalls = [result['val_recall'] for result in fold_results]
    train_f1s = [result['train_f1'] for result in fold_results]
    val_f1s = [result['val_f1'] for result in fold_results]
    train_tnrs = [result['train_tnr'] for result in fold_results]
    val_tnrs = [result['val_tnr'] for result in fold_results]

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    avg_train_accuracy = np.mean(train_accuracies)
    avg_val_accuracy = np.mean(val_accuracies)
    avg_train_precision = np.mean(train_precisions)
    avg_val_precision = np.mean(val_precisions)
    avg_train_recall = np.mean(train_recalls)
    avg_val_recall = np.mean(val_recalls)
    avg_train_f1 = np.mean(train_f1s)
    avg_val_f1 = np.mean(val_f1s)
    avg_train_tnr = np.mean(train_tnrs)
    avg_val_tnr = np.mean(val_tnrs)
    
    avg_results = {
            'fold': "Average",
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': avg_val_accuracy,
            'train_precision': avg_train_precision,
            'train_recall': avg_train_recall,
            'train_tnr': avg_train_tnr,
            'train_f1': avg_train_f1,
            'val_precision': avg_val_precision,
            'val_recall': avg_val_recall,
            'val_tnr': avg_val_tnr,
            'val_f1': avg_val_f1
        }
    
    fold_results.append(avg_results)
    results_df = pd.DataFrame(fold_results)

    def smooth_curve(x, y):
        x_smooth = np.linspace(min(x), max(x), 300)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth

    plt.figure(figsize=(20, 16))

    # Plotting losses for each fold
    plt.subplot(6, 1, 1)
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), train_losses)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='b', label='Train Loss')
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), val_losses)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='r', label='Validation Loss')
    plt.xlabel("Fold")
    plt.ylabel("Loss")
    plt.title("Loss per Fold")
    plt.grid()
    plt.legend()

    # Plotting accuracies for each fold
    plt.subplot(6, 1, 2)
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), train_accuracies)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='b', label='Train Accuracy')
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), val_accuracies)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='r', label='Validation Accuracy')
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Fold")
    plt.grid()
    plt.legend()

    # Plotting precision for each fol6
    plt.subplot(6, 1, 3)
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), train_precisions)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='b', label='Train Precision')
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), val_precisions)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='r', label='Validation Precision')
    plt.xlabel("Fold")
    plt.ylabel("Precision")
    plt.title("Precision per Fold")
    plt.grid()
    plt.legend()

    # Plotting recall for each fold
    plt.subplot(6, 1, 4)
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), train_recalls)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='b', label='Train Recall')
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), val_recalls)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='r', label='Validation Recall')
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.title("Recall per Fold")
    plt.grid()
    plt.legend()
    
    plt.subplot(6, 1, 5)
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), train_tnrs)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='b', label='Train Negative Rate')
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), val_tnrs)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='r', label='Validation Negative Rate')
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.title("Recall per Fold")
    plt.grid()
    plt.legend()


    # Plotting F1 score for each fold
    plt.subplot(6, 1, 6)
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), train_f1s)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='b', label='Train F1')
    x_smooth, y_smooth = smooth_curve(range(1, k + 1), val_f1s)
    plt.plot(x_smooth, y_smooth, marker='', linestyle='-', color='r', label='Validation F1')
    plt.xlabel("Fold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Fold")
    plt.grid()
    plt.legend()


    plt.tight_layout()
    plt.savefig(f'../Results/{name}_Xvalidation_plot.png')
    plt.show()

    print(f'Average Train Loss: {avg_train_loss:.4f}')
    print(f'Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Average Train Accuracy: {avg_train_accuracy:.4f}')
    print(f'Average Validation Accuracy: {avg_val_accuracy:.4f}')
    print(f'Average Train Precision: {avg_train_precision:.4f}')
    print(f'Average Validation Precision: {avg_val_precision:.4f}')
    print(f'Average Train Recall: {avg_train_recall:.4f}')
    print(f'Average Validation Recall: {avg_val_recall:.4f}')
    print(f'Average Train Negative Rate: {avg_train_tnr:.4f}')
    print(f'Average Validation Negative Rate: {avg_val_tnr:.4f}')
    print(f'Average Train F1: {avg_train_f1:.4f}')
    print(f'Average Validation F1: {avg_val_f1:.4f}')
    
    return results_df


# In[15]:


from pyswarm import pso

# Store the history of PSO
pso_history = []

# Define the fitness function with logging for visualization
def fitness_function(params, train_features, test_features, train_target, test_target):
    global pso_history  # To store history
    
    # Extract the hyperparameters
    num_epochs = int(params[0])
    learning_rate = params[1]

    # Define the model
    input_size = train_features.shape[1]
    hidden_size = 64
    num_classes = len(train_target.unique())

    model = CropRecommendationModel(input_size, hidden_size, num_classes).to("cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training and Validation loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(train_features)
        loss = criterion(outputs, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_features)
        test_loss = criterion(test_outputs, test_target)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_correct = (test_predicted == test_target).sum().item()
        test_accuracy = test_correct / test_target.size(0)

    # Store the current state
    pso_history.append({
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'test_accuracy': test_accuracy
    })

    return -test_accuracy  # Since we want to maximize accuracy, minimize negative accuracy

# PSO wrapper function
def pso_optimize(train_features, test_features, train_target, test_target, lb, ub, num_particles=10, maxiter=20):
    global pso_history
    pso_history = []  # Reset history

    # Run PSO
    optimal_params, optimal_val = pso(
        fitness_function,
        lb,
        ub,
        args=(train_features, test_features, train_target, test_target),
        swarmsize=num_particles,
        maxiter=maxiter
    )

    optimal_num_epochs = int(optimal_params[0])
    optimal_learning_rate = optimal_params[1]

    print(f"Optimal num_epochs: {optimal_num_epochs}")
    print(f"Optimal learning_rate: {optimal_learning_rate}")
    print(f"Optimal validation accuracy: {-optimal_val}")

    # Visualization
    visualize_pso_history(pso_history)

    return optimal_num_epochs, optimal_learning_rate

# Function to visualize the PSO history
def visualize_pso_history(history):
    # Convert history to a DataFrame for easier plotting
    import pandas as pd
    history_df = pd.DataFrame(history)

    plt.figure(figsize=(14, 6))

    # Plot validation accuracy over iterations
    sns.lineplot(data=history_df, x=history_df.index, y="test_accuracy", marker="o")
    plt.title("PSO Optimization Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.show()

    # Plot the evolution of num_epochs and learning_rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.lineplot(data=history_df, x=history_df.index, y="num_epochs", marker="o", ax=axes[0])
    axes[0].set_title("Evolution of num_epochs")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("num_epochs")

    sns.lineplot(data=history_df, x=history_df.index, y="learning_rate", marker="o", ax=axes[1])
    axes[1].set_title("Evolution of learning_rate")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("learning_rate")

    plt.tight_layout()
    plt.show()

# Example usage
# Define the bounds for the hyperparameters
lb = [100, 0.0001]  # Lower bounds: [num_epochs, learning_rate]
ub = [2000, 0.01]  # Upper bounds: [num_epochs, learning_rate]


# In[16]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek,SMOTEENN


# In[18]:


features = data.drop(columns=['Crop']).values
target = data['Crop'].values


features = torch.tensor(features, dtype=torch.float32).float().to("cuda")
target = torch.tensor(target, dtype=torch.long).long().to("cuda")
name = f"{model_name}_Simple"
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
# optimal_num_epochs, optimal_learning_rate = pso_optimize(
#     train_features, 
#     test_features, 
#     train_target, 
#     test_target, 
#     lb, 
#     ub,
#     num_particles=10,
#     maxiter=5
# )


# In[ ]:


model,criterion = train_validation(train_features, test_features, train_target, test_target,name)
df1 = cross_check(features,target,name)


# In[ ]:


features = data.drop(columns=['Crop']).values
target = data['Crop'].values

print("Random OverSampler")
ROSample = RandomOverSampler(sampling_strategy="all")
features,target = ROSample.fit_resample(features,target)


features = torch.tensor(features, dtype=torch.float32).float().to("cuda")
target = torch.tensor(target, dtype=torch.long).long().to("cuda")
name = f"{model_name}_ROS"
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
model,criterion = train_validation(train_features, test_features, train_target, test_target,name)
df2 = cross_check(features,target,name)


# In[ ]:


features = data.drop(columns=['Crop']).values
target = data['Crop'].values

print("Random UnderSampler")
ROSample = RandomUnderSampler(sampling_strategy="all")
features,target = ROSample.fit_resample(features,target)


features = torch.tensor(features, dtype=torch.float32).float().to("cuda")
target = torch.tensor(target, dtype=torch.long).long().to("cuda")
name = f"{model_name}_RUS"
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
model,criterion = train_validation(train_features, test_features, train_target, test_target,name)
df3 = cross_check(features,target,name)


# In[ ]:


features = data.drop(columns=['Crop']).values
target = data['Crop'].values

print("SMOTE")

ROSample = SMOTE()
features,target = ROSample.fit_resample(features,target)


features = torch.tensor(features, dtype=torch.float32).float().to("cuda")
target = torch.tensor(target, dtype=torch.long).long().to("cuda")
name = f"{model_name}_SMOTE"
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
model,criterion = train_validation(train_features, test_features, train_target, test_target,name)
df4 = cross_check(features,target,name)


# In[ ]:


features = data.drop(columns=['Crop']).values
target = data['Crop'].values

print("SMOTE + Tomek")

ROSample = SMOTETomek()
features,target = ROSample.fit_resample(features,target)



features = torch.tensor(features, dtype=torch.float32).float().to("cuda")
target = torch.tensor(target, dtype=torch.long).long().to("cuda")
name = f"{model_name}_SMOTETOMEK"
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
model,criterion = train_validation(train_features, test_features, train_target, test_target,name)
df5 = cross_check(features,target,name)


# In[ ]:


features = data.drop(columns=['Crop']).values
target = data['Crop'].values

print("SMOTEENN")

ROSample = SMOTEENN()
features,target = ROSample.fit_resample(features,target)



features = torch.tensor(features, dtype=torch.float32).float().to("cuda")
target = torch.tensor(target, dtype=torch.long).long().to("cuda")
name = f"{model_name}_SMOTEENN"
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)
model,criterion = train_validation(train_features, test_features, train_target, test_target,name)
df6 = cross_check(features,target,name)


# In[ ]:


df1.to_csv(f"../Results/{model_name}_Initial.csv",index=False)
df2.to_csv(f"../Results/{model_name}_ROS.csv",index=False)
df3.to_csv(f"../Results/{model_name}_RUS.csv",index=False)
df4.to_csv(f"../Results/{model_name}_SMOTE.csv",index=False)
df5.to_csv(f"../Results/{model_name}_SMOTE+TOMEK.csv",index=False)
df6.to_csv(f"../Results/{model_name}_SMOTEENN.csv",index=False)

