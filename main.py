import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from zipfile import ZipFile
from io import BytesIO
import requests

def tr_title(title):
    return title

plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

os.makedirs('figures', exist_ok=True)

def download_dataset():
    try:
        print("Attempting to download the dataset from UCI repository...")
        url = "https://archive.ics.uci.edu/static/public/1150/gallstone-1.zip"
        response = requests.get(url)
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extract('dataset-uci.xlsx')
        print("Dataset downloaded and extracted successfully.")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_dataset():
    try:
        print("Trying to load local dataset...")
        data = pd.read_excel('dataset-uci.xlsx', engine='openpyxl')
        print("Local dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading local dataset: {e}")
        
        if download_dataset():
            try:
                data = pd.read_excel('dataset-uci.xlsx', engine='openpyxl')
                print("Downloaded dataset loaded successfully.")
                return data
            except Exception as e:
                print(f"Error loading downloaded dataset: {e}")
        
        print("Creating sample dataset for testing...")
        sample_data = pd.DataFrame({
            'Gallstone Status': [0, 1, 0, 1, 0],
            'Age': [45, 65, 32, 58, 41],
            'Gender': [0, 1, 0, 1, 1],
            'Weight': [75.2, 82.5, 68.3, 90.1, 63.7],
            'Height': [175, 165, 180, 168, 162]
        })
        return sample_data

data = load_dataset()

def show_basic_info(df):
    print("\n" + "="*50)
    print("DATASET BASIC INFORMATION")
    print("="*50)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFeature Summary:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values Check:")
    missing = df.isnull().sum()
    if missing.any() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found.")
    
    print("\nFirst 5 rows of data:")
    print(df.head())

def show_statistical_summary(df):
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    
    stats = df.describe().T
    stats['Range'] = stats['max'] - stats['min']
    stats['Coef of Variation'] = (stats['std'] / stats['mean']) * 100
    
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(stats)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print("\nCategorical Variable Distributions:")
        for col in categorical_cols:
            print(f"\n{col}:")
            print(df[col].value_counts())

def plot_target_distribution(df):
    if 'Gallstone Status' in df.columns:
        plt.figure(figsize=(10, 6))
        counts = df['Gallstone Status'].value_counts()
        sns.countplot(x=df['Gallstone Status'], palette='viridis')
        
        for i, count in enumerate(counts):
            plt.text(i, count + 5, f"{count}", ha='center', fontweight='bold')
        
        plt.title(tr_title('Safra Taşı Durumu Dağılımı'), fontsize=16)
        plt.xlabel(tr_title('Safra Taşı Durumu (0: Yok, 1: Var)'), fontsize=14)
        plt.ylabel(tr_title('Hasta Sayısı'), fontsize=14)
        plt.savefig('figures/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nTarget distribution plot saved to 'figures/target_distribution.png'")

def plot_numeric_distributions(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if 'Gallstone Status' in numeric_cols:
        numeric_cols.remove('Gallstone Status')
    
    if not numeric_cols:
        print("No numeric columns to plot")
        return
    
    num_plots = min(len(numeric_cols), 9)
    cols_to_plot = numeric_cols[:num_plots]
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        if i < len(axes):
            sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/numeric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nNumeric distributions plot saved to 'figures/numeric_distributions.png'")

def plot_boxplots_by_target(df):
    if 'Gallstone Status' not in df.columns:
        print("Target column 'Gallstone Status' not found, skipping boxplots")
        return
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Gallstone Status' in numeric_cols:
        numeric_cols.remove('Gallstone Status')
    
    if not numeric_cols:
        print("No numeric columns to plot")
        return
    
    cols_to_plot = numeric_cols[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        if i < len(axes):
            sns.boxplot(x='Gallstone Status', y=col, data=df, ax=axes[i], palette='viridis')
            axes[i].set_title(f'{col} by Gallstone Status', fontsize=14)
            axes[i].set_xlabel('Gallstone Status (0: Absent, 1: Present)', fontsize=12)
            
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/boxplots_by_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nBoxplots by target saved to 'figures/boxplots_by_target.png'")

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation analysis")
        return
    
    plt.figure(figsize=(14, 12))
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .5})
    
    plt.title(tr_title('Sayısal Değişkenler Arasındaki Korelasyon Matrisi'), fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nCorrelation heatmap saved to 'figures/correlation_heatmap.png'")

def plot_gender_analysis(df):
    if 'Gender' not in df.columns or 'Gallstone Status' not in df.columns:
        print("Gender or Gallstone Status columns not found, skipping gender analysis")
        return
    
    plt.figure(figsize=(12, 7))
    
    gender_gallstone = pd.crosstab(df['Gender'], df['Gallstone Status'])
    gender_gallstone_pct = gender_gallstone.div(gender_gallstone.sum(axis=1), axis=0) * 100
    
    sns.heatmap(gender_gallstone_pct, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title(tr_title('Cinsiyete Göre Safra Taşı Durumu (%)'), fontsize=16)
    plt.xlabel(tr_title('Safra Taşı Durumu (0: Yok, 1: Var)'), fontsize=14)
    plt.ylabel(tr_title('Cinsiyet (0: Erkek, 1: Kadın)'), fontsize=14)
    
    plt.savefig('figures/gender_gallstone_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGender analysis plot saved to 'figures/gender_gallstone_heatmap.png'")

def plot_age_distribution(df):
    if 'Age' not in df.columns or 'Gallstone Status' not in df.columns:
        print("Age or Gallstone Status columns not found, skipping age distribution analysis")
        return
    
    plt.figure(figsize=(14, 8))
    
    sns.histplot(data=df, x='Age', hue='Gallstone Status', multiple='stack', 
                 palette='viridis', kde=True, element='step')
    
    plt.title(tr_title('Safra Taşı Durumuna Göre Yaş Dağılımı'), fontsize=16)
    plt.xlabel(tr_title('Yaş'), fontsize=14)
    plt.ylabel(tr_title('Hasta Sayısı'), fontsize=14)
    plt.legend(title=tr_title('Safra Taşı'), labels=['Yok', 'Var'])
    
    plt.savefig('figures/age_distribution_by_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nAge distribution plot saved to 'figures/age_distribution_by_target.png'")

def plot_bmi_analysis(df):
    if 'BMI' not in df.columns or 'Gallstone Status' not in df.columns:
        print("BMI or Gallstone Status columns not found, skipping BMI analysis")
        return
    
    plt.figure(figsize=(12, 7))
    
    sns.boxplot(x='Gallstone Status', y='BMI', data=df, palette='viridis')
    sns.stripplot(x='Gallstone Status', y='BMI', data=df, color='black', alpha=0.3, jitter=True)
    
    plt.title(tr_title('Safra Taşı Durumuna Göre BMI Dağılımı'), fontsize=16)
    plt.xlabel(tr_title('Safra Taşı Durumu (0: Yok, 1: Var)'), fontsize=14)
    plt.ylabel(tr_title('BMI (Vücut Kitle İndeksi)'), fontsize=14)
    
    plt.savefig('figures/bmi_by_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nBMI analysis plot saved to 'figures/bmi_by_target.png'")

def plot_comorbidity_analysis(df):
    comorbidity_cols = ['Comorbidity', 'Coronary Artery Disease (CAD)', 
                         'Hypothyroidism', 'Hyperlipidemia', 'Diabetes Mellitus (DM)']
    
    existing_cols = [col for col in comorbidity_cols if col in df.columns]
    
    if not existing_cols or 'Gallstone Status' not in df.columns:
        print("Comorbidity or Gallstone Status columns not found, skipping comorbidity analysis")
        return
    
    fig, axes = plt.subplots(len(existing_cols), 1, figsize=(12, 4 * len(existing_cols)))
    
    if len(existing_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(existing_cols):
        crosstab = pd.crosstab(df[col], df['Gallstone Status'], normalize='index') * 100
        crosstab.plot(kind='bar', stacked=True, ax=axes[i], colormap='viridis')
        
        axes[i].set_title(f'{col} by Gallstone Status (%)', fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Percentage (%)', fontsize=12)
        axes[i].legend(title='Gallstone Status', labels=['Absent', 'Present'])
        
    plt.tight_layout()
    plt.savefig('figures/comorbidity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComorbidity analysis plot saved to 'figures/comorbidity_analysis.png'")

def plot_pairplot(df):
    if 'Gallstone Status' not in df.columns:
        print("Target column 'Gallstone Status' not found, skipping pairplot")
        return
    
    potential_cols = ['Age', 'BMI', 'Weight', 'Height']
    existing_cols = [col for col in potential_cols if col in df.columns]
    
    if len(existing_cols) < 2:
        print("Not enough numeric columns for pairplot")
        return
    
    cols_to_plot = existing_cols[:min(4, len(existing_cols))] + ['Gallstone Status']
    
    plt.figure(figsize=(16, 14))
    g = sns.pairplot(df[cols_to_plot], hue='Gallstone Status', 
                   diag_kind='kde', plot_kws={'alpha': 0.6}, 
                   height=2.5, aspect=1.2, palette='viridis')
    
    g.fig.suptitle(tr_title('Önemli Değişkenlerin Çiftli Dağılımları'), fontsize=16, y=1.02)
    plt.savefig('figures/feature_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPairplot saved to 'figures/feature_pairplot.png'")

def plot_feature_importance(df):
    if 'Gallstone Status' not in df.columns:
        print("Target column 'Gallstone Status' not found, skipping feature importance")
        return
    
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Gallstone Status' in numeric_cols:
            numeric_cols.remove('Gallstone Status')
        
        if not numeric_cols:
            print("No numeric columns for feature importance")
            return
        
        correlations = df[numeric_cols].corrwith(df['Gallstone Status']).abs().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        correlations[:15].plot(kind='barh', color=plt.cm.viridis(np.linspace(0, 0.8, len(correlations[:15]))))
        plt.title(tr_title('Safra Taşı ile Değişkenler Arasındaki Korelasyon (Mutlak Değer)'), fontsize=16)
        plt.xlabel(tr_title('Korelasyon Katsayısı (Mutlak Değer)'), fontsize=14)
        plt.ylabel(tr_title('Değişkenler'), fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFeature importance plot saved to 'figures/feature_importance.png'")
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")

show_basic_info(data)
show_statistical_summary(data)
plot_target_distribution(data)
plot_numeric_distributions(data)
plot_boxplots_by_target(data)
plot_correlation_heatmap(data)
plot_gender_analysis(data)
plot_age_distribution(data)
plot_bmi_analysis(data)
plot_comorbidity_analysis(data)
plot_pairplot(data)
plot_feature_importance(data)

print("\n" + "="*50)
print("EDA COMPLETED SUCCESSFULLY")
print("="*50)
print("\nAll analysis results and visualizations have been saved to the 'figures' folder.")
print("You can include these in your report for the Data Science project.")