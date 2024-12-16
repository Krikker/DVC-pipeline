import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
iris_data = pd.read_csv('data/raw/Iris.csv')

# Pairplot
sns.pairplot(iris_data, hue='Species')
plt.savefig('reports/pairplot.png')
plt.close()

# Correlation heatmap
correlation_matrix = iris_data.drop('Species', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig('reports/heatmap.png')
plt.close()

# Histograms
iris_data.drop('Species', axis=1).hist(bins=20, figsize=(10, 8))
plt.suptitle("Feature Distributions")
plt.savefig('reports/histograms.png')
