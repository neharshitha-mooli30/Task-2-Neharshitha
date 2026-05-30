import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("netflix_titles.csv")

# Basic Information
print("Dataset Shape:", df.shape)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistics
print("\nStatistics:")
print(df.describe())

# Movies vs TV Shows
print("\nContent Type Count:")
print(df['type'].value_counts())

# Visualization 1
df['type'].value_counts().plot(
    kind='bar',
    title='Movies vs TV Shows'
)
plt.show()

# Visualization 2
df['rating'].value_counts().head(10).plot(
    kind='bar',
    title='Top Ratings'
)
plt.show()

# Visualization 3
df['country'].value_counts().head(10).plot(
    kind='bar',
    title='Top 10 Countries'
)
plt.show()