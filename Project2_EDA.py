import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Dataset.csv")

# Display first 5 rows
print("FIRST 5 ROWS")
print(df.head())

# Dataset Information
print("\nDATASET INFORMATION")
print(df.info())

# Shape
print("\nDATASET SHAPE")
print(df.shape)

# Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Statistical Summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Mean
print("\nMEAN VALUES")
print(df.select_dtypes(include=np.number).mean())

# Median
print("\nMEDIAN VALUES")
print(df.select_dtypes(include=np.number).median())

# Count
print("\nCOUNT VALUES")
print(df.count())

# Total Revenue
print("\nTOTAL REVENUE")
print(df["TotalPrice"].sum())

# Product Analysis
print("\nPRODUCT COUNTS")
print(df["Product"].value_counts())

plt.figure(figsize=(10,5))
df["Product"].value_counts().plot(kind="bar")
plt.title("Products Sold")
plt.xlabel("Product")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Order Status Analysis
plt.figure(figsize=(6,6))
df["OrderStatus"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Order Status Distribution")
plt.ylabel("")
plt.show()

# Payment Method Analysis
plt.figure(figsize=(8,5))
sns.countplot(
    x="PaymentMethod",
    data=df
)
plt.title("Payment Method Analysis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Referral Source Analysis
plt.figure(figsize=(8,5))
sns.countplot(
    x="ReferralSource",
    data=df
)
plt.title("Referral Source Analysis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Quantity Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Quantity"], bins=20, kde=True)
plt.title("Quantity Distribution")
plt.show()

# Total Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["TotalPrice"], bins=20, kde=True)
plt.title("Total Price Distribution")
plt.show()

# Outlier Detection
plt.figure(figsize=(8,5))
sns.boxplot(x=df["TotalPrice"])
plt.title("Outlier Detection in TotalPrice")
plt.show()

# Monthly Sales Trend
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

monthly_sales = df.groupby("Month")["TotalPrice"].sum()

plt.figure(figsize=(8,5))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# Top 5 Products by Revenue
top_products = df.groupby("Product")["TotalPrice"].sum().sort_values(ascending=False).head(5)

print("\nTOP 5 PRODUCTS BY REVENUE")
print(top_products)

plt.figure(figsize=(8,5))
top_products.plot(kind="bar")
plt.title("Top 5 Products by Revenue")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

# Key Observations
print("\nKEY OBSERVATIONS")
print("1. Dataset contains customer order information.")
print("2. Product sales distribution analyzed.")
print("3. Payment methods analyzed.")
print("4. Revenue trends identified.")
print("5. Outliers detected using boxplot.")
print("6. Monthly sales trend visualized.")
print("7. Correlation between numerical columns analyzed.")
print("8. Top revenue-generating products identified.")

print("\nEDA COMPLETED SUCCESSFULLY")