import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

df = pd.read_csv(r"D:\ARCAP\Tasks\zomato.csv",encoding='latin1')

"""Making a copy of the original dataset"""

zomato_data = df.copy()
print(zomato_data.columns)

"""Handling missing values"""

print("Missing values before handling:\n", zomato_data.isnull().sum())

"""replacing the missing values by 'unknown'"""

zomato_data=zomato_data.fillna('unknown')

"""Remove duplicates"""

initial_rows = zomato_data.shape[0]
zomato_data = zomato_data.drop_duplicates()
final_rows = zomato_data.shape[0]
print(f"\nDuplicates removed: {initial_rows - final_rows} (Initial: {initial_rows}, Final: {final_rows})")

"""Standardize text columns"""

text_cols = ['Restaurant Name', 'City','Locality','Address', 'Cuisines']
for col in text_cols:
  zomato_data[col]=zomato_data[col].astype(str).str.title().str.strip()
  if col=='Cuisines':
    zomato_data[col]=zomato_data[col].str.lower().str.strip()
print("\nSample standardized text (first 5 rows):\n", zomato_data[text_cols].head(2))

"""Correcting the data types"""

numeric_cols_to_convert = ['Votes', 'Average Cost for Two', 'Price range']  
for col in numeric_cols_to_convert:
    if col in zomato_data.columns:
        zomato_data[col] = pd.to_numeric(zomato_data[col], errors='coerce')
        if zomato_data[col].notna().all() and (zomato_data[col] % 1 == 0).all():
            zomato_data[col] = zomato_data[col].astype('int64')
        else:
            zomato_data[col] = zomato_data[col].astype('float64')
print("\nUpdated data types:\n", zomato_data.dtypes)

"""Handling outliers using IQR method"""

numeric_cols_for_outliers = zomato_data.select_dtypes(include=['int64', 'float64']).columns
Q1 = zomato_data[numeric_cols_for_outliers].quantile(0.25)
Q3 = zomato_data[numeric_cols_for_outliers].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
for col in numeric_cols_for_outliers:
    zomato_data[col] = np.where(zomato_data[col] > upper_bound[col], upper_bound[col],
                               np.where(zomato_data[col] < lower_bound[col], lower_bound[col],
                                        zomato_data[col]))
print("\nOutliers handled (first 5 rows of numeric columns):\n", zomato_data[numeric_cols_for_outliers].head())

df = df[(df['Latitude'].between(-90, 90)) & (df['Longitude'].between(-180, 180))]

"""Exploraatory data analysis (EDA)
focusing on:
1. Top locations, cuisines and restaurant types.
2. How do ratings correlate with cost, online ordering or votes?
3. Treands in meal types or booking preferences.
"""

#style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

#getting values for the basic stats
print("\nRating Summary\n")
print(zomato_data['Aggregate rating'].describe().to_string(index=True,name=False))

print ("\nCost summary\n")
print(zomato_data['Average Cost for two'].describe().to_string(index=True,name=False))

top_cities=zomato_data['City'].value_counts().head(10)
top_cities.index.name=None
print("\nTop 10 Cities\n")
print(top_cities.to_string(index=True,name=False))

top_cuisines=zomato_data['Cuisines'].str.split(',',expand=True).stack().str.strip()
cuisine_counts=top_cuisines.value_counts().head(10)
print("\nTop 10 Cuisines:\n")
print(cuisine_counts.to_string(index=True,name=False))

print("\nAverage Rating by Online Delivery:")
print(df.groupby('Has Online delivery')['Aggregate rating'].mean().to_string(index=True,name=False))
print("\nAverage Rating by Table Booking:")
print(df.groupby('Has Table booking')['Aggregate rating'].mean().to_string(index=True,name=False))

#Histogram: Rating Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Aggregate rating'], bins=20, kde=True)
plt.title('Distribution of Restaurant Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.savefig('rating_hist.png')
plt.show()

#Bar Chart: Top Localities
plt.figure(figsize=(12, 6))
top_cities.plot(kind='bar')
plt.title('Top 10 Localities by Restaurant Count')
plt.xticks(rotation=45)
plt.ylabel('Number of Restaurants')
plt.tight_layout()
plt.savefig('top_localities_bar.png')
plt.show()

#Scatter Plot: Cost vs. Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Average Cost for two', y='Aggregate rating', hue='Price range', size='Votes', alpha=0.6)
plt.title('Cost vs. Rating (Colored by Price Range, Sized by Votes)')
plt.xlabel('Average Cost for Two (INR)')
plt.ylabel('Aggregate Rating')
plt.savefig('cost_vs_rating_scatter.png')
plt.show()

#High-rated restaurants may cluster in urban centers
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Aggregate rating', palette='coolwarm', size='Votes')
plt.title('Geographic Distribution of Restaurants by Rating')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('Geo_distribution_rating.png')
plt.show()

#Online delivery restaurants tend to have slightly higher ratings
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Has Online delivery', y='Aggregate rating')
plt.title('Rating Distribution by Online Delivery')
plt.savefig('Rating_distribution_on_delivery.png')
plt.show()

"""Downloading the graphs"""

print("Visualizations saved as 'ratings_hist.png', 'top_localities_bar.png', and 'cost_vs_rating_scatter.png'")