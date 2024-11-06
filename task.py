import pandas as pd
import numpy as np  
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#load the data & show the data dimensions (1314,6)
data_path = Path("D:/cam/D100/PS2_EX2/PS2_EX2/ps_2_ex2_smoking_is_bad/data") / "smoking_data.csv"
df = pd.read_csv(data_path)
print(df.shape)

#string columns are not be included in the describe() function
print(df.describe())

#check for null values in the data
print(df.isnull().sum())

#prepare the data frame and check my cousin's result
df['smokes'] = df['smoker'].map({'Yes': 1, 'No': 0})
df['alive'] = df['outcome'].map({'Alive': 1, 'Dead': 0})
result =df.groupby(["smokes"]).agg(prob=("alive", "mean"))
print(result)

#visualize the data, check whether age is a factor in survival rate
#it shows that the alive rate is much lower for people who smoke within the age group of 40-50, 50-60
age_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
age_labels = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90'] 
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
survival_rate1 = df.groupby(['age_group', 'smokes'], observed=False)['alive'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='age_group', y='alive', hue='smokes', data=survival_rate1)
plt.title('Survival Rate by Age Group and Smoking Status')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Now check whether gender is a factor
#compare the survival rate between smoking males and non-smoking males
#it is clear that for males, smoking has bad effects on survival rate within almost all age groups
df_male = df[df['gender'] == 'male']
survival_rate_male = df_male.groupby(['age_group', 'smokes'], observed=False)['alive'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='age_group', y='alive', hue='smokes', data=survival_rate_male)
plt.title('Survival Rate by Smoking Status (male)')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#check the situation for females
#it shows that for females, smoking has bad effects on survival rate within the age group of 40-50, 50-60
df_female = df[df['gender'] == 'female']
survival_rate_female = df_female.groupby(['age_group', 'smokes'], observed=False)['alive'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='age_group', y='alive', hue='smokes', data=survival_rate_female)
plt.title('Survival Rate by Smoking Status (female)')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

'''
Conclusion:
- Generally, the survival rate is much lower for people who smoke within the age group of 40-50, 50-60.
- For males, smoking has bad effects on survival rate within almost all age groups.
- For females, smoking has bad effects on survival rate within the age group of 40-50, 50-60.
Therefore, my cousin's result is correct. When analyzing the effects of smoking on survival rate, we should consider the age group and gender of the individuals.
'''