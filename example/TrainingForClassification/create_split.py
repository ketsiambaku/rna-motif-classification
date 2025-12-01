import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('rna_labels.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df.to_csv('rna_train.csv', index=False)
val_df.to_csv('rna_validation_labels.csv', index=False)

print(f"Training set: {len(train_df)} samples")
print(train_df['label'].value_counts())
print(f"\nValidation set: {len(val_df)} samples")
print(val_df['label'].value_counts())
