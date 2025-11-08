import pandas as pd

# Load your full credit card dataset
df = pd.read_csv("creditcard.csv")   # ✅ correct file name

# ✅ Create a fraud-only dataset
fraud_df = df[df["Class"] == 1]
fraud_df.to_csv("sample_test_fraud.csv", index=False)
print("✅ sample_test_fraud.csv created with", fraud_df.shape[0], "fraud rows")

# ✅ Create a balanced dataset (mix of fraud + normal)
fraud_sample = fraud_df.sample(min(100, len(fraud_df)), random_state=42)   # up to 100 fraud
normal_sample = df[df["Class"] == 0].sample(400, random_state=42)          # 400 normal

mixed_df = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=42)  # shuffle
mixed_df.to_csv("mixed_test.csv", index=False)
print("✅ mixed_test.csv created with", fraud_sample.shape[0], "fraud +", normal_sample.shape[0], "normal")
