import pandas as pd

# Load only the first 5000 rows of your large CSV file
df = pd.read_csv("creditcard.csv", nrows=5000)

# Save it as a smaller CSV file
df.to_csv("creditcard_sample.csv", index=False)

print("✅ Small sample CSV created successfully: creditcard_sample.csv")
