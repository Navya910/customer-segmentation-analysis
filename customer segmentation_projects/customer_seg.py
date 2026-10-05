import pandas as pd

df = pd.read_csv(
    "Online Retail.csv",
    usecols=[0, 3, 5, 6]
)

df.columns = ["InvoiceNo", "Quantity", "UnitPrice", "CustomerID"]

df["InvoiceNo"] = df["InvoiceNo"].astype(str)

print("Original shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna(subset=["CustomerID"])

df = df[~df["InvoiceNo"].str.startswith("C")]

df = df[df["Quantity"] > 0]

df = df[df["UnitPrice"] > 0]

df = df.drop_duplicates()

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

customer_data = df.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",
    "Quantity": "sum",
    "TotalAmount": "sum"
}).reset_index()

customer_data.columns = [
    "CustomerID",
    "PurchaseFrequency",
    "TotalQuantity",
    "TotalSpend"
]

print("\nCustomer data:")
print(customer_data.head())

print("\nCustomer data shape:")
print(customer_data.shape)

print("\nMissing values in customer data:")
print(customer_data.isnull().sum())

from sklearn.preprocessing import StandardScaler


features = customer_data[
    ["PurchaseFrequency", "TotalQuantity", "TotalSpend"]
]


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("\nScaled data:")
print(scaled_features[:5])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

customer_data["Segment"] = kmeans.fit_predict(scaled_features)

print("\nCustomer Segments:")
print(customer_data.head())

print("\nSegment counts:")
print(customer_data["Segment"].value_counts())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    customer_data["PurchaseFrequency"],
    customer_data["TotalSpend"],
    c=customer_data["Segment"]
)

plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spend")
plt.title("Customer Segmentation")
plt.show()

segment_summary = customer_data.groupby("Segment").agg({
    "PurchaseFrequency": "mean",
    "TotalQuantity": "mean",
    "TotalSpend": "mean"
}).round(2)

print("\nSegment Summary:")
print(segment_summary)