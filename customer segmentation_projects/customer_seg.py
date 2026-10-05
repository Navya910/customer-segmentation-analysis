import os
import zipfile
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

zip_file = "Online Retail.zip"
extract_folder = "data"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_folder)

print("Dataset extracted successfully.")

data_file = None

for root, dirs, files in os.walk(extract_folder):
    for file in files:
        if file.lower().endswith((".csv", ".xlsx", ".xls")):
            data_file = os.path.join(root, file)
            break

    if data_file:
        break


if data_file is None:
    raise FileNotFoundError(
        "CSV or Excel file not found inside Online Retail.zip"
    )

print("Dataset found:", data_file)

if data_file.lower().endswith(".csv"):

    df = pd.read_csv(
        data_file,
        usecols=[0, 3, 5, 6]
    )

else:

    df = pd.read_excel(
        data_file,
        usecols=[0, 3, 5, 6]
    )

df.columns = [
    "InvoiceNo",
    "Quantity",
    "UnitPrice",
    "CustomerID"
]


df["InvoiceNo"] = df["InvoiceNo"].astype(str)

print("\nOriginal shape:", df.shape)

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

features = [
    "PurchaseFrequency",
    "TotalQuantity",
    "TotalSpend"
]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    customer_data[features]
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

customer_data["Segment"] = kmeans.fit_predict(
    scaled_features
)

print("\nCustomer Segments:")
print(customer_data.head())

print("\nSegment counts:")
print(customer_data["Segment"].value_counts())


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