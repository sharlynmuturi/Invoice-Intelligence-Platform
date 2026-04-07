import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data"
DATA_PATH.mkdir(exist_ok=True)

from sqlalchemy import create_engine
engine = create_engine(f"sqlite:///{BASE_DIR / 'inventorydb.db'}")

fake = Faker()
np.random.seed(42)

N_PRODUCTS = 1000
N_PURCHASES = 50000
N_VENDORS = 100

# Vendors
vendors = pd.DataFrame({
    "VendorNo": np.arange(1000, 1000 + N_VENDORS),
    "VendorName": [fake.company() for _ in range(N_VENDORS)],
    "vendor_city": [fake.city() for _ in range(N_VENDORS)],
    "vendor_distance": np.random.uniform(50, 1000, N_VENDORS),  # in km
    "vendor_rate": np.random.uniform(0.8, 1.2, N_VENDORS)
})

# Products (purchase_prices)
brands = [fake.company() for _ in range(200)]

products = []
for i in range(N_PRODUCTS):
    vendor = vendors.sample(1).iloc[0]
    
    products.append({
        "brand": random.choice(brands),
        "description": fake.word().capitalize() + " Product",
        "price": round(random.uniform(5, 100), 2),
        "size": random.choice(["Small", "Medium", "Large"]),
        "volume": random.choice([250, 500, 750, 1000]),
        "classification": random.choice(["A", "B", "C"]),
        "purchaseprice": round(random.uniform(3, 80), 2),
        "vendornumber": vendor["VendorNo"],
        "vendorname": vendor["VendorName"]
    })

purchase_prices = pd.DataFrame(products)

# Purchases
purchases = []

products_records = purchase_prices.to_dict("records")

for i in range(N_PURCHASES):
    product = random.choice(products_records)
    
    po_date = fake.date_between(start_date='-2y', end_date='-1y')
    receiving_date = po_date + timedelta(days=random.randint(1, 10))
    invoice_date = receiving_date + timedelta(days=random.randint(1, 5))
    pay_date = invoice_date + timedelta(days=random.randint(5, 30))
    
    qty = random.randint(1, 100)
    price = product["purchaseprice"]
    
    purchases.append({
        "inventoryid": i,
        "Store": random.randint(1, 50),
        "Brand": product["brand"],
        "Description": product["description"],
        "Size": product["size"],
        "VendorNo": product["vendornumber"],
        "VendorName": product["vendorname"],
        "PONumber": random.randint(10000, 99999),
        "PODate": po_date,
        "ReceivingDate": receiving_date,
        "InvoiceDate": invoice_date,
        "PayDate": pay_date,
        "PurchasePrice": price,
        "Quantity": qty,
        "Dollars": round(qty * price, 2),
        "Classification": product["classification"]
    })

purchases_df = pd.DataFrame(purchases)


# Vendor Invoice
vendor_invoice = (
    purchases_df
    .groupby(["VendorNo", "VendorName", "PONumber"], as_index=False)
    .agg({
        "InvoiceDate": "max",
        "PODate": "min",
        "PayDate": "max",
        "Quantity": "sum",
        "Dollars": "sum"
    })
)

# Merging distance
vendor_invoice = vendor_invoice.merge(
    vendors[["VendorNo", "vendor_distance", "vendor_rate"]],
    on="VendorNo",
    how="left"
)

# Freight logic
base_cost = np.random.uniform(10, 30, len(vendor_invoice))  # fixed handling cost
distance_cost = vendor_invoice["vendor_distance"] * np.random.uniform(0.02, 0.06, len(vendor_invoice)) # transport cost
weight_cost = vendor_invoice["Quantity"] * np.random.uniform(1.0, 2.5, len(vendor_invoice)) # shipment size (to increase with quantity)
scale_discount = np.log1p(vendor_invoice["Quantity"]) * 0.5 # Economies of scale (large orders are cheaper per unit)
adjusted_weight_cost = weight_cost / scale_discount # Applying discount only to weight cost (reduces per-unit cost)
noise = np.random.normal(0, 5, len(vendor_invoice)) # introducing some randomness

vendor_invoice["Freight"] = (base_cost + distance_cost + adjusted_weight_cost * vendor_invoice["vendor_rate"] + noise).round(2)

vendor_invoice["Approval"] = np.where(vendor_invoice["Dollars"] > 5000, "Pending", "Approved")


vendor_invoice.rename(columns={
    "VendorNo": "vendornumber",
    "VendorName": "vendorname",
    "PONumber": "ponumber",
    "InvoiceDate": "invoicedate",
    "PODate": "podate",
    "PayDate": "paydate",
    "Quantity": "quantity",
    "Dollars": "dollars",
    "Freight": "freight"
}, inplace=True)

# Inventory
def generate_inventory(tag):
    inventory = []
    
    for i in range(10000):
        product = purchase_prices.sample(1).iloc[0]
        
        inventory.append({
            "inventoryid": i,
            "store": random.randint(1, 50),
            "city": fake.city(),
            "brand": product["brand"],
            "description": product["description"],
            "size": product["size"],
            "onhand": max(0, random.randint(50, 300) - random.randint(0, 200)),
            "price": product["price"],
            "startdate": fake.date_between(start_date='-2y', end_date='-1y')
        })
    
    return pd.DataFrame(inventory)

begin_inventory = generate_inventory("begin")
end_inventory = generate_inventory("end")

purchase_prices.to_sql("purchase_prices", engine, if_exists="replace", index=False)
purchases_df.to_sql("purchases", engine, if_exists="replace", index=False, chunksize=5000)
vendor_invoice.to_sql("vendor_invoice", engine, if_exists="replace", index=False)
begin_inventory.to_sql("begin_inventory", engine, if_exists="replace", index=False)
end_inventory.to_sql("end_inventory", engine, if_exists="replace", index=False)

print("Data saved to SQLite database!")

purchase_prices.to_csv(DATA_PATH / "purchase_prices.csv", index=False)
purchases_df.to_csv(DATA_PATH / "purchases.csv", index=False)
vendor_invoice.to_csv(DATA_PATH / "vendor_invoice.csv", index=False)
begin_inventory.to_csv(DATA_PATH / "begin_inventory.csv", index=False)
end_inventory.to_csv(DATA_PATH / "end_inventory.csv", index=False)

print("All datasets generated successfully!")
