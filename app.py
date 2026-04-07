import streamlit as st
import pandas as pd
import numpy as np
import joblib


freight_model = joblib.load("artifacts/freight_predictor.pkl")
anomaly_model = joblib.load("artifacts/anomaly_detector.pkl")
scaler = joblib.load("artifacts/anomaly_scaler.pkl")
iso_model = joblib.load("artifacts/iso_forest.pkl")  

st.set_page_config(page_title="Invoice Operations Dashboard", layout="wide")
st.title("Invoice Operations Dashboard")

st.markdown("""
This dashboard allows teams to:
- Predict freight costs for new POs.
- Detect suspicious invoices (supervised & unsupervised).
- Optimize operations by identifying potential overbilling or late invoices.
""")

def prepare_freight_features(quantity, vendor_distance, avg_vendor_qty=50):
    
    log_quantity = np.log1p(quantity)
    relative_order_size = quantity / avg_vendor_qty
    qty_distance_interaction = quantity * vendor_distance
    
    return pd.DataFrame([{
        "qty_distance_interaction": qty_distance_interaction,
        "relative_order_size": relative_order_size,
        "vendor_distance": vendor_distance,
        "quantity": quantity,
        "log_quantity": log_quantity
    }])

st.sidebar.header("Freight Cost Prediction")

invoice_quantity = st.sidebar.number_input("Invoice Quantity", min_value=1, value=50,help="Total number of items listed on the invoice.")
vendor_distance = st.sidebar.number_input("Vendor Distance (km)", min_value=1.0, value=250.0, help="Distance from the vendor to your location in kilometers.")
# estimate vendor average (loading from DB later)
avg_vendor_qty = st.sidebar.number_input("Avg Vendor Quantity", value=50, help="Average number of items this vendor typically ships. Used to calculate relative order size.")

X_freight = prepare_freight_features(invoice_quantity, vendor_distance, avg_vendor_qty)

if st.sidebar.button("Predict Freight Cost"):
    predicted_freight = freight_model.predict(X_freight)[0]
    st.subheader("Predicted Freight Cost")
    st.metric(label="Freight ($)", value=f"${predicted_freight:.2f}")


st.header("Invoice Anomaly Detection")

def prepare_anomaly_features(invoice_quantity, invoice_dollars, freight, total_brands, total_item_quantity, total_item_dollars):
    total_item_dollars = max(total_item_dollars, 1)
    total_item_quantity = max(total_item_quantity, 1)
    total_brands = max(total_brands, 1)

    invoice_ratio = invoice_dollars / total_item_dollars
    freight_per_unit = freight / invoice_quantity
    quantity_per_brand = invoice_quantity / total_brands
    relative_order_size = invoice_quantity / total_item_quantity

    log_invoice_dollars = np.log1p(invoice_dollars)

    return pd.DataFrame([{
        "total_brands": total_brands,
        "total_item_quantity": total_item_quantity,
        "total_item_dollars": total_item_dollars,
        "quantity_per_brand": quantity_per_brand,
        "relative_order_size": relative_order_size,
        "log_invoice_dollars": log_invoice_dollars
    }])


invoice_quantity_a = st.number_input("Invoice Quantity", min_value=1, value=50, key="qty_a" ,help="Total number of items listed on this invoice.")
invoice_dollars_a = st.number_input("Invoice Total ($)", min_value=1.0, value=500.0, key="dollars_a", help="Total dollar amount billed on this invoice.")
freight_a = st.number_input("Freight ($)", min_value=0.0, value=50.0, help="Cost charged for shipping the invoice items.")
total_brands_a = st.number_input("Number of Brands", min_value=1, value=2, help="Number of distinct brands included in the invoice.")
total_item_quantity_a = st.number_input("Total Item Quantity", min_value=1, value=60, help="Sum of quantities across all items in this invoice.")
total_item_dollars_a = st.number_input("Total Item Dollars", min_value=1.0, value=550.0, help="Sum of item-level prices (before any discounts or adjustments).")


X_anomaly = prepare_anomaly_features(invoice_quantity_a, invoice_dollars_a, freight_a, total_brands_a, total_item_quantity_a, total_item_dollars_a)

X_anomaly_scaled = scaler.transform(X_anomaly)

col1, col2 = st.columns(2)

with col1:
    if st.button("Check Invoice Risk (Supervised)"):
        risk_prob = anomaly_model.predict_proba(X_anomaly_scaled)[0, 1]
        risk_flag = "Suspicious" if risk_prob > 0.5 else "Normal"
        st.subheader("Supervised Model Risk")
        st.metric(label="Suspicious Probability", value=f"{risk_prob:.2f}")
        st.write("Status:", risk_flag)

with col2:
    if st.button("Check Invoice Risk (Unsupervised)"):
        iso_flag = iso_model.predict(X_anomaly_scaled)[0]
        iso_status = "Outlier" if iso_flag == -1 else "Normal"
        st.subheader("Isolation Forest Status")
        st.write("Status:", iso_status)