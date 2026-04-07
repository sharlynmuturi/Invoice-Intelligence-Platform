# Invoice Intelligence Platform

### Freight Prediction & Invoice Anomaly Detection

This project builds an **end-to-end data science pipeline** for:

- Generating realistic supply chain data (procurement & invoice data)
- Predicting **freight costs** for new purchase orders
- Detecting **suspicious** or **unusual** invoices using supervised and unsupervised models.
- Deploying an interactive **Streamlit dashboard** for business use

Can be used by:

- **Procurement Team** - Estimate freight before placing orders, optimize vendor selection
- **Finance Team** - Detect overbilling early, prioritize invoice audits
- **Operations** - Identify inefficient vendors, monitor logistics cost drivers

#### Generated Tables

| Table | Description |
| --- | --- |
| purchase_prices | Product catalog with vendor mapping |
| purchases | Line-level purchase transactions |
| vendor_invoice | Aggregated invoices per PO |
| begin_inventory / end_inventory | Inventory snapshots |

#### Feature Engineering

##### Freight Prediction Features

| Feature | Meaning |
| --- | --- |
| quantity | Shipment size |
| log_quantity | Reduces skew |
| vendor_distance | Transport cost driver |
| relative_order_size | Order vs vendor average |
| qty_distance_interaction | Core cost driver |

##### Anomaly Detection Features

| Feature | Meaning |
| --- | --- |
| invoice_ratio | Invoice vs item-level total |
| freight_per_unit | Cost efficiency |
| quantity_per_brand | Complexity indicator |
| relative_order_size | Abnormal order size |
| log_invoice_dollars | Stabilized scale |


#### Freight Prediction Model - Random Forest Regressor

Freight is driven mainly by:
- Distance × Quantity interaction
- Order size context


#### Anomaly Detection System

- Supervised Model - **Random Forest Classifier**
- Unsupervised Model - **Isolation Forest**

### How to Run

#### 1. Clone Repository

```bash
git clone https://github.com/sharlynmuturi/Invoice-Intelligence-Platform.git  
cd Invoice-Intelligence-Platform
```

### 2\. Install Dependencies

```bash
pip install -r full-requirements.txt
```

#### 3. Generate Data

```bash
python scripts/generate_data.py
```
#### 4. Train Models

Run notebooks:

- EDA
- Freight prediction
- Anomaly detection


#### 5. Launch App

```bash
streamlit run app.py
```