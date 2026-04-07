# Invoice Intelligence Platform

### Freight Prediction & Invoice Anomaly Detection

This project builds an **end-to-end data science pipeline** for:

- Generating realistic supply chain data (procurement & invoice data)
- Predicting **freight costs** for new purchase orders
- Detecting **suspicious** or **unusual** invoices using supervised and unsupervised models.
- Deploying an interactive **Streamlit dashboard** for business use

Can be used by:

**Procurement Team** - Estimate freight before placing orders, optimize vendor selection
**Finance Team** - Detect overbilling early, prioritize invoice audits
**Operations** - Identify inefficient vendors, monitor logistics cost drivers

### How to Run

#### 1. Clone Repository

```bash
git clone https://github.com/sharlynmuturi/Invoice-Intelligence-Platform.git  
cd Invoice-Intelligence-Platform
```

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
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