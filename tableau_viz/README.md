# tableau_viz/

This folder contains the data files and (eventually) the packaged Tableau workbook for the **2025 E-com RFM Dashboard**.

## Files

| File | Description |
|------|-------------|
| `rfm_scores.csv` | One row per customer with RFM quintile scores (1–5) and total revenue |
| `full_ecom.csv` | Individual order-level transactions used to build the RFM table and trend charts |
| `Ecom_RFM_Dashboard.twbx` | *(coming soon)* Tableau Public packaged workbook |

## Column Reference

### rfm_scores.csv

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | String | Unique customer identifier (e.g. `CUST_0001`) |
| `r_score` | Integer 1–5 | Recency score – 5 = most recent purchase |
| `f_score` | Integer 1–5 | Frequency score – 5 = most frequent buyer |
| `m_score` | Integer 1–5 | Monetary score – 5 = highest total spend |
| `customers` | Integer | Row count (1 per customer; useful for COUNT aggregation in Tableau) |
| `total_revenue` | Float | Cumulative spend for this customer across all orders |

### full_ecom.csv

| Column | Type | Description |
|--------|------|-------------|
| `Order_Date` | Date (YYYY-MM-DD) | Date the order was placed |
| `customer_id` | String | Links to `rfm_scores.customer_id` |
| `Sales` | Float | Revenue for this order line |
| `Category` | String | Product category (Electronics, Clothing, Books, …) |
| `Device` | String | Device used to place the order (Desktop, Mobile, Tablet) |
| `Country` | String | Shipping country |
| `State` | String | Shipping state / province |

## Tableau Quick-Start

1. Open **Tableau Public** → **New Workbook**
2. **Connect → Text file** → select `rfm_scores.csv`
3. Build the **RFM Heatmap** sheet:
   - Rows: `r_score` (Discrete)
   - Columns: `f_score` (Discrete)
   - Marks → Color: `customer_id` as **Count Distinct**
   - Marks type → **Square**
4. Add a second data source: **Data → New Data Source → Text file** → `full_ecom.csv`
5. Relate the two sources on `customer_id`
6. Build the **Sales Trend** sheet using `Order_Date` and `Sales`
7. Assemble sheets on a new **Dashboard** tab
8. Save as `Ecom_RFM_Dashboard.twbx` (File → Save As… → packaged workbook)

See the full step-by-step tutorial in the project issue for detailed Tableau click instructions.
