# Retail Customer Transaction Analysis – EDA Portfolio Project

## Overview
This project explores a synthetic **retail customer transaction dataset** (~302k rows, 30 columns) spanning March 2023 to February 2024. The goal is to demonstrate professional **exploratory data analysis (EDA)**, data cleaning, and business insight generation for a data analyst portfolio.

## Dataset

- **Source**: Retail Customer Transaction Data (Opendatabay, CC0 Public Domain)
- **File**: `new_retail_data.csv`
- **Size**: ~84.9 MB, ~302,000 rows, 30 columns
- **Time period**: March 2023 – February 2024
- **Scope**:
  - Customers: demographics (age, gender, income, segment)
  - Transactions: dates, times, amounts, totals
  - Products: category, brand, type, product names
  - Operations: payment and shipping methods, order status, ratings, feedback

Key columns include:

- Customer: `Customer_ID`, `Age`, `Gender`, `Income`, `Customer_Segment`, `Country`, `State`, `City`
- Transaction: `Transaction_ID`, `Date`, `Time`, `Total_Purchases`, `Amount`, `Total_Amount`
- Product: `Product_Category`, `Product_Brand`, `Product_Type`, `products`
- Experience: `Ratings`, `Feedback`, `Shipping_Method`, `Payment_Method`, `Order_Status`

## Objectives

1. Clean and validate the dataset (missing values, duplicates, ranges).
2. Understand customer demographics and purchasing behavior.
3. Analyze sales trends over time (daily, monthly, seasonality).
4. Evaluate product and category performance.
5. Explore geographic patterns in sales.
6. Investigate satisfaction drivers (ratings, feedback, order status).
7. Produce clear visualizations and business recommendations.

## Tech Stack

- **Python 3.x**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Jupyter Notebook**

## Project Structure

```text
retail-eda-portfolio/
├── data/
│   └── new_retail_data.csv         # raw dataset (ignored by git)
├── notebooks/
│   └── retail_eda_analysis.ipynb   # main EDA notebook
├── outputs/
│   └── *.png                       # saved charts
├── .cursorrules                    # Cursor rules for consistent code style
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup & Installation

```bash
# Clone or download the project
cd retail-eda-portfolio

# Create and activate virtual environment (Linux/Mac)
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Place `new_retail_data.csv` in the `data/` folder.

## How to Run the Analysis

```bash
cd retail-eda-portfolio
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
jupyter notebook notebooks/retail_eda_analysis.ipynb
```

In Jupyter, run all cells to:

1. Load and clean the dataset.
2. Generate univariate and bivariate analyses.
3. Produce time-series and categorical visualizations.
4. Summarize key insights.

To export an HTML report:

```bash
jupyter nbconvert --to html notebooks/retail_eda_analysis.ipynb --output retail_eda_analysis.html
```

## Key Analyses & Visuals

The notebook includes:

- Data quality checks (missing values, duplicates, range validation).
- Distributions of age, income, customer segments, and ratings.
- Sales trends by date, month, and time of day.
- Revenue by product category, brand, and product type.
- Sales by country, state, and city.
- Payment and shipping method usage.
- Relationships between ratings/feedback and order characteristics.

Example visualizations:

- Monthly revenue line charts and seasonality plots.
- Bar charts of revenue by category and customer segment.
- Heatmaps and boxplots for satisfaction vs product/operations.

## Example Insights

- Identification of high-value segments (e.g., Premium customers) and their share of total revenue.
- Detection of peak sales months and hours, supporting staffing and promotion decisions.
- Product categories and brands contributing the largest share of revenue.
- Operational patterns in payment and shipping methods.
- Factors associated with higher ratings and positive feedback.

## Potential Extensions

- Customer segmentation using clustering.
- RFM (Recency, Frequency, Monetary) analysis.
- Cohort retention analysis.
- Building a Tableau or Power BI dashboard on top of the cleaned dataset.

## License

- **Dataset**: CC0 (Public Domain) from Opendatabay.
- **Code**: MIT License (or your chosen license).

## Contact

For questions or collaboration opportunities, please reach out via GitHub or email.
