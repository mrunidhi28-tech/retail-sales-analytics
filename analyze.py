import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = "retail_transactions_50mb.npy"
OUTPUT_CSV = "cleaned_transactions.csv"
OUTPUT_XLSX = "excel_analysis.xlsx"
CHART_FILE = "monthly_sales_trend.png"
LOG_FILE = "cleaning_log.txt"


def load_data(file_name: str) -> pd.DataFrame:
    """Load the raw .npy file into a DataFrame."""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")

    raw_data = np.load(file_name, allow_pickle=False)
    df = pd.DataFrame.from_records(raw_data)
    return df


def write_cleaning_log(lines):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


# --------------------------
# 1. Load raw data
# --------------------------
raw_df = load_data(DATA_FILE)
print("Raw file loaded successfully.")
print(f"Initial rows: {len(raw_df)}")
print(f"Initial columns: {list(raw_df.columns)}")
print(raw_df.head())

# --------------------------
# 2. Validate expected schema
# --------------------------
expected_columns = [
    "order_id",
    "order_date",
    "customer_id",
    "region",
    "city",
    "category",
    "product",
    "channel",
    "quantity",
    "unit_price",
    "discount_pct",
    "shipping_cost",
    "payment_method",
    "customer_segment",
    "returned",
]

missing_columns = [col for col in expected_columns if col not in raw_df.columns]
if missing_columns:
    print(f"Missing expected columns: {missing_columns}")

# --------------------------
# 3. Data quality checks
# --------------------------
cleaning_lines = []
cleaning_lines.append("Retail Sales Data Cleaning Log")
cleaning_lines.append(f"Initial rows: {len(raw_df)}")
cleaning_lines.append(f"Initial columns: {list(raw_df.columns)}")

# Duplicate orders
duplicate_order_count = raw_df["order_id"].duplicated().sum() if "order_id" in raw_df.columns else "N/A"
cleaning_lines.append(f"Duplicate order IDs found: {duplicate_order_count}")

# Missing values
missing_values = raw_df.isnull().sum()
cleaning_lines.append("Missing values per column:")
for col, count in missing_values.items():
    if count > 0:
        cleaning_lines.append(f"- {col}: {count}")

# Date range
if "order_date" in raw_df.columns:
    raw_df["order_date"] = pd.to_datetime(raw_df["order_date"], errors="coerce")
    cleaning_lines.append(f"Date range before cleaning: {raw_df['order_date'].min()} to {raw_df['order_date'].max()}")

# --------------------------
# 4. Cleaning logic
# --------------------------
# Fill missing discount with zero as a start point, but record it in the log.
if "discount_pct" in raw_df.columns:
    raw_df["discount_pct"] = raw_df["discount_pct"].fillna(0)
    cleaning_lines.append("Filled missing discount_pct values with 0.")

if "city" in raw_df.columns:
    raw_df["city"] = raw_df["city"].fillna("Unknown")
    cleaning_lines.append("Filled missing city values with 'Unknown'.")

if "shipping_cost" in raw_df.columns:
    raw_df["shipping_cost"] = raw_df["shipping_cost"].fillna(0)
    cleaning_lines.append("Filled missing shipping_cost values with 0.")

# Normalize return flags
if "returned" in raw_df.columns:
    raw_df["returned"] = raw_df["returned"].astype(str).str.strip().str.lower()
    raw_df["returned"] = raw_df["returned"].replace({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "nan": False})
    raw_df["returned"] = raw_df["returned"].fillna(False)
    raw_df["returned"] = raw_df["returned"].astype(bool)
    cleaning_lines.append("Normalized returned field to boolean values.")

# Remove duplicate order ids
if "order_id" in raw_df.columns:
    raw_df = raw_df.drop_duplicates(subset="order_id", keep="first")
    cleaning_lines.append("Removed duplicate order_id rows.")

# Convert numeric columns to numeric types
numeric_columns = ["quantity", "unit_price", "discount_pct", "shipping_cost"]
for col in numeric_columns:
    if col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

# --------------------------
# 5. Business metrics
# --------------------------
raw_df["gross_sales"] = raw_df["quantity"] * raw_df["unit_price"]
raw_df["discount_amount"] = raw_df["gross_sales"] * (raw_df["discount_pct"] / 100)
raw_df["net_sales"] = raw_df["gross_sales"] - raw_df["discount_amount"]

# Additional useful metrics
if "order_id" in raw_df.columns:
    raw_df["order_value"] = raw_df["net_sales"]

# --------------------------
# 6. Summary outputs
# --------------------------
print("\nAfter cleaning: ")
print(f"Rows after cleaning: {len(raw_df)}")
print(f"Columns after cleaning: {list(raw_df.columns)}")
print(raw_df.head())

# Save cleaned dataset
raw_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nCleaned CSV exported: {OUTPUT_CSV}")

# Save Excel workbook for Excel phase
raw_df.to_excel(OUTPUT_XLSX, index=False)
print(f"Excel file exported: {OUTPUT_XLSX}")

# Save cleaning log
write_cleaning_log(cleaning_lines)
print(f"Cleaning log created: {LOG_FILE}")

# --------------------------
# 7. Monthly trend chart
# --------------------------
monthly_sales = raw_df.set_index("order_date").resample("M")["net_sales"].sum().reset_index()
monthly_sales.columns = ["order_date", "net_sales"]

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales["order_date"], monthly_sales["net_sales"], marker="o")
plt.title("Monthly Net Sales Trend")
plt.xlabel("Month")
plt.ylabel("Net Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(CHART_FILE)
print(f"Chart saved: {CHART_FILE}")

# --------------------------
# 8. Final summary statistics
# --------------------------
total_orders = raw_df["order_id"].nunique() if "order_id" in raw_df.columns else len(raw_df)
total_net_sales = raw_df["net_sales"].sum() if "net_sales" in raw_df.columns else 0
avg_order_value = total_net_sales / total_orders if total_orders else 0
return_rate = raw_df["returned"].mean() if "returned" in raw_df.columns else 0

print("\nKey results:")
print(f"Unique orders: {total_orders}")
print(f"Net sales: {total_net_sales:,.2f}")
print(f"Average order value: {avg_order_value:,.2f}")
print(f"Return rate: {return_rate:.2%}")

print("\nPython phase completed successfully.")
