"""
Real Estate Sales Data Analysis and Visualization
This script performs exploratory data analysis on rolling sale data including:
- Data cleaning and filtering
- Distribution analysis
- Correlation analysis
- Outlier detection using Price Per Square Foot (PPSF)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("Loading data...")
df = pd.read_excel("rolling_sale_data.xlsx")
print(f"Initial data shape: {df.shape}")
print(df.head())

# =============================================================================
# 2. DATA CLEANING
# =============================================================================
print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

# Bỏ các cột không cần thiết hoặc có quá nhiều giá trị thiếu
columns_to_drop = ['EASEMENT', 'APARTMENT NUMBER', 'ADDRESS'] 
df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")

# Lọc bỏ R* (condo units) ngoại trừ RR (CONDOMINIUM RENTALS)
mask_R  = df["BUILDING CLASS AT TIME OF SALE"].astype(str).str.startswith("R")
mask_RR = df["BUILDING CLASS AT TIME OF SALE"].astype(str).eq("RR")
raw_data = df[~mask_R].copy()
print(f"Sau khi bỏ R*: {raw_data.shape}")
print(raw_data["BUILDING CLASS AT TIME OF SALE"].value_counts().head())

# Lọc bỏ coop classes
coop_classes = ["D4", "D0", "C6", "C8", "A8", "CC", "DC", "H7"]
mask_coop = raw_data["BUILDING CLASS AT TIME OF SALE"].astype(str).isin(coop_classes)
raw_data = raw_data[~mask_coop].copy()
print(f"Sau khi bỏ coop: {raw_data.shape}")
print(raw_data["BUILDING CLASS AT TIME OF SALE"].value_counts().head())

# Loại bỏ SALE PRICE = 0
raw_data = raw_data[raw_data["SALE PRICE"] != 0]
print(f"\nFinal data shape after cleaning: {raw_data.shape}")
raw_data.info()

# =============================================================================
# 3. DATA TYPE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("DATA TYPE ANALYSIS")
print("="*80)

# Phân biệt kiểu dữ liệu categorical và numerical
categorical_cols = raw_data.select_dtypes(include=['object', 'category']).columns
numerical_cols = raw_data.select_dtypes(include=['number']).columns
print(f"Categorical columns: {list(categorical_cols)}")
print(f"Numerical columns: {list(numerical_cols)}")

# =============================================================================
# 4. DISTRIBUTION ANALYSIS - CATEGORICAL FEATURES
# =============================================================================
print("\n" + "="*80)
print("CATEGORICAL VARIABLES DISTRIBUTION")
print("="*80)

n_cols = 2
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    ax = axes[i]
    data = raw_data[col].value_counts().head(10).sort_values(ascending=False)
    
    # Vẽ bar chart với palette tab20 cho category
    colors = sns.color_palette("tab20", len(data))
    data.plot(kind='barh', ax=ax, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Frequency', fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(f'Bar Chart of {col}', fontsize=12, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Ẩn các subplot trống nếu có
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# =============================================================================
# 5. DISTRIBUTION ANALYSIS - NUMERICAL FEATURES
# =============================================================================
print("\n" + "="*80)
print("NUMERICAL VARIABLES DISTRIBUTION")
print("="*80)

cols = [
    'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 
    'LAND SQUARE FEET', 'GROSS SQUARE FEET']

# Đặt thông số cắt tail (phần trăm)
clip_percentile = 0.99

plt.figure(figsize=(12, 12))
sns.set_style("white")

n_cols = 2
n_rows = (len(cols) + n_cols - 1) // n_cols

for i, col in enumerate(cols, 1):
    ax = plt.subplot(n_rows, n_cols, i)

    # Lấy dữ liệu và cắt tail để tránh nén về 0
    data = raw_data[col].dropna()
    upper = np.percentile(data, clip_percentile*100)
    clipped = data[data <= upper]

    # Vẽ histogram + KDE
    sns.histplot(clipped, bins=40, kde=True)

    plt.title(f"Distribution of {col}\n(Clipped at {clip_percentile*100}th percentile)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.axvline(clipped.mean(), color='red', linestyle='--', label=f'Mean: {clipped.mean():.2f}')
    plt.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# =============================================================================
# 6. CORRELATION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

features = [
    'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS',
    'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT'
 ]
target = 'SALE PRICE'

# 1) Tính ma trận tương quan (loại NaN ở YEAR BUILT)
corr_data = raw_data[features + [target]].dropna(subset=['YEAR BUILT'])
corr_matrix = corr_data.corr()

# Đổi tên cột để xuống dòng (dễ đọc hơn)
new_labels = {
    'RESIDENTIAL UNITS': 'RESIDENTIAL\nUNITS',
    'COMMERCIAL UNITS': 'COMMERCIAL\nUNITS',
    'TOTAL UNITS': 'TOTAL\nUNITS',
    'LAND SQUARE FEET': 'LAND SQ.\nFEET',
    'GROSS SQUARE FEET': 'GROSS SQ.\nFEET',
    'YEAR BUILT': 'YEAR\nBUILT',
    'SALE PRICE': 'SALE\nPRICE'
}
corr_matrix = corr_matrix.rename(index=new_labels, columns=new_labels)

# Tăng kích thước figure để tên biến không bị dính
plt.figure(figsize=(11, 9))

# 2) Mask tam giác trên để chỉ hiển thị tam giác dưới
mask_lower = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)

# 3) Colormap
gray_cmap = sns.light_palette("gray", as_cmap=True)
target_cmap = sns.color_palette("coolwarm", as_cmap=True)

ax = plt.gca()

# 4) Lớp nền xám cho toàn bộ tam giác dưới
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap=gray_cmap,
    vmin=-1, vmax=1, center=0,
    linewidths=.5,
    cbar=False,
    mask=mask_lower,
    ax=ax
)

# 5) Lớp nổi bật: chỉ hàng target 'SALE PRICE' (nằm ngang)
target_idx = corr_matrix.index.get_loc('SALE\nPRICE')
highlight_matrix = pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)
highlight_matrix.loc['SALE\nPRICE'] = corr_matrix.loc['SALE\nPRICE']

# Mask để giữ lại đúng hàng target ở tam giác dưới, bỏ cột dọc
mask_row = np.ones_like(corr_matrix, dtype=bool)
mask_row[target_idx, :] = False
mask_highlight = np.logical_or(mask_lower, mask_row)

sns.heatmap(
    highlight_matrix,
    annot=True,
    fmt=".2f",
    cmap=target_cmap,
    vmin=-1, vmax=1, center=0,
    linewidths=.5,
    cbar_kws={"shrink": .8, "label": "Correlation coefficient with SALE PRICE"},
    mask=mask_highlight,
    ax=ax
)

plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.title(
    "Correlation Between Property Attributes & SALE PRICE",
    fontsize=16,
    fontweight='bold'
 )

plt.tight_layout()
plt.show()

# =============================================================================
# 7. YEAR BUILT ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("YEAR BUILT ANALYSIS")
print("="*80)

# Vẽ Histogram của biến YEAR BUILT và mode của nó
year_built_mode = raw_data["YEAR BUILT"].mode()[0]
print(f"Mode of YEAR BUILT: {int(year_built_mode)}")

plt.figure(figsize=(10, 6))
plt.hist(raw_data["YEAR BUILT"].dropna(), bins=30, color='#20B2AA', edgecolor='white', linewidth=0.5)
plt.axvline(year_built_mode, color='red', linestyle='--', linewidth=2.5, label=f'Mode: {int(year_built_mode)}')
plt.xlabel('YEAR BUILT', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Histogram of YEAR BUILT', fontsize=13, fontweight='bold', pad=15)
plt.legend(fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# =============================================================================
# 8. NOISE DETECTION - PRICE PER SQUARE FOOT (PPSF)
# =============================================================================
print("\n" + "="*80)
print("NOISE DETECTION USING PPSF")
print("="*80)

# Tính Price Per Square Foot (PPSF) để phát hiện noise
def compute_ppsf(data, eps=1e-6):
    """
    Tính Price Per Square Foot (PPSF)
    - Ưu tiên GROSS SQUARE FEET nếu > 0
    - Nếu GROSS = 0, dùng LAND SQUARE FEET
    - eps: tránh chia cho 0
    """
    sqft = np.where(
        data["GROSS SQUARE FEET"] > 0,
        data["GROSS SQUARE FEET"],
        data["LAND SQUARE FEET"],
    )
    return data["SALE PRICE"] / np.maximum(sqft, eps)

# Tính PPSF cho toàn bộ dữ liệu
raw_data_with_ppsf = raw_data.copy()
raw_data_with_ppsf["PPSF"] = compute_ppsf(raw_data_with_ppsf)

# Tính ngưỡng PPSF (1%-99% quantile)
q_low, q_high = raw_data_with_ppsf["PPSF"].quantile([0.01, 0.99])
print(f"Ngưỡng PPSF (1%-99%): ${q_low:.2f} - ${q_high:.2f}")

# Phân loại noise: PPSF nằm ngoài khoảng [q_low, q_high]
noise_mask = ~raw_data_with_ppsf["PPSF"].between(q_low, q_high)
noise_data = raw_data_with_ppsf[noise_mask]
normal_data = raw_data_with_ppsf[~noise_mask]

print(f"\nSố lượng records 'noise' (PPSF outliers): {len(noise_data)}")
print(f"Số lượng records bình thường: {len(normal_data)}")
print(f"\nThống kê 'noise' data:")
print(noise_data[['GROSS SQUARE FEET', 'LAND SQUARE FEET', 'SALE PRICE', 'PPSF']].describe())

# Tạo visualization - scatter plot với extreme values được đánh dấu
fig, ax = plt.subplots(figsize=(12, 6))

# Tính giới hạn để "kéo gần" các điểm outliers quá xa
ppsf_max_display = raw_data_with_ppsf["PPSF"].quantile(0.999)
price_max_display = raw_data_with_ppsf["SALE PRICE"].quantile(0.999)

# Tạo bản sao để clip các giá trị quá xa
normal_data_clipped = normal_data.copy()
noise_data_clipped = noise_data.copy()

# Đánh dấu các điểm bị clip (kéo gần)
normal_extreme_ppsf = normal_data["PPSF"] > ppsf_max_display
normal_extreme_price = normal_data["SALE PRICE"] > price_max_display
noise_extreme_ppsf = noise_data["PPSF"] > ppsf_max_display
noise_extreme_price = noise_data["SALE PRICE"] > price_max_display

# Clip giá trị
normal_data_clipped["PPSF"] = normal_data["PPSF"].clip(upper=ppsf_max_display)
normal_data_clipped["SALE PRICE"] = normal_data["SALE PRICE"].clip(upper=price_max_display)
noise_data_clipped["PPSF"] = noise_data["PPSF"].clip(upper=ppsf_max_display)
noise_data_clipped["SALE PRICE"] = noise_data["SALE PRICE"].clip(upper=price_max_display)

# Scatter plot: PPSF vs Sale Price (dữ liệu đã clip)
ax.scatter(normal_data_clipped["SALE PRICE"], normal_data_clipped["PPSF"], 
           alpha=0.5, s=30, label='Normal', color='steelblue')
ax.scatter(noise_data_clipped["SALE PRICE"], noise_data_clipped["PPSF"], 
           alpha=0.5, s=30, label=f'Noise (PPSF outliers)', color='red')

# Đánh dấu các điểm bị kéo gần (extreme outliers) bằng marker đặc biệt
normal_extreme = normal_data_clipped[normal_extreme_ppsf | normal_extreme_price]
noise_extreme = noise_data_clipped[noise_extreme_ppsf | noise_extreme_price]

if len(normal_extreme) > 0:
    ax.scatter(normal_extreme["SALE PRICE"], normal_extreme["PPSF"], 
               alpha=0.5, s=30, color='blue', 
               label=f'Normal (extreme, n={len(normal_extreme)})')
if len(noise_extreme) > 0:
    ax.scatter(noise_extreme["SALE PRICE"], noise_extreme["PPSF"], 
               alpha=0.8, s=100, marker='x', color='darkred', linewidths=2,
               label=f'Noise (extreme, n={len(noise_extreme)})')

ax.set_xlabel('Sale Price ($)', fontsize=11)
ax.set_ylabel('Price Per Square Foot ($/sqft)', fontsize=11)
ax.set_title('Noise Detection: PPSF Outliers (Extreme values clipped and marked with X)', fontsize=12, fontweight='bold')
ax.axhline(y=q_low, color='green', linestyle='--', linewidth=1.5, label=f'1% quantile: ${q_low:.0f}')
ax.axhline(y=q_high, color='orange', linestyle='--', linewidth=1.5, label=f'99% quantile: ${q_high:.0f}')
ax.legend(fontsize=9, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# =============================================================================
# 9. OUTLIER FILTERING SUMMARY
# =============================================================================
print("\n" + "="*80)
print("OUTLIER FILTERING SUMMARY")
print("="*80)

# Biểu đồ cột thể hiện số lượng outliers và dữ liệu giữ lại
fig, ax = plt.subplots(figsize=(6, 4))

# Đếm số lượng
removed = len(noise_data)  # Bỏ đi (outliers)
kept = len(normal_data)     # Giữ lại (normal)
total = removed + kept

# Vẽ cột - phần giữ lại (dữ liệu) - cùng màu với scatter (steelblue)
ax.bar(0, kept, width=0.5, color='steelblue', alpha=0.85, label=f'Kept: {kept:,}')

# Vẽ cột - phần bỏ đi (trên) - cùng màu với scatter outliers
ax.bar(0, removed, width=0.5, bottom=kept, color='red', alpha=0.85, label=f'Removed: {removed:,}')

# Đường chia ngang
ax.axhline(y=kept, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Formatting
ax.set_xlim(-0.8, 0.8)
ax.set_ylabel('Number of Records', fontsize=11)
ax.set_title(f'PPSF Outlier Filtering (1%-99% Quantile)\nTotal: {total:,} records', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks([])
ax.legend(loc='upper right', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Y-axis formatting
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.tight_layout()
plt.show()

print(f"\nTotal: {total:,} | Kept: {kept:,} ({kept/total*100:.1f}%) | Removed: {removed:,} ({removed/total*100:.1f}%)")

# =============================================================================
# 10. TAX CLASS DISTRIBUTION
# =============================================================================
print("\n" + "="*80)
print("TAX CLASS DISTRIBUTION")
print("="*80)

# Vẽ histogram cho biến tax class at time of sale
plt.figure(figsize=(10, 6))
raw_data['TAX CLASS AT TIME OF SALE'].value_counts().plot(kind='bar')
plt.title('Distribution of TAX CLASS AT TIME OF SALE')
plt.xlabel('TAX CLASS AT TIME OF SALE')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

# =============================================================================
# 11. SALE PRICE - LOG TRANSFORMATION
# =============================================================================
print("\n" + "="*80)
print("SALE PRICE - LOG TRANSFORMATION ANALYSIS")
print("="*80)

# Vẽ subplot cho biến SALE PRICE để so sánh trước và sau log
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cắt tail để tránh nén về 0
trimmed_data = raw_data_with_ppsf[raw_data_with_ppsf['SALE PRICE'] < raw_data_with_ppsf['SALE PRICE'].quantile(0.99)]

# Trước log
sns.histplot(trimmed_data['SALE PRICE'], bins=50, kde=True, ax=axes[0], color='steelblue')  
axes[0].set_title('SALE PRICE Distribution (Before Log Transformation)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('SALE PRICE ($)', fontsize=12)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Sau log
sns.histplot(np.log1p(trimmed_data['SALE PRICE']), bins=50, kde=True, ax=axes[1], color='steelblue')
axes[1].set_title('SALE PRICE Distribution (After Log Transformation)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('SALE PRICE (log scale)', fontsize=12)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETED")
print("="*80)
