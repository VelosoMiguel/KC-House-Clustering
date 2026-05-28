"""# 🏠 Seattle Real Estate Market Segmentation
## K-Means Clustering Analysis — King County, WA (2014–2015)

---

**Goal:** Use unsupervised machine learning to segment the King County housing market into meaningful groups,  
revealing hidden patterns that pricing alone cannot capture.

**Dataset:** 21,613 residential property sales from May 2014 to May 2015  
**Method:** K-Means Clustering with PCA for dimensionality reduction  
**Key Finding:** Three structurally distinct market segments emerge — each with a clear real-estate narrative.

---
*Author: Miguel | CS Portfolio Project*"""
"""## 1. Imports & Configuration"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# ── Plotting style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0f1117',
    'axes.facecolor':    '#0f1117',
    'axes.edgecolor':    '#2a2d3a',
    'axes.labelcolor':   '#c9d1d9',
    'xtick.color':       '#8b949e',
    'ytick.color':       '#8b949e',
    'text.color':        '#c9d1d9',
    'grid.color':        '#21262d',
    'grid.linewidth':    0.8,
    'font.family':       'monospace',
    'figure.dpi':        120,
})

CLUSTER_COLORS = ['#58a6ff', '#f78166', '#3fb950']
CLUSTER_LABELS = {0: 'Cluster 0 — Premium Modern', 1: 'Cluster 1 — Waterfront Luxury', 2: 'Cluster 2 — Entry-Level Classic'}

pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
pd.set_option('display.max_columns', None)

print("✓ Libraries loaded")
"""## 2. Data Loading & Exploratory Analysis

We start by loading the dataset and getting a feel for its shape, distributions, and quality."""
df = pd.read_csv('kc_house_data.csv')

print(f"Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Date range   : {df['date'].min()[:8]} → {df['date'].max()[:8]}")
print(f"\nPrice range  : ${df['price'].min():,.0f} → ${df['price'].max():,.0f}")
print(f"Median price : ${df['price'].median():,.0f}")

df.describe().T[['mean','std','min','50%','max']].round(2)
# ── Price distribution ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Price Distribution — King County Housing', fontsize=14, fontweight='bold', y=1.02)

# Raw
axes[0].hist(df['price'] / 1e6, bins=60, color='#58a6ff', edgecolor='none', alpha=0.85)
axes[0].set_xlabel('Price ($ millions)')
axes[0].set_ylabel('Count')
axes[0].set_title('Raw Distribution')
axes[0].axvline(df['price'].median() / 1e6, color='#f78166', lw=1.5, linestyle='--', label='Median')
axes[0].legend()

# Log-transformed
axes[1].hist(np.log1p(df['price']), bins=60, color='#3fb950', edgecolor='none', alpha=0.85)
axes[1].set_xlabel('log(Price)')
axes[1].set_title('Log-Transformed (more symmetric)')

for ax in axes:
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('01_price_distribution.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("The raw distribution is heavily right-skewed — a few ultra-luxury homes pull the mean well above the median.")
# ── Correlation heatmap ───────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include='number').drop(columns=['id','zipcode']).columns
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(
    corr, mask=mask, cmap=cmap, center=0,
    annot=True, fmt='.2f', annot_kws={'size': 7},
    linewidths=0.4, linecolor='#21262d',
    cbar_kws={'shrink': 0.8}, ax=ax
)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("Key: sqft_living correlates strongly with price (0.70), grade (0.76), and sqft_above (0.88).")
print("Multicollinearity is expected in housing data — StandardScaler handles the scale differences.")
"""## 3. Preprocessing & Feature Engineering

Steps:
1. Drop non-informative identifiers (`id`, `date`)
2. Remove extreme outliers using the 1st–99th percentile on `sqft_living`
3. Select semantically meaningful features for clustering
4. Standardise with `StandardScaler` (zero mean, unit variance)"""
# ── Drop identifiers ─────────────────────────────────────────────────────────
df_clean = df.drop(columns=['id', 'date'])

# ── Outlier removal on sqft_living ───────────────────────────────────────────
q_low  = df_clean['sqft_living'].quantile(0.01)
q_high = df_clean['sqft_living'].quantile(0.99)
before = len(df_clean)
df_clean = df_clean[(df_clean['sqft_living'] >= q_low) & (df_clean['sqft_living'] <= q_high)].copy()
print(f"Removed {before - len(df_clean):,} extreme outliers ({(before - len(df_clean))/before:.1%} of data)")

# ── Feature selection ─────────────────────────────────────────────────────────
FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'yr_built', 'lat', 'long'
]
df_feat = df_clean[FEATURES].copy()

# ── Scaling ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat)
df_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

print(f"\nFinal dataset : {df_feat.shape[0]:,} rows × {df_feat.shape[1]} features")
print(f"Features used : {FEATURES}")
print(f"\nScaled stats (should be ~0 mean, ~1 std):")
pd.DataFrame(X_scaled, columns=FEATURES).describe().loc[['mean','std']].round(3)
"""## 4. Finding the Optimal Number of Clusters

Two complementary methods:
- **Elbow Method** (inertia): look for the "elbow" where adding more clusters gives diminishing returns
- **Silhouette Score**: measures how well each point fits its own cluster vs. neighbouring clusters (higher = better)"""
k_range = range(2, 11)
inertias, silhouettes = [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  k={k}  |  inertia={km.inertia_:>12,.0f}  |  silhouette={silhouettes[-1]:.4f}")

optimal_k = list(k_range)[np.argmax(silhouettes)]
print(f"\n→ Optimal k = {optimal_k}  (highest silhouette score: {max(silhouettes):.4f})")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Selecting the Optimal Number of Clusters', fontsize=14, fontweight='bold')

ks = list(k_range)

# Elbow
ax1.plot(ks, inertias, marker='o', color='#58a6ff', lw=2, markersize=6)
ax1.axvline(optimal_k, color='#f78166', linestyle='--', lw=1.5, label=f'k={optimal_k}')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-Cluster SSE)')
ax1.set_title('Elbow Method')
ax1.legend()
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}k'))
ax1.grid(alpha=0.4)

# Silhouette
ax2.bar(ks, silhouettes, color=['#f78166' if k==optimal_k else '#3fb950' for k in ks], width=0.6, alpha=0.85)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score')
ax2.axhline(max(silhouettes), color='#f78166', linestyle='--', lw=1, label=f'Best: {max(silhouettes):.4f}')
ax2.legend()
ax2.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('03_optimal_k.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
"""## 5. Training the Final K-Means Model"""
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

print("Cluster sizes:")
sizes = df_clean['cluster'].value_counts().sort_index()
for c, n in sizes.items():
    print(f"  Cluster {c}: {n:,} houses ({n/len(df_clean):.1%})")

print(f"\nFinal inertia  : {kmeans.inertia_:,.2f}")
print(f"Silhouette score: {silhouette_score(X_scaled, df_clean['cluster']):.4f}")
"""## 6. Cluster Analysis & Interpretation

This is where data science becomes actionable. Let's understand *who* each cluster represents."""
# ── Summary table ────────────────────────────────────────────────────────────
analysis_cols = ['price', 'sqft_living', 'grade', 'bedrooms', 'bathrooms',
                 'yr_built', 'waterfront', 'view', 'lat', 'long']

summary = df_clean.groupby('cluster')[analysis_cols].mean().round(2)
summary.index = [CLUSTER_LABELS[i] for i in summary.index]
summary.columns = ['Avg Price ($)', 'Avg sqft', 'Grade', 'Bedrooms', 'Bathrooms',
                   'Yr Built', 'Waterfront %', 'View Score', 'Lat', 'Long']
summary['Waterfront %'] = (summary['Waterfront %'] * 100).round(1)
summary['Avg Price ($)'] = summary['Avg Price ($)'].map(lambda x: f"${x:,.0f}")

print("=" * 90)
print("  CLUSTER PROFILES")
print("=" * 90)
print(summary.T.to_string())
# ── Price distribution per cluster ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle('Price Distribution by Market Segment', fontsize=14, fontweight='bold')

for i, (ax, color) in enumerate(zip(axes, CLUSTER_COLORS)):
    data = df_clean[df_clean['cluster'] == i]['price'] / 1e6
    ax.hist(data, bins=50, color=color, alpha=0.85, edgecolor='none')
    ax.axvline(data.median(), color='white', lw=1.5, linestyle='--', label=f'Median: ${data.median():.2f}M')
    ax.set_title(CLUSTER_LABELS[i], fontsize=10, fontweight='bold')
    ax.set_xlabel('Price ($ millions)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('04_price_by_cluster.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
# ── Radar / feature comparison ───────────────────────────────────────────────
from matplotlib.patches import FancyArrowPatch

compare_features = ['sqft_living', 'grade', 'bathrooms', 'view', 'waterfront', 'floors']
cluster_means = df_clean.groupby('cluster')[compare_features].mean()

# Normalise 0-1 per feature
cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(compare_features))
width = 0.25

for i, color in enumerate(CLUSTER_COLORS):
    ax.bar(x + i * width, cluster_norm.iloc[i], width, label=CLUSTER_LABELS[i],
           color=color, alpha=0.85, edgecolor='none')

ax.set_xticks(x + width)
ax.set_xticklabels([f.replace('_', ' ').title() for f in compare_features])
ax.set_ylabel('Normalised Score (0 = lowest, 1 = highest)')
ax.set_title('Feature Comparison Across Segments (Normalised)', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.4)
ax.set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig('05_feature_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
# ── Boxplot: price by cluster ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

data_by_cluster = [df_clean[df_clean['cluster']==i]['price'].values / 1e6 for i in range(optimal_k)]
bp = ax.boxplot(data_by_cluster, patch_artist=True, notch=True,
                medianprops=dict(color='white', lw=2),
                whiskerprops=dict(color='#8b949e'),
                capprops=dict(color='#8b949e'),
                flierprops=dict(marker='o', markersize=2, alpha=0.3, color='#8b949e'))

for patch, color in zip(bp['boxes'], CLUSTER_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticklabels([CLUSTER_LABELS[i] for i in range(optimal_k)], fontsize=9)
ax.set_ylabel('Price ($ millions)')
ax.set_title('Price Distribution — Notched Boxplot by Segment', fontweight='bold')
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('06_boxplot_price.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
"""## 7. PCA Visualisation

We use Principal Component Analysis (PCA) to project the 13-dimensional feature space down to 2D,  
allowing us to visually inspect cluster separation."""
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
df_pca['cluster'] = df_clean['cluster'].values
df_pca['price']   = df_clean['price'].values

var1, var2 = pca.explained_variance_ratio_
print(f"PC1 explains {var1:.1%} of variance")
print(f"PC2 explains {var2:.1%} of variance")
print(f"Combined    : {var1+var2:.1%}")

# ── Top contributing features ─────────────────────────────────────────────────
loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=['PC1','PC2'])
print("\nTop 5 features driving PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(5))
fig, ax = plt.subplots(figsize=(11, 8))

sample = df_pca.sample(min(5000, len(df_pca)), random_state=42)

for i, color in enumerate(CLUSTER_COLORS):
    mask = sample['cluster'] == i
    ax.scatter(
        sample.loc[mask, 'PC1'], sample.loc[mask, 'PC2'],
        c=color, alpha=0.45, s=12, label=CLUSTER_LABELS[i], edgecolors='none'
    )

# Cluster centroids in PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
for i, (cx, cy) in enumerate(centroids_pca):
    ax.scatter(cx, cy, c=CLUSTER_COLORS[i], s=250, marker='*', edgecolors='white', lw=1.5, zorder=5)
    ax.annotate(f'C{i}', (cx, cy), textcoords='offset points', xytext=(10, 5),
                fontsize=11, fontweight='bold', color=CLUSTER_COLORS[i])

ax.set_xlabel(f'PC1 ({var1:.1%} variance explained)', fontsize=11)
ax.set_ylabel(f'PC2 ({var2:.1%} variance explained)', fontsize=11)
ax.set_title('PCA — Cluster Separation in 2D Feature Space\n(★ = cluster centroid)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, markerscale=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('07_pca_clusters.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
"""## 8. Geographic Distribution 🗺️

King County has strong spatial patterns. Plotting clusters on a geographic map reveals  
whether the market segments are also spatially concentrated — a key insight for real estate."""
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sample_geo = df_clean.sample(min(8000, len(df_clean)), random_state=42)

# ── Left: coloured by cluster ─────────────────────────────────────────────────
ax = axes[0]
for i, color in enumerate(CLUSTER_COLORS):
    mask = sample_geo['cluster'] == i
    ax.scatter(
        sample_geo.loc[mask, 'long'], sample_geo.loc[mask, 'lat'],
        c=color, s=8, alpha=0.5, label=CLUSTER_LABELS[i], edgecolors='none'
    )
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Cluster Distribution — King County', fontweight='bold')
ax.legend(fontsize=8, markerscale=3)
ax.grid(alpha=0.3)

# ── Right: coloured by price ───────────────────────────────────────────────────
ax = axes[1]
sc = ax.scatter(
    sample_geo['long'], sample_geo['lat'],
    c=sample_geo['price'] / 1e6,
    cmap='plasma', s=8, alpha=0.6, edgecolors='none',
    vmin=0, vmax=2
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label('Price ($ millions)', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='#8b949e')
ax.set_xlabel('Longitude')
ax.set_title('Price Heatmap — Same Geography', fontweight='bold')
ax.grid(alpha=0.3)

fig.suptitle('Geographic Analysis — Seattle Metro Area', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('08_geographic.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("Observation: Cluster 1 (Luxury Waterfront) concentrates near the coast.")
print("Cluster 0 (Premium Modern) spreads across the northern suburbs with newer builds.")
print("Cluster 2 (Entry-Level) dominates the southern and eastern inland areas.")
"""## 9. Silhouette Analysis

A silhouette plot shows how well each individual sample is matched to its own cluster  
versus neighbouring clusters. Scores close to +1 indicate confident assignment; scores near 0 or negative indicate borderline or misclassified samples."""
from sklearn.metrics import silhouette_samples

sil_vals = silhouette_samples(X_scaled, df_clean['cluster'].values)
overall_sil = silhouette_score(X_scaled, df_clean['cluster'].values)

fig, ax = plt.subplots(figsize=(10, 6))

y_lower = 10
for i, color in enumerate(CLUSTER_COLORS):
    cluster_sil = np.sort(sil_vals[df_clean['cluster'].values == i])
    size_i = len(cluster_sil)
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                     alpha=0.7, color=color, label=CLUSTER_LABELS[i])
    ax.text(-0.05, y_lower + 0.5 * size_i, f'C{i}', ha='right', va='center',
            fontsize=10, fontweight='bold', color=color)
    y_lower = y_upper + 10

ax.axvline(overall_sil, color='white', linestyle='--', lw=1.5,
           label=f'Overall: {overall_sil:.4f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_title('Silhouette Plot — Sample-Level Cluster Quality', fontweight='bold')
ax.set_yticks([])
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('09_silhouette.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print(f"Overall silhouette score: {overall_sil:.4f}")
print("Interpretation: Scores above 0.2 are considered meaningful for housing data with high natural variance.")
"""## 10. Conclusions & Business Insights

---

### Market Segments Identified

| Segment | Size | Avg Price | Key Profile |
|---------|------|-----------|-------------|
| **Cluster 0 — Premium Modern** | 8,849 homes (41%) | $660K | Large modern builds, high grade, suburban sprawl |
| **Cluster 1 — Waterfront Luxury** | 143 homes (0.7%) | $1.43M | 100% waterfront, top view scores, premium pricing |
| **Cluster 2 — Entry-Level Classic** | 12,203 homes (57%) | $420K | Smaller, older, lower grade, inland locations |

---

### Key Findings

1. **The market is structurally bimodal** — over 57% of homes are affordable entry-level stock,  
   while a premium segment (41%) commands a 57% price premium on top of that.

2. **Waterfront drives extreme value.** Cluster 1, despite being only 0.7% of the market,  
   commands prices 3.4× higher than entry-level homes. Location clearly dominates all structural features.

3. **Grade and sqft_living are the strongest clustering signals.** The PC1 axis (most variance)  
   is heavily loaded by these two features, confirming that build quality and size are the primary market differentiators.

4. **Geography matters.** Clusters show clear spatial separation across the county —  
   the luxury waterfront homes are concentrated along the shoreline, while entry-level stock fills the inland east.

5. **Temporal mix.** Entry-level homes are older (built ~1957 on average) while premium modern  
   homes average ~1991, suggesting that post-1980 construction correlates with higher quality grades.

---

### Limitations & Future Work

- K-Means assumes spherical, equally-sized clusters — DBSCAN or Gaussian Mixture Models may capture irregular shapes better
- Adding school district ratings, commute times, and crime data could improve cluster interpretability
- A supervised model (e.g. XGBoost) could use these cluster labels as features to improve price prediction
- Time-series analysis could reveal whether segment composition shifts seasonally

---

*This project is part of my CS portfolio. See the [README](README.md) for setup instructions.*"""
