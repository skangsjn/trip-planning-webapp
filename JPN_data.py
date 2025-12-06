import io
import os
import sqlite3 as sl
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for, send_file
from datetime import datetime
import pytz
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import re
import math
from sklearn.metrics import pairwise_distances
from pathlib import Path

pd.set_option('display.max_columns', None)

# file_path = '/Users/sarahkang/Documents/Projects/Trip-Planner-main/Japan webapp/data/japan_saved_080424.json'
# df0 = pd.read_json(file_path)
#
# file_path = '/Users/sarahkang/Documents/Projects/Trip-Planner-main/Japan webapp/data/additional_080424.json'
# to_add = pd.read_json(file_path)

DATA_DIR = Path(__file__).resolve().parent / "data"
file_path = DATA_DIR / "japan_saved_080424.json"
file_path2 = DATA_DIR / "additional_080424.json"

df0 = pd.read_json(file_path)
to_add = pd.read_json(file_path2)

df = pd.concat([df0, to_add], ignore_index=True)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
      - Remove non-ASCII characters from all string fields
      - Drop rows where 'name' is empty
      - Drop the 'info' column
      - Expand 'gps' dict into 'latitude' / 'longitude'
      - Drop rows with missing latitude or longitude
      - Lowercase the 'type' column
    """
    # 1. Helper to strip non-ASCII chars
    def remove_unicode(text):
        if isinstance(text, str):
            return re.sub(r'[^\x00-\x7F]', '', text)
        return text

    # Work on a copy to avoid mutating the original df
    df_clean = df.copy()

    # 2. Apply unicode cleaning to every element - may need to change to .map() depending on version
    df_clean = df_clean.applymap(remove_unicode)

    # 3. Drop rows with empty 'name'
    df_clean = df_clean[df_clean['name'] != '']

    # 4. Drop 'info' column (ignore if it doesn't exist)
    df_clean = df_clean.drop(columns=['info'], errors='ignore')

    # 5. Expand 'gps' into latitude / longitude
    if 'gps' in df_clean.columns:
        gps_expanded = pd.json_normalize(df_clean['gps'])
        df_clean = df_clean.join(gps_expanded)

        # Rename to latitude / longitude
        df_clean = df_clean.rename(columns={'lat': 'latitude', 'lng': 'longitude'})

        # 6. Remove rows with missing latitude or longitude
        df_clean = df_clean.dropna(subset=['latitude', 'longitude'])

    # 7. Lowercase 'type' column if present
    if 'type' in df_clean.columns:
        df_clean['type'] = df_clean['type'].str.lower()

    return df_clean
df_clean = clean_dataset(df)

def extract_city(plusCode):
    # Handle missing or non-string values
    if pd.isna(plusCode):
        return None
    if not isinstance(plusCode, str):
        plusCode = str(plusCode)

    # Split the string by commas
    parts = plusCode.split(', ')
    # Ensure there are at least two commas to have a city part
    if len(parts) > 2:
        # The city is the second to last element
        return parts[-2]
    elif len(parts) <= 2:
        return re.search(r'\+[^ ]+ (.+)$', parts[0]).group(1)
    return None
# Apply the function to the DataFrame
df_clean['city'] = df_clean['plusCode'].apply(extract_city)

# Check
df_clean['city'].unique()

# 1. Split rows with / without city
mask_known = df_clean['city'].notna()
mask_missing = df_clean['city'].isna()

if mask_missing.any():
    # 2. Training data: lat/lon → city
    X_train = df_clean.loc[mask_known, ['latitude', 'longitude']]
    y_train = df_clean.loc[mask_known, 'city']

    # 3. Data to predict
    X_missing = df_clean.loc[mask_missing, ['latitude', 'longitude']]

    # 4. Fit a simple KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # 5. Predict cities for missing rows
    df_clean.loc[mask_missing, 'city'] = knn.predict(X_missing)

def categorize_type(type_str):
    '''
    Map a raw place 'type' string into a high-level category.
    Categories: Dessert, Food, Thrifting, Shopping, Tourist Attraction, Beauty, None
    Improvement: use unsupervised learning to create clusters?
    '''
    if pd.isna(type_str):
        return None

    s = str(type_str).lower()

    if any(keyword in s for keyword in ['dessert', 'pastry', 'sweets', 'confectionery',
                                               'cafeteria', 'bakery', 'cafe', 'tea house',
                                               'traditional teahouse', 'chocolate shop',
                                               'coffee shop', 'souvenir store']):
        return 'Dessert'
    elif s == 'shop':
        return 'Dessert'
    elif any(keyword in s for keyword in ['restaurant', 'food court', 'stand bar']):
        return 'Food'
    elif any(keyword in s for keyword in ['used clothing', 'vintage clothing', 'thrift store', 'racecourse']):
        return 'Thrifting'
    elif any(keyword in s for keyword in ['mall', 'clothing store', 'video game store', 'business park',
                                                'discount store']):
        return 'Shopping'
    elif any(keyword in s for keyword in ['garden', 'government office', 'tourist attraction',
                                                'observation deck', 'museum', 'tenant ownership']):
        return 'Tourist Attraction'
    elif any(keyword in s for keyword in ['salon']):
        return 'Beauty'
    else:
        return None
df_clean['category'] = df_clean['type'].apply(categorize_type)


def apply_kmeans_balanced(
    df,
    n_clusters=3,
    random_state=0,
    max_capacity_ratio=1.15,
    min_capacity_ratio=0.75
):
    """
    Location-based K-Means with soft upper and lower per-cluster size limits.

    - max_capacity_ratio: how much larger than average a cluster is allowed to be.
    - min_capacity_ratio: how much smaller than average a cluster is allowed to be.
    """
    df = df.copy()

    # --- 1. K-Means on scaled coordinates ---
    coords = df[['latitude', 'longitude']].to_numpy()
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    labels = kmeans.fit_predict(coords_scaled)
    centers = kmeans.cluster_centers_

    N = len(df)
    if N <= n_clusters:
        # Edge case: not enough points, just return raw KMeans
        df.loc[:, 'cluster'] = labels
        return df

    # --- 2. Soft size bounds ---
    avg_size = N / n_clusters
    max_capacity = int(math.ceil(avg_size * max_capacity_ratio))
    min_capacity = max(1, int(math.floor(avg_size * min_capacity_ratio)))
    # ensure min_capacity <= max_capacity
    min_capacity = min(min_capacity, max_capacity)

    # Distances from each point to cluster centers in scaled space
    dists = pairwise_distances(coords_scaled, centers, metric='euclidean')

    # Track membership per cluster
    labels = labels.astype(int)
    cluster_members = {j: np.where(labels == j)[0].tolist() for j in range(n_clusters)}
    cluster_sizes = {j: len(cluster_members[j]) for j in range(n_clusters)}

    # --- 3. Trim oversized clusters (respect max_capacity) ---
    for j in range(n_clusters):
        # While cluster j is over max_capacity, move farthest points out
        while cluster_sizes[j] > max_capacity:
            members = cluster_members[j]
            if not members:
                break

            # Farthest points from center j first
            members_sorted = sorted(
                members,
                key=lambda idx: dists[idx, j],
                reverse=True
            )

            moved_any = False

            for idx in members_sorted:
                if cluster_sizes[j] <= max_capacity:
                    break

                # Try assigning this point to its next closest cluster
                alt_clusters = np.argsort(dists[idx])  # ascending distances

                for k in alt_clusters:
                    if k == j:
                        continue
                    if cluster_sizes[k] < max_capacity:
                        # Move idx from j -> k
                        labels[idx] = k
                        cluster_members[j].remove(idx)
                        cluster_members[k].append(idx)
                        cluster_sizes[j] -= 1
                        cluster_sizes[k] += 1
                        moved_any = True
                        break

                if moved_any:
                    # Recompute membership ordering for j after each move
                    break

            # Safety: if we couldn't move anything else, stop
            if not moved_any:
                break

    # --- 4. Fill undersized clusters (respect min_capacity) ---
    # Move points from larger clusters into tiny clusters until they're at least min_capacity,
    # as long as donors still stay above min_capacity.
    changed = True
    while changed:
        changed = False
        # Clusters that are too small
        underfull = [j for j in range(n_clusters) if cluster_sizes[j] < min_capacity]
        if not underfull:
            break

        for j in underfull:
            if cluster_sizes[j] >= min_capacity:
                continue

            # Donors: clusters that have more than min_capacity points
            donors = [k for k in range(n_clusters)
                      if cluster_sizes[k] > min_capacity and k != j]
            if not donors:
                # No donors with spare points, can't fix this one
                continue

            best_idx = None
            best_from_cluster = None
            best_dist = math.inf

            # Find the best candidate point to move into cluster j
            for k in donors:
                for idx in cluster_members[k]:
                    # Prefer points that are relatively close to cluster j
                    d = dists[idx, j]
                    if d < best_dist:
                        best_dist = d
                        best_idx = idx
                        best_from_cluster = k

            if best_idx is None:
                continue

            # Move the chosen point from donor -> underfull cluster
            labels[best_idx] = j
            cluster_members[best_from_cluster].remove(best_idx)
            cluster_members[j].append(best_idx)
            cluster_sizes[best_from_cluster] -= 1
            cluster_sizes[j] += 1
            changed = True

    # Save back into df
    df.loc[:, 'cluster'] = labels
    return df


# --- Apply per-city with desired cluster counts ---

results = []

for city in df_clean['city'].unique():
    city_df = df_clean[df_clean['city'] == city].copy()

    if city == 'Tokyo':
        n_clusters = 7
    elif city == 'Kyoto':
        n_clusters = 3
    elif city == 'Osaka':
        n_clusters = 3
    else:
        # Skip cities we don't want to cluster (e.g. Nara)
        continue

    clustered_df = apply_kmeans_balanced(city_df, n_clusters=n_clusters, random_state=0)
    results.append(clustered_df)

# Combine results into a single DataFrame
final_df = pd.concat(results, ignore_index=True)
final_df['city_cluster'] = final_df['city'] + final_df['cluster'].astype(str)

# print(final_df.head())
print(final_df.groupby('city_cluster').size().sort_values(ascending=False))
