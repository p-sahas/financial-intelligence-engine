
import os
import yaml
import weaviate
import sys

# Load config
try:
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)

# Get connection details
vectordb_cfg = config.get("vector_db", {})
wcs_url = vectordb_cfg.get("wcs_url") or os.environ.get("WEAVIATE_URL")
wcs_api_key = vectordb_cfg.get("wcs_api_key") or os.environ.get("WEAVIATE_API_KEY")
collection_name = vectordb_cfg.get("collection_name", "FinancialReport")

print(f"Checking collection: {collection_name}")
print(f"URL: {wcs_url}")

client = None
try:
    if wcs_url and wcs_api_key:
        print("Connecting to Weaviate Cloud...")
        client = weaviate.connect_to_wcs(
            cluster_url=wcs_url,
            auth_credentials=weaviate.auth.AuthApiKey(wcs_api_key)
        )
    else:
        print("Connecting to Weaviate Local...")
        client = weaviate.connect_to_local()
    
    print(f"Client Ready: {client.is_ready()}")
    
    if client.collections.exists(collection_name):
        print(f"SUCCESS: Collection '{collection_name}' exists.")
        # Optional: Print count
        col = client.collections.get(collection_name)
        print(f"Object count: {col.aggregate.over_all(total_count=True).total_count}")
    else:
        print(f"FAILURE: Collection '{collection_name}' DOES NOT exist.")
        list_cols = client.collections.list_all()
        print(f"Available collections: {list(list_cols.keys())}")

except Exception as e:
    print(f"Connection/Query Error: {e}")
finally:
    if client:
        client.close()
