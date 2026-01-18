"""
OpenSearch Serverless Setup Script for Political Strategy Maker.

This script:
1. Creates an OpenSearch Serverless Collection
2. Creates the required access policies (encryption, network, data)
3. Waits for the collection to be active
4. Creates the political-data index
5. Returns the endpoint URL for configuration

Usage:
    python scripts/setup_opensearch.py

Requirements:
    - AWS CLI configured with appropriate permissions
    - boto3 installed
"""
import os
import sys
import json
import time
import boto3
from pathlib import Path

# Configuration
COLLECTION_NAME = "political-strategy"
INDEX_NAME = "political-data"
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Embedding dimension for text-embedding-3-large
EMBEDDING_DIM = 3072


def create_encryption_policy(client, collection_name: str) -> bool:
    """Create encryption policy for the collection."""
    policy_name = f"{collection_name}-encryption"
    policy = {
        "Rules": [
            {
                "ResourceType": "collection",
                "Resource": [f"collection/{collection_name}"]
            }
        ],
        "AWSOwnedKey": True
    }
    
    try:
        # Check if policy exists
        try:
            client.get_security_policy(name=policy_name, type="encryption")
            print(f"  [OK] Encryption policy '{policy_name}' already exists")
            return True
        except client.exceptions.ResourceNotFoundException:
            pass
        
        # Create policy
        client.create_security_policy(
            name=policy_name,
            type="encryption",
            policy=json.dumps(policy),
            description="Encryption policy for political strategy maker"
        )
        print(f"  [OK] Created encryption policy: {policy_name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create encryption policy: {e}")
        return False


def create_network_policy(client, collection_name: str) -> bool:
    """Create network policy for public access."""
    policy_name = f"{collection_name}-network"
    policy = [
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"]
                },
                {
                    "ResourceType": "dashboard",
                    "Resource": [f"collection/{collection_name}"]
                }
            ],
            "AllowFromPublic": True
        }
    ]
    
    try:
        # Check if policy exists
        try:
            client.get_security_policy(name=policy_name, type="network")
            print(f"  [OK] Network policy '{policy_name}' already exists")
            return True
        except client.exceptions.ResourceNotFoundException:
            pass
        
        # Create policy
        client.create_security_policy(
            name=policy_name,
            type="network",
            policy=json.dumps(policy),
            description="Network policy for political strategy maker"
        )
        print(f"  [OK] Created network policy: {policy_name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create network policy: {e}")
        return False


def create_data_access_policy(client, collection_name: str, principal_arn: str) -> bool:
    """Create data access policy."""
    policy_name = f"{collection_name}-data-access"
    policy = [
        {
            "Rules": [
                {
                    "ResourceType": "index",
                    "Resource": [f"index/{collection_name}/*"],
                    "Permission": [
                        "aoss:CreateIndex",
                        "aoss:DeleteIndex",
                        "aoss:UpdateIndex",
                        "aoss:DescribeIndex",
                        "aoss:ReadDocument",
                        "aoss:WriteDocument"
                    ]
                },
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"],
                    "Permission": [
                        "aoss:CreateCollectionItems",
                        "aoss:DeleteCollectionItems",
                        "aoss:UpdateCollectionItems",
                        "aoss:DescribeCollectionItems"
                    ]
                }
            ],
            "Principal": [principal_arn],
            "Description": "Data access for Lambda and admin users"
        }
    ]
    
    try:
        # Check if policy exists
        try:
            client.get_access_policy(name=policy_name, type="data")
            print(f"  [OK] Data access policy '{policy_name}' already exists")
            return True
        except client.exceptions.ResourceNotFoundException:
            pass
        
        # Create policy
        client.create_access_policy(
            name=policy_name,
            type="data",
            policy=json.dumps(policy),
            description="Data access policy for political strategy maker"
        )
        print(f"  [OK] Created data access policy: {policy_name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create data access policy: {e}")
        return False


def create_collection(client, collection_name: str) -> dict:
    """Create OpenSearch Serverless collection."""
    try:
        # Check if collection exists
        response = client.batch_get_collection(names=[collection_name])
        if response.get("collectionDetails"):
            collection = response["collectionDetails"][0]
            print(f"  [OK] Collection '{collection_name}' already exists")
            return collection
    except Exception:
        pass
    
    # Create collection
    try:
        response = client.create_collection(
            name=collection_name,
            type="VECTORSEARCH",
            description="Political Strategy Maker vector search collection"
        )
        print(f"  [OK] Created collection: {collection_name}")
        return response.get("createCollectionDetail", {})
    except Exception as e:
        print(f"  [ERROR] Failed to create collection: {e}")
        return {}


def wait_for_collection(client, collection_name: str, timeout: int = 600) -> str:
    """Wait for collection to become active."""
    print(f"  Waiting for collection to become active (timeout: {timeout}s)...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = client.batch_get_collection(names=[collection_name])
            if response.get("collectionDetails"):
                collection = response["collectionDetails"][0]
                status = collection.get("status")
                endpoint = collection.get("collectionEndpoint")
                
                if status == "ACTIVE":
                    print(f"  [OK] Collection is ACTIVE!")
                    return endpoint
                elif status == "FAILED":
                    print(f"  [ERROR] Collection creation failed")
                    return None
                else:
                    print(f"    Status: {status}...")
        except Exception as e:
            print(f"    Error checking status: {e}")
        
        time.sleep(15)
    
    print(f"  [ERROR] Timeout waiting for collection")
    return None


def create_index(endpoint: str, index_name: str, region: str) -> bool:
    """Create the vector search index."""
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth
    
    # Get AWS credentials
    session = boto3.Session()
    credentials = session.get_credentials()
    
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        'aoss',
        session_token=credentials.token
    )
    
    # Extract host from endpoint
    host = endpoint.replace("https://", "")
    
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60
    )
    
    # Index mapping for political data
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "source_path": {"type": "keyword"},
                "source_file": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 16
                        }
                    }
                },
                "metadata": {"type": "object", "enabled": True},
                "entity_type": {"type": "keyword"},
                "entity_name": {"type": "keyword"},
                "constituency": {"type": "keyword"},
                "district": {"type": "keyword"},
                "party": {"type": "keyword"},
                "year": {"type": "keyword"},
                "data_type": {"type": "keyword"},
                "winner_2021": {"type": "keyword"},
                "predicted_winner_2026": {"type": "keyword"},
                "race_rating": {"type": "keyword"},
                "margin_2021": {"type": "float"},
                "predicted_margin_2026": {"type": "float"},
                "tmc_vote_share": {"type": "float"},
                "bjp_vote_share": {"type": "float"},
                "swing": {"type": "float"},
                "timestamp": {"type": "date"}
            }
        }
    }
    
    try:
        # Check if index exists
        if client.indices.exists(index=index_name):
            print(f"  [OK] Index '{index_name}' already exists")
            return True
        
        # Create index
        client.indices.create(index=index_name, body=index_body)
        print(f"  [OK] Created index: {index_name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create index: {e}")
        return False


def get_current_user_arn() -> str:
    """Get the ARN of the current AWS user/role."""
    sts = boto3.client("sts", region_name=REGION)
    identity = sts.get_caller_identity()
    return identity["Arn"]


def update_env_file(endpoint: str):
    """Update the .env file with the OpenSearch endpoint."""
    env_path = Path(__file__).parent.parent / "backend" / ".env"
    
    if env_path.exists():
        with open(env_path, "r") as f:
            content = f.read()
        
        # Check if OPENSEARCH_ENDPOINT already exists
        if "OPENSEARCH_ENDPOINT" in content:
            # Update existing
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                if line.startswith("OPENSEARCH_ENDPOINT"):
                    new_lines.append(f"OPENSEARCH_ENDPOINT={endpoint}")
                else:
                    new_lines.append(line)
            content = "\n".join(new_lines)
        else:
            # Add new
            content += f"\n\n# OpenSearch Serverless\nOPENSEARCH_ENDPOINT={endpoint}\n"
        
        with open(env_path, "w") as f:
            f.write(content)
        print(f"  [OK] Updated {env_path}")
    else:
        print(f"  [WARN] .env file not found at {env_path}")
        print(f"  Add this to your .env: OPENSEARCH_ENDPOINT={endpoint}")


def main():
    print("=" * 60)
    print("Political Strategy Maker - OpenSearch Serverless Setup")
    print("=" * 60)
    
    print(f"\n[CONFIG]")
    print(f"  Collection Name: {COLLECTION_NAME}")
    print(f"  Index Name: {INDEX_NAME}")
    print(f"  Region: {REGION}")
    print(f"  Embedding Dimension: {EMBEDDING_DIM}")
    
    # Get current user ARN
    print(f"\n[STEP 1] Getting AWS identity...")
    try:
        user_arn = get_current_user_arn()
        print(f"  [OK] Current identity: {user_arn}")
    except Exception as e:
        print(f"  [ERROR] Failed to get AWS identity: {e}")
        return 1
    
    # Create OpenSearch Serverless client
    client = boto3.client("opensearchserverless", region_name=REGION)
    
    # Create policies
    print(f"\n[STEP 2] Creating security policies...")
    
    if not create_encryption_policy(client, COLLECTION_NAME):
        return 1
    
    if not create_network_policy(client, COLLECTION_NAME):
        return 1
    
    if not create_data_access_policy(client, COLLECTION_NAME, user_arn):
        return 1
    
    # Create collection
    print(f"\n[STEP 3] Creating collection...")
    collection = create_collection(client, COLLECTION_NAME)
    if not collection:
        return 1
    
    # Wait for collection to be active
    print(f"\n[STEP 4] Waiting for collection...")
    endpoint = wait_for_collection(client, COLLECTION_NAME)
    if not endpoint:
        return 1
    
    print(f"\n  Endpoint: {endpoint}")
    
    # Create index
    print(f"\n[STEP 5] Creating index...")
    # Wait a bit for collection to be fully ready
    time.sleep(30)
    
    if not create_index(endpoint, INDEX_NAME, REGION):
        print("  [WARN] Index creation may need to be retried")
        print("  Run: python scripts/ingest_opensearch.py after a few minutes")
    
    # Update .env file
    print(f"\n[STEP 6] Updating configuration...")
    update_env_file(endpoint)
    
    # Summary
    print("\n" + "=" * 60)
    print("OpenSearch Serverless Setup Complete!")
    print("=" * 60)
    print(f"\n  Collection: {COLLECTION_NAME}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Index: {INDEX_NAME}")
    print(f"\n  Next Steps:")
    print(f"  1. Add to Lambda environment: OPENSEARCH_ENDPOINT={endpoint}")
    print(f"  2. Run data ingestion: python scripts/ingest_opensearch.py")
    print(f"  3. Redeploy: sam build && sam deploy")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

