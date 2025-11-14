"""Example: Recommender system with torch9.rec."""

from torch9 import rec


def recommender_example():
    """Demonstrate recommender system functionality."""
    
    print("=" * 60)
    print("torch9 Recommender System Example")
    print("=" * 60)
    
    # Create recommender model
    print("\n1. Creating Recommender Model")
    print("-" * 40)
    num_users = 1000
    num_items = 500
    embedding_dim = 64
    
    model = rec.RecommenderModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim
    )
    print(f"Model created: {num_users} users, {num_items} items")
    print(f"Embedding dimension: {embedding_dim}")
    
    # Make predictions
    print("\n2. Making Predictions")
    print("-" * 40)
    user_ids = [1, 2, 3, 4, 5]
    item_ids = [10, 20, 30, 40, 50]
    
    scores = model.predict(user_ids, item_ids)
    print(f"User IDs: {user_ids}")
    print(f"Item IDs: {item_ids}")
    print(f"Prediction scores: {scores}")
    
    # Embedding lookup
    print("\n3. Embedding Lookup")
    print("-" * 40)
    user_embeddings = model.user_embeddings([1, 2, 3])
    print(f"Retrieved embeddings for 3 users: shape={user_embeddings.shape}")
    
    item_embeddings = model.item_embeddings([10, 20, 30])
    print(f"Retrieved embeddings for 3 items: shape={item_embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("Recommender example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    recommender_example()
