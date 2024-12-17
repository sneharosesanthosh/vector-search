from sentence_transformers import SentenceTransformer
import torch

def get_embeddings(texts, model_name="BAAI/bge-base-en", batch_size=32):
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # Normalize embeddings to unit length
        show_progress_bar=True
    )
    
    return embeddings

# Example usage
if __name__ == "__main__":
    # Test texts
    texts = [
        "Hello, this is a test sentence.",
        "Another example sentence for embedding.",
        "The BGE model is creating vector embeddings."
    ]
    
    # Get embeddings
    embeddings = get_embeddings(texts)
    
    # Print results
    print(f"Number of texts: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding vector: {embeddings[0][:5]}...")  # Show first 5 dimensions
    
    # Verify embeddings are normalized (length should be close to 1.0)
    print(f"Embedding vector length: {torch.norm(torch.tensor(embeddings[0])):.6f}")