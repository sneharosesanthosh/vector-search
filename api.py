from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from typing import List, Optional
import uuid

app = FastAPI()

# Initialize the model and Qdrant client globally
model = SentenceTransformer('BAAI/bge-base-en')
qdrant_client = QdrantClient("localhost", port=6333)

# Constants
VECTOR_SIZE = 768  # BGE base model dimension

class CollectionCreate(BaseModel):
    collection_name: str
    distance_metric: str = "Cosine"  # Optional, defaults to Cosine

class TextInput(BaseModel):
    collection_name: str  # Add collection name to input
    text: str
    metadata: Optional[dict] = None

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int
    id: str
    metadata: Optional[dict]

class SearchQuery(BaseModel):
    text: str
    collection_name: str
    limit: int = 5

class SearchResult(BaseModel):
    text: str
    score: float

class DeleteQuery(BaseModel):
    collection_name: str
    ids: List[str] = None  # For deleting specific points by their IDs
    filter: dict = None    # For deleting points based on filter conditions

@app.post("/create_collection")
async def create_collection(collection_input: CollectionCreate):
    try:
        distance = models.Distance.COSINE
        if collection_input.distance_metric.lower() == "dot":
            distance = models.Distance.DOT
        elif collection_input.distance_metric.lower() == "euclidean":
            distance = models.Distance.EUCLID
            
        qdrant_client.create_collection(
            collection_name=collection_input.collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=distance
            )
        )
        return {"message": f"Collection {collection_input.collection_name} created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating collection: {str(e)}")

@app.post("/get_embedding")
async def get_embedding(input_data: TextInput) -> EmbeddingResponse:
    # Check if collection exists
    try:
        collection_info = qdrant_client.get_collection(collection_name=input_data.collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/search/", response_model=List[SearchResult])
async def search_text(query: SearchQuery):
    try:
        # Generate embeddings for the search query
        query_vector = model.encode(query.text).tolist()

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=query.collection_name,
            query_vector=query_vector,
            limit=query.limit,
            score_threshold=0.8  # Only return results with similarity > 0.7
        )

        # Format results
        results = []
        for result in search_results:
            results.append(
                SearchResult(
                    text=result.payload.get("text", ""),
                    score=float(result.score)
                )
            )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_points/")
async def delete_points(query: DeleteQuery):
    try:
        if query.ids:
            # Delete specific points by their IDs
            qdrant_client.delete(
                collection_name=query.collection_name,
                points_selector=models.PointIdsList(
                    points=query.ids
                )
            )
        elif query.filter:
            # Delete points based on filter conditions
            qdrant_client.delete(
                collection_name=query.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(**query.filter)
                )
            )
        else:
            # If no ids or filter provided, delete all points in the collection
            qdrant_client.delete(
                collection_name=query.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
        
        return {"message": "Points deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
