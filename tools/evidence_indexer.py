"""
Evidence indexing tool for semantic similarity search of resume content.

This tool implements vector-based evidence indexing using sentence-transformers
and ChromaDB for finding resume bullets and skills that support generated claims.
It provides the core evidence validation capability for the apply.ai system.

This tool is designed to be used by validator agents within the LangGraph workflow
to ensure all generated content has proper evidence backing from the resume.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer
import numpy as np

from src.schemas.core import Resume, ResumeBullet


logger = logging.getLogger(__name__)


class EvidenceIndexingError(Exception):
    """Raised when evidence indexing encounters unrecoverable errors."""

    pass


class EvidenceMatch:
    """Represents a single evidence match with metadata."""

    def __init__(
        self,
        bullet: ResumeBullet,
        similarity_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.bullet = bullet
        self.similarity_score = similarity_score
        self.metadata = metadata or {}

    def __repr__(self):
        return f"EvidenceMatch(score={self.similarity_score:.3f}, text='{self.bullet.text[:50]}...')"


class EvidenceIndexer:
    """
    Pure tool for indexing and searching resume evidence using vector embeddings.

    This tool provides stateless, deterministic indexing of resume bullets and skills
    using sentence-transformers for embedding generation and ChromaDB for vector storage.
    It supports semantic similarity search for evidence validation.

    Features:
    - SBERT-based embedding generation with configurable models
    - ChromaDB integration with persistent collections
    - Text preprocessing and normalization
    - Configurable similarity thresholds and top-k results
    - Batch operations for efficient indexing
    - Comprehensive metadata tracking for evidence provenance
    - Error handling and recovery mechanisms
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "resume_evidence",
        persist_directory: Optional[str] = None,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize the evidence indexer with embedding model and vector store.

        Args:
            model_name: Sentence-transformers model name
            collection_name: ChromaDB collection name
            persist_directory: Optional directory for persistent storage
            similarity_threshold: Minimum similarity score for evidence matches
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold

        # Initialize sentence transformer model
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dimension = (
                self.embedding_model.get_sentence_embedding_dimension()
            )
            logger.info(
                f"Loaded embedding model: {model_name} (dim={self.embedding_dimension})"
            )
        except Exception as e:
            raise EvidenceIndexingError(
                f"Failed to load embedding model {model_name}: {str(e)}"
            ) from e

        # Initialize ChromaDB client
        try:
            if persist_directory:
                persist_path = Path(persist_directory)
                persist_path.mkdir(parents=True, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=Settings(anonymized_telemetry=False),
                )
                logger.info(f"Using persistent ChromaDB storage: {persist_path}")
            else:
                self.chroma_client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info("Using ephemeral ChromaDB storage")

            # Get or create collection
            self.collection = self._get_or_create_collection()

        except Exception as e:
            raise EvidenceIndexingError(
                f"Failed to initialize ChromaDB: {str(e)}"
            ) from e

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one with proper metadata schema."""
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
            return collection

        except Exception:
            # Create new collection with metadata schema
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Resume evidence vectors for semantic similarity search",
                    "embedding_model": self.model_name,
                    "embedding_dimension": self.embedding_dimension,
                    "created_by": "EvidenceIndexer",
                },
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for consistent embedding generation.

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove bullet point indicators for consistent comparison
        bullet_patterns = [
            r"^\s*[•·▪▫◦‣⁃]\s*",  # Unicode bullets
            r"^\s*[-*+]\s*",  # ASCII bullets
            r"^\s*\d+[\.\)]\s*",  # Numbered lists
            r"^\s*[a-zA-Z][\.\)]\s*",  # Lettered lists
        ]

        for pattern in bullet_patterns:
            text = re.sub(pattern, "", text)

        # Remove common resume artifacts
        text = re.sub(r"\b(page \d+|\d+ of \d+)\b", "", text, flags=re.IGNORECASE)

        return text.strip()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings

        Raises:
            EvidenceIndexingError: If embedding generation fails
        """
        if not texts:
            return np.array([])

        try:
            # Preprocess texts
            preprocessed_texts = [self.preprocess_text(text) for text in texts]

            # Filter empty texts
            valid_texts = [text for text in preprocessed_texts if text]
            if not valid_texts:
                logger.warning("No valid texts to embed after preprocessing")
                return np.array([])

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                valid_texts,
                batch_size=32,
                show_progress_bar=len(valid_texts) > 100,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )

            logger.info(f"Generated embeddings for {len(valid_texts)} texts")
            return embeddings

        except Exception as e:
            raise EvidenceIndexingError(
                f"Failed to generate embeddings: {str(e)}"
            ) from e

    def index_resume(
        self, resume: Resume, resume_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index all bullets and skills from a resume for evidence search.

        Args:
            resume: Resume object with bullets and skills
            resume_id: Optional unique identifier for this resume

        Returns:
            Dictionary with indexing statistics

        Raises:
            EvidenceIndexingError: If indexing fails
        """
        if not resume_id:
            resume_id = str(uuid4())[:8]

        try:
            # Prepare all content for indexing
            indexing_items = []

            # Add bullets
            for i, bullet in enumerate(resume.bullets):
                item = {
                    "id": f"bullet_{resume_id}_{i}",
                    "text": bullet.text,
                    "type": "bullet",
                    "section": bullet.section,
                    "start_offset": bullet.start_offset,
                    "end_offset": bullet.end_offset,
                    "resume_id": resume_id,
                }
                indexing_items.append(item)

            # Add skills as individual items
            for i, skill in enumerate(resume.skills):
                item = {
                    "id": f"skill_{resume_id}_{i}",
                    "text": skill,
                    "type": "skill",
                    "section": "skills",
                    "start_offset": -1,  # Skills don't have specific offsets
                    "end_offset": -1,
                    "resume_id": resume_id,
                }
                indexing_items.append(item)

            if not indexing_items:
                logger.warning(f"No content to index for resume {resume_id}")
                return {
                    "items_indexed": 0,
                    "bullets_indexed": 0,
                    "skills_indexed": 0,
                    "resume_id": resume_id,
                    "collection_size": self.collection.count(),
                }

            # Generate embeddings
            texts = [item["text"] for item in indexing_items]
            embeddings = self.generate_embeddings(texts)

            if len(embeddings) == 0:
                logger.warning(f"No embeddings generated for resume {resume_id}")
                return {
                    "items_indexed": 0,
                    "bullets_indexed": 0,
                    "skills_indexed": 0,
                    "resume_id": resume_id,
                    "collection_size": self.collection.count(),
                }

            # Prepare data for ChromaDB
            ids = [item["id"] for item in indexing_items[: len(embeddings)]]
            documents = [item["text"] for item in indexing_items[: len(embeddings)]]
            metadatas = [
                {
                    "type": item["type"],
                    "section": item["section"],
                    "start_offset": item["start_offset"],
                    "end_offset": item["end_offset"],
                    "resume_id": item["resume_id"],
                }
                for item in indexing_items[: len(embeddings)]
            ]

            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )

            # Update embeddings in resume bullets for consistency
            bullet_embeddings_count = 0
            for i, bullet in enumerate(resume.bullets):
                if i < len(embeddings):
                    bullet.embedding = embeddings[i].tolist()
                    bullet_embeddings_count += 1

            indexing_stats = {
                "items_indexed": len(embeddings),
                "bullets_indexed": bullet_embeddings_count,
                "skills_indexed": len(resume.skills),
                "resume_id": resume_id,
                "collection_size": self.collection.count(),
            }

            logger.info(f"Indexed resume {resume_id}: {indexing_stats}")
            return indexing_stats

        except Exception as e:
            raise EvidenceIndexingError(
                f"Failed to index resume {resume_id}: {str(e)}"
            ) from e

    def find_evidence(
        self,
        claim_text: str,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceMatch]:
        """
        Find the most similar resume evidence for a given claim.

        Args:
            claim_text: Text claim to find evidence for
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold (defaults to class threshold)
            filter_metadata: Optional metadata filters for search

        Returns:
            List of EvidenceMatch objects sorted by similarity score

        Raises:
            EvidenceIndexingError: If search fails
        """
        if not claim_text.strip():
            return []

        # Use class threshold if not specified
        if min_similarity is None:
            min_similarity = self.similarity_threshold

        try:
            # Generate embedding for claim
            claim_embedding = self.generate_embeddings([claim_text])
            if len(claim_embedding) == 0:
                logger.warning(f"Could not generate embedding for claim: {claim_text}")
                return []

            # Prepare ChromaDB query
            query_params = {
                "query_embeddings": claim_embedding.tolist(),
                "n_results": min(top_k * 2, 50),  # Get extra results for filtering
                "include": ["distances", "documents", "metadatas"],
            }

            if filter_metadata:
                query_params["where"] = filter_metadata

            # Execute similarity search
            results = self.collection.query(**query_params)

            if not results["ids"] or not results["ids"][0]:
                logger.info("No evidence found for claim")
                return []

            # Process results into EvidenceMatch objects
            evidence_matches = []

            for i, (doc_id, distance, document, metadata) in enumerate(
                zip(
                    results["ids"][0],
                    results["distances"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                )
            ):
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                # For normalized embeddings, cosine similarity = 1 - (L2_distance^2 / 2)
                similarity_score = max(0.0, 1.0 - (distance**2) / 2)

                # Filter by minimum similarity
                if similarity_score < min_similarity:
                    continue

                # Create ResumeBullet object from stored metadata
                bullet = ResumeBullet(
                    text=document,
                    section=metadata.get("section", "unknown"),
                    start_offset=metadata.get("start_offset", 0),
                    end_offset=metadata.get("end_offset", 0),
                )

                # Create evidence match
                match = EvidenceMatch(
                    bullet=bullet,
                    similarity_score=similarity_score,
                    metadata={
                        "id": doc_id,
                        "type": metadata.get("type", "unknown"),
                        "resume_id": metadata.get("resume_id", "unknown"),
                        "raw_distance": distance,
                    },
                )

                evidence_matches.append(match)

            # Sort by similarity score (descending) and limit to top_k
            evidence_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            evidence_matches = evidence_matches[:top_k]

            logger.info(
                f"Found {len(evidence_matches)} evidence matches for claim "
                f"(min_similarity={min_similarity:.3f})"
            )

            return evidence_matches

        except Exception as e:
            raise EvidenceIndexingError(
                f"Failed to search for evidence: {str(e)}"
            ) from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current evidence collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()

            # Get sample of metadata to analyze distribution
            if count > 0:
                sample_size = min(100, count)
                sample_results = self.collection.get(
                    limit=sample_size, include=["metadatas"]
                )

                # Analyze types and sections
                types_count = {}
                sections_count = {}
                resume_ids = set()

                if sample_results["metadatas"]:
                    for metadata in sample_results["metadatas"]:
                        item_type = metadata.get("type", "unknown")
                        section = metadata.get("section", "unknown")
                        resume_id = metadata.get("resume_id", "unknown")

                        types_count[item_type] = types_count.get(item_type, 0) + 1
                        sections_count[section] = sections_count.get(section, 0) + 1
                        resume_ids.add(resume_id)

                stats = {
                    "total_items": count,
                    "unique_resumes": len(resume_ids),
                    "types_distribution": types_count,
                    "sections_distribution": sections_count,
                    "embedding_model": self.model_name,
                    "collection_name": self.collection_name,
                    "similarity_threshold": self.similarity_threshold,
                }
            else:
                stats = {
                    "total_items": 0,
                    "unique_resumes": 0,
                    "types_distribution": {},
                    "sections_distribution": {},
                    "embedding_model": self.model_name,
                    "collection_name": self.collection_name,
                    "similarity_threshold": self.similarity_threshold,
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """
        Clear all data from the evidence collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection
            self.chroma_client.delete_collection(name=self.collection_name)

            # Recreate empty collection
            self.collection = self._get_or_create_collection()

            logger.info(f"Cleared collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False

    def delete_resume(self, resume_id: str) -> Dict[str, Any]:
        """
        Delete all evidence entries for a specific resume.

        Args:
            resume_id: Resume identifier to delete

        Returns:
            Dictionary with deletion statistics
        """
        try:
            # Query for all items with this resume_id
            results = self.collection.get(
                where={"resume_id": resume_id}, include=["metadatas"]
            )

            if not results["ids"]:
                logger.info(f"No evidence found for resume {resume_id}")
                return {"deleted_count": 0, "resume_id": resume_id}

            # Delete all matching items
            self.collection.delete(ids=results["ids"])

            deleted_count = len(results["ids"])
            logger.info(
                f"Deleted {deleted_count} evidence items for resume {resume_id}"
            )

            return {
                "deleted_count": deleted_count,
                "resume_id": resume_id,
                "remaining_items": self.collection.count(),
            }

        except Exception as e:
            logger.error(f"Failed to delete resume {resume_id}: {str(e)}")
            return {"error": str(e), "resume_id": resume_id}


# Convenience function for simple evidence search
def find_evidence(
    claim_text: str,
    indexer: Optional[EvidenceIndexer] = None,
    top_k: int = 5,
    similarity_threshold: float = 0.8,
) -> List[EvidenceMatch]:
    """
    Convenience function for finding evidence with default indexer.

    Args:
        claim_text: Text claim to find evidence for
        indexer: Optional EvidenceIndexer instance (creates ephemeral if None)
        top_k: Maximum number of results to return
        similarity_threshold: Minimum similarity score threshold

    Returns:
        List of EvidenceMatch objects
    """
    if indexer is None:
        indexer = EvidenceIndexer(similarity_threshold=similarity_threshold)

    return indexer.find_evidence(
        claim_text=claim_text, top_k=top_k, min_similarity=similarity_threshold
    )
