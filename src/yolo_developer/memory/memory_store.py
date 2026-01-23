"""Enhanced memory store with context management and semantic search capabilities.

This module provides a sophisticated memory storage and retrieval system
using ChromaDB for vector storage and a custom context management approach.
It enables intelligent memory tracking, contextual retrieval, and efficient
token management for AI-powered applications.

Key Functions:
    - store_memory: Store semantic memories with metadata
    - retrieve_memories: Search and retrieve relevant memories
    - get_contextual_memory: Generate refined context for queries
    - get_memory_stats: Retrieve memory store statistics

Example:
    >>> memory_store = MemoryStore()
    >>> memory_id = memory_store.store_memory(
    ...     "Learned Python programming techniques", 
    ...     memory_type="learning"
    ... )
    >>> memories = memory_store.retrieve_memories("Python techniques")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import structlog
from chromadb import Client
from chromadb.config import Settings

from .context_manager import ContextManager

logger = structlog.get_logger(__name__)


class MemoryStore:
    """Enhanced memory store with context refinement capabilities.
    
    Manages semantic memory storage, retrieval, and context management
    using vector database and intelligent context tracking.
    """
    
    def __init__(
        self,
        collection_name: str = "yolo_developer_memory",
        max_context_tokens: int = 4000,
        context_max_age_minutes: int = 60
    ) -> None:
        """Initialize memory store with context management.
        
        Sets up ChromaDB collection and context manager with specified
        configuration for memory storage and context tracking.
        
        Args:
            collection_name: ChromaDB collection name for memory storage.
            max_context_tokens: Maximum tokens allowed in context window.
            context_max_age_minutes: Maximum age for context items before expiration.
        
        Example:
            >>> store = MemoryStore(max_context_tokens=3000)
            >>> store.collection_name
            'yolo_developer_memory'
        """
        self.collection_name = collection_name
        self._client = Client(Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(collection_name)
        
        self.context_manager = ContextManager(
            max_tokens=max_context_tokens,
            max_age_minutes=context_max_age_minutes
        )
    
    def store_memory(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 1
    ) -> str:
        """Store a semantic memory item with contextual metadata.
        
        Stores memory in ChromaDB and adds to context manager for immediate use.
        Generates a unique memory identifier and enriches metadata.
        
        Args:
            content: Semantic content of the memory to store.
            memory_type: Categorization type for the memory (e.g., 'conversation').
            metadata: Optional additional metadata for the memory.
            priority: Contextual priority for memory management.
        
        Returns:
            Unique memory identifier for the stored item.
        
        Raises:
            ValueError: If memory content is empty.
        
        Example:
            >>> store = MemoryStore()
            >>> memory_id = store.store_memory(
            ...     "Learned advanced Python techniques", 
            ...     memory_type="learning", 
            ...     priority=2
            ... )
        """
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
        
        memory_id = f"{memory_type}_{datetime.now().isoformat()}"
        
        # Enhance metadata with system-generated information
        enhanced_metadata = {
            "memory_type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            **(metadata or {})
        }
        
        try:
            self._collection.add(
                documents=[content],
                metadatas=[enhanced_metadata],
                ids=[memory_id]
            )
            
            # Add to context manager for immediate contextual relevance
            self.context_manager.add_context(
                content=content,
                priority=priority,
                tags=[memory_type],
                metadata=enhanced_metadata
            )
            
            logger.info(
                "Stored memory item",
                memory_id=memory_id,
                memory_type=memory_type,
                priority=priority
            )
            
            return memory_id
            
        except Exception as e:
            logger.error("Failed to store memory", error=str(e), memory_id=memory_id)
            raise
    
    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve semantic memories relevant to the given query.
        
        Performs vector similarity search with optional type filtering.
        
        Args:
            query: Semantic search query string.
            memory_types: Optional list of memory types to filter results.
            limit: Maximum number of memory results to return.
        
        Returns:
            List of retrieved memory items with content and metadata.
        
        Example:
            >>> store = MemoryStore()
            >>> memories = store.retrieve_memories(
            ...     "Python programming", 
            ...     memory_types=["learning"]
            ... )
        """
        if not query.strip():
            logger.warning("Empty query provided for memory retrieval")
            return []
        
        try:
            # Construct optional metadata filter for memory types
            where_filter = None
            if memory_types:
                where_filter = {"memory_type": {"$in": memory_types}}
            
            results = self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter
            )
            
            memories = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    memory_item = {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results["distances"] else None
                    }
                    memories.append(memory_item)
            
            logger.info(
                "Retrieved memories",
                query_length=len(query),
                result_count=len(memories)
            )
            
            return memories
            
        except Exception as e:
            logger.error("Failed to retrieve memories", error=str(e), query=query)
            raise
    
    def get_contextual_memory(self, query: str) -> str:
        """Generate refined contextual memory for a specific query.
        
        Dynamically updates context relevance and retrieves additional
        memories to enhance query context.
        
        Args:
            query: Query string to generate contextual memory for.
        
        Returns:
            Refined context string optimized for token usage.
        
        Example:
            >>> store = MemoryStore()
            >>> context = store.get_contextual_memory("Python techniques")
        """
        # Update context relevance based on current query
        self.context_manager.update_relevance_scores(query)
        
        # Retrieve additional relevant memories from storage
        stored_memories = self.retrieve_memories(query, limit=5)
        
        # Integrate high-relevance stored memories into context
        for memory in stored_memories:
            if memory.get("distance", 1.0) < 0.5:  # Only add highly relevant items
                priority = memory["metadata"].get("priority", 1)
                self.context_manager.add_context(
                    content=memory["content"],
                    priority=priority + 1,  # Boost priority for retrieved items
                    metadata=memory["metadata"]
                )
        
        return self.context_manager.get_refined_context()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retrieve current memory store statistics.
        
        Provides insights into stored memories and current context usage.
        
        Returns:
            Dictionary containing memory store statistics including
            stored memory count, context statistics, and collection name.
        
        Example:
            >>> store = MemoryStore()
            >>> stats = store.get_memory_stats()
            >>> print(stats['stored_memories'])
        """
        try:
            collection_count = self._collection.count()
            context_stats = self.context_manager.get_token_usage()
            
            return {
                "stored_memories": collection_count,
                "context_stats": context_stats,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {
                "stored_memories": 0,
                "context_stats": {"error": str(e)},
                "collection_name": self.collection_name
            }
    
    def clear_context(self) -> None:
        """Reset the current context window to its initial state.
        
        Reinitializes context manager with existing configuration,
        effectively clearing all stored context.
        
        Example:
            >>> store = MemoryStore()
            >>> store.clear_context()
        """
        self.context_manager = ContextManager(
            max_tokens=self.context_manager.context_window.max_tokens,
            max_age_minutes=self.context_manager.context_window.max_age_minutes,
            min_priority=self.context_manager.context_window.min_priority
        )
        
        logger.info("Cleared context window")