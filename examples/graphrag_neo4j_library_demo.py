#!/usr/bin/env python3
"""
Neo4j GraphRAG Library Demo

展示如何使用 neo4j-graphrag 库进行 GraphRAG 操作。

安装依赖:
    uv pip install neo4j-graphrag

运行前准备:
    docker compose up -d neo4j
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from retriever import (
    GraphRAGQueryEngine,
    GraphRAGIndexManager,
    IntentBasedRetriever,
)
from retriever.embedding import OpenAIEmbeddingProvider
from graph_agent.neo4j.driver import Neo4jDriver


def _load_env_file() -> None:
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)


async def demo_basic_vector_search():
    """演示基本的向量搜索。"""
    print("=== Demo 1: Basic Vector Search ===\n")
    
    # 连接 Neo4j
    driver = Neo4jDriver()
    await driver.connect()
    
    # 创建 embedding provider
    embedding = OpenAIEmbeddingProvider()
    
    # 创建查询引擎
    engine = GraphRAGQueryEngine(driver.driver, embedding)
    
    # 示例：搜索意图
    query = "如何登录系统"
    print(f"Query: {query}")
    
    try:
        results = await engine.vector_search(
            query_text=query,
            index_name="intent_embeddings",
            top_k=5
        )
        
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"  - {result['text'][:100]}... (score: {result['score']:.3f})")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure vector index exists. Run tests first to create embeddings.")
    finally:
        await driver.close()


async def demo_index_manager():
    """演示索引管理。"""
    print("\n=== Demo 2: Index Management ===\n")
    
    driver = Neo4jDriver()
    await driver.connect()
    
    try:
        # 创建索引管理器
        manager = GraphRAGIndexManager(driver.driver)
        
        # 列出现有索引
        indexes = await manager.list_indexes()
        print(f"Found {len(indexes)} vector indexes:")
        for idx in indexes:
            print(f"  - {idx['name']} on {idx['label']}.{idx['property']}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await driver.close()


async def demo_hybrid_search():
    """演示混合搜索（向量 + 全文）。"""
    print("\n=== Demo 3: Hybrid Search ===\n")
    
    driver = Neo4jDriver()
    await driver.connect()
    
    embedding = OpenAIEmbeddingProvider()
    
    try:
        # 创建意图检索器（使用混合搜索）
        retriever = IntentBasedRetriever(
            driver=driver.driver,
            embedding_provider=embedding
        )
        
        # 混合搜索
        query = "登录页面"
        print(f"Query: {query}")
        
        results = await retriever.retrieve(
            query_text=query,
            intent_types=["navigation", "auth"],
            top_k=5
        )
        
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  - {r.node_id}: {r.text[:80]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Hybrid search requires both vector and fulltext indexes.")
    finally:
        await driver.close()


async def main():
    """运行所有演示。"""
    # Load environment variables first
    _load_env_file()
    
    print("Neo4j GraphRAG Library Demo")
    print("=" * 50)
    
    await demo_basic_vector_search()
    await demo_index_manager()
    await demo_hybrid_search()
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
