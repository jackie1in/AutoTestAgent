#!/usr/bin/env python3
"""
清理 Neo4j 数据库中的所有数据。

用法:
    python scripts/clean_neo4j.py
    
环境变量 (从 .env 加载):
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from neo4j import AsyncGraphDatabase


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


async def clean_neo4j():
    """删除 Neo4j 中的所有节点和关系。"""
    # Load environment variables first
    _load_env_file()
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "autotestagent")
    
    print(f"Connecting to Neo4j at {uri}...")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    try:
        async with driver.session() as session:
            # 获取统计信息
            result = await session.run("""
                MATCH (n)
                RETURN count(n) AS node_count
            """)
            record = await result.single()
            node_count = record["node_count"] if record else 0
            
            result = await session.run("""
                MATCH ()-[r]->()
                RETURN count(r) AS rel_count
            """)
            record = await result.single()
            rel_count = record["rel_count"] if record else 0
            
            print(f"Found {node_count} nodes and {rel_count} relationships")
            
            if node_count == 0 and rel_count == 0:
                print("Database is already empty.")
                return
            
            # 确认删除
            confirm = input("\nAre you sure you want to delete ALL data? (yes/no): ")
            if confirm.lower() != "yes":
                print("Aborted.")
                return
            
            # 删除所有数据
            print("Deleting all relationships...")
            await session.run("MATCH ()-[r]->() DELETE r")
            
            print("Deleting all nodes...")
            await session.run("MATCH (n) DELETE n")
            
            # 验证
            result = await session.run("MATCH (n) RETURN count(n) AS count")
            record = await result.single()
            remaining = record["count"] if record else 0
            
            if remaining == 0:
                print("✅ Neo4j database cleaned successfully!")
            else:
                print(f"⚠️  Warning: {remaining} nodes still remain")
                
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(clean_neo4j())
