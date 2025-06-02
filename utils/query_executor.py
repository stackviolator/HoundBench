#!/usr/bin/env python3
"""
query_executor.py - Neo4j Query Execution and Result Comparison Utility
-----------------------------------------------------------------------
Executes Cypher queries against Neo4j database and provides functionality
to compare query results for evaluation purposes.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from dotenv import load_dotenv

# Rich imports for console output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Neo4j imports
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    Neo4jError = Exception
    NEO4J_AVAILABLE = False

# Initialize Rich Console
console = Console()

# Load environment variables
load_dotenv()

# Neo4j Configuration from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changethispassword")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

@dataclass
class QueryResult:
    """Represents the result of a Cypher query execution."""
    success: bool
    data: List[Dict[str, Any]]
    error: Optional[str]
    execution_time: float
    record_count: int

@dataclass
class ResultComparison:
    """Represents the comparison between two query results."""
    strict_match: bool
    fuzzy_match: bool
    ground_truth_count: int
    generated_count: int
    common_records: int
    missing_records: int
    extra_records: int
    similarity_score: float
    comparison_details: Dict[str, Any]

class QueryExecutor:
    """Neo4j query execution utility."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        """Initialize the query executor with connection parameters."""
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.database = database or NEO4J_DATABASE
        self.driver = None

        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
    
    def connect(self) -> bool:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test the connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1").consume()
            return True
        except Exception as e:
            console.print(f"[red]Failed to connect to Neo4j:[/red] {e}")
            return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def execute_query(self, query: str, show_progress: bool = True) -> QueryResult:
        """
        Execute a Cypher query and return the results.
        
        Args:
            query: The Cypher query to execute
            show_progress: Whether to show progress messages
            
        Returns:
            QueryResult object with execution details
        """
        if not self.driver:
            if not self.connect():
                return QueryResult(
                    success=False,
                    data=[],
                    error="Failed to connect to Neo4j",
                    execution_time=0.0,
                    record_count=0
                )
        
        start_time = time.monotonic()
        
        try:
            with self.driver.session(database=self.database) as session:
                if show_progress:
                    console.print(f"[dim]Executing query against Neo4j...[/dim]")
                
                result = session.run(query)
                records = []
                
                for record in result:
                    # Convert Neo4j record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Convert Neo4j types to serializable types
                        record_dict[key] = self._convert_neo4j_value(value)
                    records.append(record_dict)
                
                execution_time = time.monotonic() - start_time
                
                if show_progress:
                    console.print(f"[green]✓[/green] Query executed successfully, returned {len(records)} records")
                
                return QueryResult(
                    success=True,
                    data=records,
                    error=None,
                    execution_time=execution_time,
                    record_count=len(records)
                )
                
        except Neo4jError as e:
            execution_time = time.monotonic() - start_time
            error_msg = f"Neo4j Error: {e.code} - {e.message}"
            
            if show_progress:
                console.print(f"[red]✗[/red] Query execution failed: {error_msg}")
            
            return QueryResult(
                success=False,
                data=[],
                error=error_msg,
                execution_time=execution_time,
                record_count=0
            )
            
        except Exception as e:
            execution_time = time.monotonic() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            
            if show_progress:
                console.print(f"[red]✗[/red] Query execution failed: {error_msg}")
            
            return QueryResult(
                success=False,
                data=[],
                error=error_msg,
                execution_time=execution_time,
                record_count=0
            )
    
    def _convert_neo4j_value(self, value):
        """Convert Neo4j-specific types to serializable Python types."""
        if hasattr(value, '__dict__'):
            # Handle Neo4j Node, Relationship, Path objects
            if hasattr(value, 'labels') and hasattr(value, 'items'):
                # Neo4j Node
                return {
                    'type': 'node',
                    'labels': list(value.labels),
                    'properties': dict(value.items())
                }
            elif hasattr(value, 'type') and hasattr(value, 'items'):
                # Neo4j Relationship
                return {
                    'type': 'relationship',
                    'rel_type': value.type,
                    'properties': dict(value.items())
                }
            elif hasattr(value, 'nodes') and hasattr(value, 'relationships'):
                # Neo4j Path
                return {
                    'type': 'path',
                    'length': len(value),
                    'nodes': [self._convert_neo4j_value(node) for node in value.nodes],
                    'relationships': [self._convert_neo4j_value(rel) for rel in value.relationships]
                }
        
        # Handle lists and dictionaries recursively
        if isinstance(value, list):
            return [self._convert_neo4j_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._convert_neo4j_value(v) for k, v in value.items()}
        
        # Return primitive types as-is
        return value
    
    def compare_results(self, ground_truth_result: QueryResult, generated_result: QueryResult, 
                       fuzzy_threshold: float = 0.8) -> ResultComparison:
        """
        Compare two query results and return comparison metrics.
        
        Args:
            ground_truth_result: Result from the ground truth query
            generated_result: Result from the generated query
            fuzzy_threshold: Threshold for fuzzy matching (0.0 to 1.0)
            
        Returns:
            ResultComparison object with detailed comparison metrics
        """
        # If either query failed, no meaningful comparison can be made
        if not ground_truth_result.success or not generated_result.success:
            return ResultComparison(
                strict_match=False,
                fuzzy_match=False,
                ground_truth_count=ground_truth_result.record_count,
                generated_count=generated_result.record_count,
                common_records=0,
                missing_records=ground_truth_result.record_count,
                extra_records=generated_result.record_count,
                similarity_score=0.0,
                comparison_details={
                    "ground_truth_error": ground_truth_result.error,
                    "generated_error": generated_result.error
                }
            )
        
        gt_data = ground_truth_result.data
        gen_data = generated_result.data
        
        # Convert records to comparable format (normalized dictionaries)
        gt_normalized = [self._normalize_record(record) for record in gt_data]
        gen_normalized = [self._normalize_record(record) for record in gen_data]
        
        # Find common records using set operations on string representations
        gt_set = set(str(record) for record in gt_normalized)
        gen_set = set(str(record) for record in gen_normalized)
        
        common_set = gt_set.intersection(gen_set)
        missing_set = gt_set - gen_set
        extra_set = gen_set - gt_set
        
        common_count = len(common_set)
        missing_count = len(missing_set)
        extra_count = len(extra_set)
        
        # Calculate similarity score
        if len(gt_set) == 0 and len(gen_set) == 0:
            similarity_score = 1.0  # Both empty
        elif len(gt_set) == 0:
            similarity_score = 0.0  # Ground truth empty, generated has data
        else:
            # Jaccard similarity: intersection / union
            union_size = len(gt_set.union(gen_set))
            similarity_score = len(common_set) / union_size if union_size > 0 else 0.0
        
        # Determine strict and fuzzy matches
        strict_match = (len(gt_set) == len(gen_set) == len(common_set))
        fuzzy_match = similarity_score >= fuzzy_threshold
        
        return ResultComparison(
            strict_match=strict_match,
            fuzzy_match=fuzzy_match,
            ground_truth_count=len(gt_data),
            generated_count=len(gen_data),
            common_records=common_count,
            missing_records=missing_count,
            extra_records=extra_count,
            similarity_score=similarity_score,
            comparison_details={
                "ground_truth_sample": gt_data[:3] if gt_data else [],
                "generated_sample": gen_data[:3] if gen_data else [],
                "common_sample": list(common_set)[:3],
                "missing_sample": list(missing_set)[:3],
                "extra_sample": list(extra_set)[:3]
            }
        )
    
    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a record for comparison by sorting keys and handling nested structures.
        """
        if not isinstance(record, dict):
            return record
        
        normalized = {}
        for key in sorted(record.keys()):
            value = record[key]
            if isinstance(value, dict):
                normalized[key] = self._normalize_record(value)
            elif isinstance(value, list):
                # Sort lists if they contain comparable items
                try:
                    if all(isinstance(item, (str, int, float)) for item in value):
                        normalized[key] = sorted(value)
                    else:
                        normalized[key] = [self._normalize_record(item) if isinstance(item, dict) else item for item in value]
                except TypeError:
                    normalized[key] = value
            else:
                normalized[key] = value
        
        return normalized

def test_connection(uri: str = None, user: str = None, password: str = None, 
                   database: str = None) -> Tuple[bool, str]:
    """
    Test the Neo4j connection and return status.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        executor = QueryExecutor(uri, user, password, database)
        if executor.connect():
            executor.close()
            return True, "Connection successful"
        else:
            return False, "Connection failed"
    except Exception as e:
        return False, f"Connection error: {e}"

if __name__ == "__main__":
    """Test the query executor."""
    console.print(Panel(
        Text("Testing Neo4j Query Executor", style="bold cyan"),
        title="Query Executor Test",
        border_style="blue"
    ))
    
    # Test connection
    success, message = test_connection()
    if success:
        console.print(f"[green]✓[/green] {message}")
        
        # Test query execution
        executor = QueryExecutor()
        if executor.connect():
            test_query = "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            result = executor.execute_query(test_query)
            
            if result.success:
                console.print(Panel(
                    Text(f"Test query successful!\nReturned {result.record_count} records\nExecution time: {result.execution_time:.3f}s", style="green"),
                    title="Query Test Result",
                    border_style="green"
                ))
            else:
                console.print(Panel(
                    Text(f"Test query failed: {result.error}", style="red"),
                    title="Query Test Result",
                    border_style="red"
                ))
            
            executor.close()
    else:
        console.print(f"[red]✗[/red] {message}") 