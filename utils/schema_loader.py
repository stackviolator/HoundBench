#!/usr/bin/env python3
"""
schema_loader.py - Neo4j Schema Extraction Utility
--------------------------------------------------
Dynamically loads the Neo4j database schema including node labels,
relationship types, and properties for inclusion in system prompts.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
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

# Neo4j Configuration from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_SECRET", "bloodhoundcommunityedition")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Load environment variables
load_dotenv()

class SchemaLoader:
    """Neo4j schema extraction utility."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        """Initialize the schema loader with connection parameters."""
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
    
    def get_node_labels(self) -> List[str]:
        """Get all node labels in the database."""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.labels()")
                return [record["label"] for record in result]
        except Exception as e:
            console.print(f"[yellow]Warning: Could not retrieve node labels:[/yellow] {e}")
            return []
    
    def get_relationship_types(self) -> List[str]:
        """Get all relationship types in the database."""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.relationshipTypes()")
                return [record["relationshipType"] for record in result]
        except Exception as e:
            console.print(f"[yellow]Warning: Could not retrieve relationship types:[/yellow] {e}")
            return []
    
    def get_property_keys(self) -> List[str]:
        """Get all property keys in the database."""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.propertyKeys()")
                return [record["propertyKey"] for record in result]
        except Exception as e:
            console.print(f"[yellow]Warning: Could not retrieve property keys:[/yellow] {e}")
            return []
    
    def get_node_properties(self) -> Dict[str, List[str]]:
        """Get properties for each node label."""
        if not self.driver:
            return {}
        
        node_properties = {}
        labels = self.get_node_labels()
        
        for label in labels:
            try:
                with self.driver.session(database=self.database) as session:
                    # Get a sample of nodes with this label to determine properties
                    query = f"MATCH (n:{label}) RETURN keys(n) as properties LIMIT 100"
                    result = session.run(query)
                    
                    all_properties = set()
                    for record in result:
                        all_properties.update(record["properties"])
                    
                    node_properties[label] = sorted(list(all_properties))
            except Exception as e:
                console.print(f"[yellow]Warning: Could not retrieve properties for {label}:[/yellow] {e}")
                node_properties[label] = []
        
        return node_properties
    
    def get_relationship_properties(self) -> Dict[str, List[str]]:
        """Get properties for each relationship type."""
        if not self.driver:
            return {}
        
        rel_properties = {}
        rel_types = self.get_relationship_types()
        
        for rel_type in rel_types:
            try:
                with self.driver.session(database=self.database) as session:
                    # Get a sample of relationships with this type to determine properties
                    query = f"MATCH ()-[r:{rel_type}]-() RETURN keys(r) as properties LIMIT 100"
                    result = session.run(query)
                    
                    all_properties = set()
                    for record in result:
                        all_properties.update(record["properties"])
                    
                    rel_properties[rel_type] = sorted(list(all_properties))
            except Exception as e:
                console.print(f"[yellow]Warning: Could not retrieve properties for {rel_type}:[/yellow] {e}")
                rel_properties[rel_type] = []
        
        return rel_properties
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a comprehensive schema summary."""
        if not self.connect():
            return {}
        
        try:
            schema = {
                "node_labels": self.get_node_labels(),
                "relationship_types": self.get_relationship_types(),
                "property_keys": self.get_property_keys(),
                "node_properties": self.get_node_properties(),
                "relationship_properties": self.get_relationship_properties()
            }
            
            return schema
        finally:
            self.close()
    
    def format_schema_for_prompt(self) -> str:
        """Format the schema information for inclusion in a system prompt."""
        schema = self.get_schema_summary()
        
        if not schema:
            return "Schema information could not be retrieved from the database."
        
        formatted_schema = []
        
        # Node Labels and Properties
        if schema.get("node_labels"):
            formatted_schema.append("## Node Labels and Properties:")
            for label in sorted(schema["node_labels"]):
                properties = schema.get("node_properties", {}).get(label, [])
                if properties:
                    formatted_schema.append(f"- {label}: {', '.join(properties)}")
                else:
                    formatted_schema.append(f"- {label}")
        
        # Relationship Types and Properties
        if schema.get("relationship_types"):
            formatted_schema.append("\n## Relationship Types and Properties:")
            for rel_type in sorted(schema["relationship_types"]):
                properties = schema.get("relationship_properties", {}).get(rel_type, [])
                if properties:
                    formatted_schema.append(f"- {rel_type}: {', '.join(properties)}")
                else:
                    formatted_schema.append(f"- {rel_type}")
        
        # Common Properties
        if schema.get("property_keys"):
            formatted_schema.append(f"\n## All Property Keys:")
            formatted_schema.append(f"{', '.join(sorted(schema['property_keys']))}")
        
        return "\n".join(formatted_schema)

def load_schema(uri: str = None, user: str = None, password: str = None, 
                database: str = None, show_progress: bool = True) -> str:
    """
    Load and format Neo4j schema for system prompt inclusion.
    
    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        database: Neo4j database name
        show_progress: Whether to show progress messages
        
    Returns:
        Formatted schema string for prompt inclusion
    """
    if show_progress:
        console.print("[dim]Loading Neo4j database schema...[/dim]")
    
    try:
        loader = SchemaLoader(uri, user, password, database)
        schema_text = loader.format_schema_for_prompt()
        
        if show_progress:
            if schema_text and "could not be retrieved" not in schema_text:
                console.print("[green]✓[/green] Schema loaded successfully")
            else:
                console.print("[yellow]⚠[/yellow] Schema loading had issues")
        
        return schema_text
        
    except ImportError as e:
        error_msg = "Neo4j driver not available. Schema will not be included."
        if show_progress:
            console.print(f"[yellow]Warning:[/yellow] {error_msg}")
        return f"# {error_msg}"
    
    except Exception as e:
        error_msg = f"Failed to load schema: {e}"
        if show_progress:
            console.print(f"[red]Error:[/red] {error_msg}")
        return f"# Schema loading failed: {e}"

def test_schema_connection(uri: str = None, user: str = None, password: str = None, 
                          database: str = None) -> Tuple[bool, str]:
    """
    Test the Neo4j connection and return status.
    
    Returns:
        Tuple of (success, message)
    """

    print(f"uri: {uri}")
    print(f"user: {user}")
    print(f"password: {password}")
    print(f"database: {database}")
    try:
        loader = SchemaLoader(uri, user, password, database)
        if loader.connect():
            loader.close()
            return True, "Connection successful"
        else:
            return False, "Connection failed"
    except Exception as e:
        return False, f"Connection error: {e}"

if __name__ == "__main__":
    """Test the schema loader."""
    console.print(Panel(
        Text("Testing Neo4j Schema Loader", style="bold cyan"),
        title="Schema Loader Test",
        border_style="blue"
    ))
    
    # Test connection
    success, message = test_schema_connection()
    if success:
        console.print(f"[green]✓[/green] {message}")
        
        # Load and display schema
        schema = load_schema(show_progress=True)
        if schema:
            console.print(Panel(
                Text(schema, style="dim"),
                title="Database Schema",
                border_style="green"
            ))
    else:
        console.print(f"[red]✗[/red] {message}") 