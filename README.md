# HoundBench - Cypher Query Evaluation & Validation

A comprehensive toolkit for evaluating and validating Cypher queries against BloodHound instances, featuring both offline syntax validation and live query execution testing.

## Features

### 🔍 **Cypher Query Validation**
- **Offline Syntax Validation**: Fast ANTLR-based syntax checking without database connection
- **Online Semantic Validation**: Neo4j EXPLAIN-based validation for semantic correctness
- **Pre-execution Validation**: Automatic syntax checking before running queries against BloodHound
- **Rich Console Output**: Beautiful, color-coded validation results with detailed error reporting

### 📊 **Query Execution & Evaluation**
- **BloodHound API Integration**: Execute queries against live BloodHound instances
- **Comprehensive Statistics**: Detailed performance metrics and success/failure rates
- **Error Classification**: Distinguish between syntax errors, execution errors, and empty results
- **Rate Limiting**: Configurable delays between queries to prevent API overload

### 🤖 **LLM Query Generation**
- **AI-Powered Query Creation**: Generate Cypher queries from natural language descriptions using LLMs
- **Multiple LLM Providers**: Support for OpenAI GPT models (extensible to other providers)
- **Automatic Validation**: Built-in syntax validation of generated queries
- **Evaluation Metrics**: Track generation success rates and syntax validity
- **Batch Processing**: Generate queries for entire description datasets

### 📁 **Data Management**
- **Query Dataset**: Curated collection of BloodHound Cypher queries with descriptions
- **Failed Query Tracking**: Automatic saving of failed queries with error details
- **Timestamped Reports**: Organized output files for analysis and debugging

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd HoundBench
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   # BloodHound API Configuration
   BHE_DOMAIN=your-bloodhound-domain.com
   BHE_PORT=443
   BHE_SCHEME=https
   BHE_TOKEN_ID=your-token-id
   BHE_TOKEN_KEY=your-token-key
   
   # LLM API Configuration (for query generation)
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_MODEL=gpt-4
   OPENAI_BASE_URL=https://api.openai.com/v1
   
   # Optional Configuration
   QUERY_DELAY_SECONDS=0.5
   ENABLE_PRE_VALIDATION=true
   REQUEST_DELAY_SECONDS=1.0
   ```

## Usage

### 🚀 **Quick Start**

**Validate all queries (syntax only)**:
```bash
python evals/validate_cypher_syntax.py
```

**Run full evaluation against BloodHound**:
```bash
python evals/make_bh_queries.py
```

**Generate queries using LLM**:
```bash
python evals/llm_query_generation.py
```

### 📋 **Detailed Usage**

#### **Syntax Validation Only**

```bash
# Validate entire dataset with summary
python evals/validate_cypher_syntax.py

# Validate with detailed output for each query
python evals/validate_cypher_syntax.py --verbose

# Validate a single query
python evals/validate_cypher_syntax.py --query "MATCH (n:User) RETURN n"

# Validate custom queries file
python evals/validate_cypher_syntax.py --file /path/to/custom_queries.json

# Skip saving failed queries to file
python evals/validate_cypher_syntax.py --no-save
```

#### **Full Query Evaluation**

The main evaluator (`make_bh_queries.py`) automatically includes syntax pre-validation when `ENABLE_PRE_VALIDATION=true` in your `.env` file.

**Features**:
- ✅ Pre-execution syntax validation (optional)
- ✅ Live query execution against BloodHound
- ✅ Comprehensive error handling and classification
- ✅ Performance metrics and timing statistics
- ✅ Automatic report generation

#### **LLM Query Generation**

Generate Cypher queries from natural language descriptions using AI:

```bash
# Generate queries using OpenAI GPT-4 (default)
python evals/llm_query_generation.py

# Generate without syntax validation
python evals/llm_query_generation.py --no-validate

# Use custom description and prompt files
python evals/llm_query_generation.py --descriptions /path/to/descriptions.txt --system-prompt /path/to/prompt.txt

# Don't save results to files
python evals/llm_query_generation.py --no-save

# Use different LLM provider (when available)
python evals/llm_query_generation.py --provider openai
```

**LLM Generation Features**:
- 🤖 Automatic query generation from descriptions
- ✅ Built-in syntax validation of generated queries
- 📊 Comprehensive generation statistics
- 💾 Saves both raw results and valid queries
- 🔄 Rate limiting to respect API limits
- 🎨 Beautiful console output with progress tracking

#### **Programmatic Usage**

```python
from utils.cypher_validator import validate_query, validate_query_batch

# Validate a single query
result = validate_query("MATCH (n:User) RETURN n")
print(f"Valid: {result.ok}, Errors: {result.errors}")

# Validate with Neo4j semantic checking
result = validate_query(
    "MATCH (n:User) RETURN n",
    uri="bolt://localhost:7687",
    user="neo4j",
    pwd="password"
)

# Batch validation
queries = [
    {"query": "MATCH (n) RETURN n", "description": "Get all nodes"},
    {"query": "MATCH (u:User) RETURN u", "description": "Get all users"}
]
results = validate_query_batch(queries)
```

## Output Examples

### **Syntax Validation Results**
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Validation Type    ┃ Status ┃ Result        ┃     Time ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Syntax (ANTLR)     │   ✓    │ Valid         │  0.003s  │
│ Semantic (Neo4j)   │   ✓    │ Valid         │  0.045s  │
│                    │        │               │          │
│ Overall            │   ✓    │ Valid         │  0.048s  │
└────────────────────┴────────┴───────────────┴──────────┘
```

### **Validation Summary**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                     ┃    Count ┃ Percentage ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Total Queries              │      150 │      100.0% │
│ Syntax Valid               │      147 │       98.0% │
│ Semantic Valid             │      145 │       96.7% │
│ Overall Valid              │      145 │       96.7% │
│                            │          │             │
│ Average Validation Time    │  0.025s  │             │
└────────────────────────────┴──────────┴─────────────┘
```

### **LLM Generation Results**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                     ┃    Count ┃ Percentage ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Total Descriptions         │       31 │      100.0% │
│ Successful Generations     │       30 │       96.8% │
│ Generation Errors          │        1 │        3.2% │
│ Syntax Valid Queries       │       28 │       93.3% │
│ Syntax Invalid Queries     │        2 │        6.7% │
│ Syntax Success Rate        │          │       93.3% │
│ Total Runtime              │  45.2s   │             │
│ Average Time per Query     │   1.5s   │             │
└────────────────────────────┴──────────┴─────────────┘
```

## Configuration Options

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `BHE_DOMAIN` | - | BloodHound domain (required) |
| `BHE_PORT` | `443` | BloodHound port |
| `BHE_SCHEME` | `https` | Connection scheme |
| `BHE_TOKEN_ID` | - | API token ID (required) |
| `BHE_TOKEN_KEY` | - | API token key (required) |
| `QUERY_DELAY_SECONDS` | `0.5` | Delay between queries |
| `ENABLE_PRE_VALIDATION` | `true` | Enable syntax pre-validation |
| `OPENAI_API_KEY` | - | OpenAI API key (for LLM generation) |
| `OPENAI_MODEL` | `gpt-4` | OpenAI model to use |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `REQUEST_DELAY_SECONDS` | `1.0` | Delay between LLM requests |

### **Query Dataset Format**

The `data/queries.json` file should contain an array of query objects:

```json
[
  {
    "description": "Find all users with SPN",
    "query": "MATCH (n:User) WHERE n.hasspn=true RETURN n",
    "source": "https://example.com/cypher-cheatsheet"
  }
]
```

## Output Files

### **Generated Reports**

- `failed_queries_YYYY-MM-DD_HH-MM-SS.json` - Execution failures
- `syntax_failed_queries_YYYY-MM-DD_HH-MM-SS.json` - Syntax validation failures  
- `syntax_validation_failures_YYYY-MM-DD_HH-MM-SS.json` - Standalone validation failures
- `llm_generation_results_YYYY-MM-DD_HH-MM-SS.json` - Complete LLM generation results
- `llm_generated_queries_YYYY-MM-DD_HH-MM-SS.json` - Valid queries from LLM generation

### **Report Structure**

```json
{
  "validation_timestamp": "2024-01-15T10:30:00",
  "total_queries": 150,
  "failed_queries_count": 5,
  "validation_type": "syntax_only",
  "failed_queries": [
    {
      "query_number": 42,
      "description": "Complex path query",
      "query": "MATCH (n)-[r*1..5]-(m) RETURN n,r,m",
      "syntax_errors": ["line 1:15 missing ')' at '-'"],
      "validation_time": 0.003
    }
  ]
}
```

### **LLM Generation Report Structure**

```json
{
  "evaluation_timestamp": "2024-01-15T10:30:00",
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "total_descriptions": 31,
  "successful_generations": 30,
  "generation_errors": 1,
  "syntax_validation_enabled": true,
  "syntax_valid_queries": 28,
  "syntax_invalid_queries": 2,
  "total_runtime": 45.2,
  "generated_queries": [
    {
      "description_number": 1,
      "description": "Find all users with SPN",
      "raw_response": "<query>MATCH (n:User) WHERE n.hasspn=true RETURN n</query>",
      "generated_query": "MATCH (n:User) WHERE n.hasspn=true RETURN n",
      "generation_time": 1.2,
      "generation_error": null,
      "syntax_valid": true,
      "syntax_errors": [],
      "validation_time": 0.003
    }
  ]
}
```

## Dependencies

- **python-dotenv** - Environment variable management
- **rich** - Beautiful console output and formatting
- **requests** - HTTP client for API calls
- **antlr4-python3-runtime** - ANTLR runtime for parsing
- **antlr4-cypher** - Pre-built Cypher grammar for ANTLR
- **neo4j** - Neo4j driver for semantic validation (optional)
- **openai** - OpenAI API client for LLM query generation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Ensure all validation tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
