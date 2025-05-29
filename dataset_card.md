---
license: mit
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- cybersecurity
- bloodhound
- cypher
- neo4j
- active-directory
- red-team
- blue-team
- graph-database
- query-generation
size_categories:
- 100<n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/queries.json
pretty_name: "BloodHound Cypher Queries Dataset"
---

# BloodHound Cypher Queries Dataset

## Dataset Description

This dataset contains 180 curated Cypher queries specifically designed for BloodHound, the popular Active Directory reconnaissance tool. Each entry pairs a natural language description with its corresponding Cypher query, train and eval your agents for BloodHound query generation :D.

### Dataset Summary

- **Total Examples**: 180 query-description pairs
- **Language**: English (descriptions), Cypher (queries)
- **Domain**: Cybersecurity, Active Directory analysis, Graph databases
- **Use Cases**: Query generation, cybersecurity education, BloodHound automation

### Supported Tasks

- **Text-to-Code Generation**: Generate Cypher queries from natural language descriptions
- **Query Understanding**: Understand the intent behind cybersecurity queries
- **Educational Resource**: Learn BloodHound query patterns and techniques

## Dataset Structure

### Data Instances

Each example contains:

```json
{
  "description": "Find all users with an SPN (Kerberoastable users)",
  "query": "MATCH (n:User) WHERE n.hasspn=true RETURN n",
  "source": "https://hausec.com/2019/09/09/bloodhound-cypher-cheatsheet/"
}
```

### Data Fields

- `description` (string): Natural language description of what the query accomplishes
- `query` (string): The corresponding Cypher query for BloodHound/Neo4j
- `source` (string): Attribution to the original source (URL, author, or publication)

### Data Splits

The dataset is provided as a single collection. Users can create custom splits using the provided utilities:

```python
from datasets import load_dataset
from utils.dataset_utils import split_dataset

dataset = load_dataset("joshtmerrill/HoundBench")
train_set, test_set = split_dataset(dataset, train_ratio=0.8)
```

## Additional Information

### Dataset Curators

This dataset was curated as part of the HoundBench project, a comprehensive toolkit for evaluating and validating Cypher queries against BloodHound instances.

Queries were curated from open and closed sources.

### Licensing Information

This dataset is released under the MIT License. While the dataset itself is freely available, users should respect the original sources and their respective licenses.

### Citation Information

If you use this dataset in your research, please cite:

```bibtex
@dataset{houndbench,
  title={HoundBench: Benchmarking offensive agents},
  author={Josh Merrill},
  year={2025},
  url={https://huggingface.co/datasets/joshtmerrill/HoundBench},
}
```

### Contributions

We welcome contributions to improve and expand this dataset. Please see our [contribution guidelines](https://github.com/your-repo/HoundBench) for more information.

## Usage Examples

### Loading the Dataset

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("joshtmerrill/bloodhound-cypher-queries")

# Load with custom split
train_dataset = load_dataset("joshtmerrill/bloodhound-cypher-queries", split="train[:80%]")
test_dataset = load_dataset("joshtmerrill/bloodhound-cypher-queries", split="train[80%:]")
```

### Basic Usage

```python
# Iterate through examples
for example in dataset:
    print(f"Description: {example['description']}")
    print(f"Query: {example['query']}")
    print(f"Source: {example['source']}")
    print("---")
```

### Integration with HoundBench

```python
from utils.dataset_utils import load_queries_dataset, split_dataset

# Load using HoundBench utilities
dataset = load_queries_dataset("joshtmerrill/bloodhound-cypher-queries")

# Create train/test split
train_set, test_set = split_dataset(dataset, train_ratio=0.8, random_seed=42)

# Filter by source
hausec_queries = filter_dataset_by_source(dataset, ["hausec.com"])
```

### Query Generation Example

```python
from transformers import pipeline

# Load a text generation model
generator = pipeline("text-generation", model="your-model")

# Generate query from description
description = "Find all Domain Admins with active sessions"
prompt = f"Description: {description}\nQuery:"
result = generator(prompt, max_length=100)
print(result[0]['generated_text'])
``` 