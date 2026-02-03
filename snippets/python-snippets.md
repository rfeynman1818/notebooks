# Python Snippets

A collection of reusable Python code snippets for common tasks.

## 🐍 Python Basics

### List Comprehension with Filter
```python
# Basic list comprehension
result = [x for x in items if x > 0]

# With transformation
result = [x.upper() for x in strings if len(x) > 5]

# Nested comprehension
matrix = [[i*j for j in range(5)] for i in range(5)]
```

### Dictionary Comprehension
```python
# Basic dict comprehension
result = {k: v for k, v in items if v is not None}

# Swap keys and values
inverted = {v: k for k, v in original_dict.items()}

# From two lists
result = {k: v for k, v in zip(keys, values)}
```

### Decorator Template
```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Before function execution
        print(f"Calling {func.__name__}")
        
        result = func(*args, **kwargs)
        
        # After function execution
        print(f"Finished {func.__name__}")
        return result
    return wrapper

@my_decorator
def my_function():
    pass
```

### Context Manager
```python
from contextlib import contextmanager

@contextmanager
def my_context_manager(resource):
    # Setup
    print("Acquiring resource")
    try:
        yield resource
    finally:
        # Cleanup
        print("Releasing resource")

# Usage
with my_context_manager(some_resource) as r:
    # Use resource
    pass
```

### Class with Properties
```python
class MyClass:
    def __init__(self, value):
        self._value = value
    
    @property
    def value(self):
        """Getter for value."""
        return self._value
    
    @value.setter
    def value(self, new_value):
        """Setter with validation."""
        if new_value < 0:
            raise ValueError("Value must be non-negative")
        self._value = new_value
    
    def __repr__(self):
        return f"MyClass(value={self._value})"
```

## 📁 File Operations

### Read File Safely
```python
from pathlib import Path

# Modern way with pathlib
def read_file(filepath):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return path.read_text(encoding='utf-8')

# With context manager
def read_file_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]
```

### Write JSON
```python
import json
from pathlib import Path

def write_json(data, filepath, indent=2):
    """Write data to JSON file with pretty printing."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def read_json(filepath):
    """Read JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
```

## 🌐 API & HTTP

### Requests with Retry
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retry():
    """Create requests session with automatic retry."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Usage
session = create_session_with_retry()
response = session.get('https://api.example.com/data')
```

### Async API Call
```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    """Fetch a single URL."""
    async with session.get(url) as response:
        return await response.json()

async def fetch_multiple(urls):
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage
urls = ['https://api.example.com/1', 'https://api.example.com/2']
results = asyncio.run(fetch_multiple(urls))
```

## 💾 Data Processing

### Pandas Quick Operations
```python
import pandas as pd

# Read CSV with options
df = pd.read_csv('data.csv', parse_dates=['date_column'])

# Filter and select
result = df[df['value'] > 100][['col1', 'col2', 'col3']]

# Group and aggregate
summary = df.groupby('category').agg({
    'value': ['mean', 'sum', 'count'],
    'price': 'mean'
}).reset_index()

# Handle missing values
df_cleaned = df.fillna({'numeric_col': 0, 'string_col': ''})

# Apply function to column
df['new_col'] = df['old_col'].apply(lambda x: x * 2)
```

### List Manipulation
```python
# Flatten nested list
flat = [item for sublist in nested_list for item in sublist]

# Remove duplicates preserving order
unique = list(dict.fromkeys(items))

# Split list into chunks
def chunk_list(lst, n):
    """Split list into chunks of size n."""
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# Find common elements
common = list(set(list1) & set(list2))
```

## 🔧 Utility Functions

### Timing Decorator
```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
```

### Logging Setup
```python
import logging

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with console and optional file output."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

# Usage
logger = setup_logger('my_app', 'app.log')
logger.info('Application started')
```

### Environment Variables
```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get with default
API_KEY = os.getenv('API_KEY', 'default_key')

# Get required (raises error if missing)
def get_required_env(key):
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

DATABASE_URL = get_required_env('DATABASE_URL')
```

## 🧪 Testing

### Pytest Fixture Example
```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        'items': [1, 2, 3],
        'name': 'test'
    }

def test_something(sample_data):
    assert len(sample_data['items']) == 3
    assert sample_data['name'] == 'test'
```

### Mock API Call
```python
from unittest.mock import Mock, patch

def test_api_call():
    """Test function that makes API call."""
    with patch('requests.get') as mock_get:
        # Setup mock
        mock_response = Mock()
        mock_response.json.return_value = {'key': 'value'}
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Test your function
        result = your_function_that_uses_requests()
        
        assert result == {'key': 'value'}
        mock_get.assert_called_once()
```

## 🤖 AI/ML Snippets

### Simple Neural Network Setup (PyTorch)
```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Usage
model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Data Loader (PyTorch)
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Usage
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_data, batch_labels in dataloader:
    # Training loop
    pass
```

## 🔐 Security

### Hash Password
```python
import hashlib
import secrets

def hash_password(password: str) -> tuple[str, str]:
    """Hash a password with a salt."""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return pwd_hash.hex(), salt

def verify_password(password: str, pwd_hash: str, salt: str) -> bool:
    """Verify a password against its hash."""
    new_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return new_hash.hex() == pwd_hash
```

---
Tags: #snippets #python #reference
