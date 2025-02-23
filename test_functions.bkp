"""This module contains test functions for the docstring generator."""

from typing import List, Dict, Optional, Tuple, Any
import datetime
import math
import random
import time
import asyncio


def simple_add(a: int, b: int) -> int:
    return a + b


def process_list(items: List[str], separator: str = ", ") -> str:
    """This is an existing docstring that should be preserved."""
    return separator.join(items)


async def fetch_data(url: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    # This function has no docstring and should get one
    await asyncio.sleep(1)
    return {"status": "success", "data": "example"}


class DataProcessor:
    def __init__(self, name: str):
        self.name = name
        self.processed = 0

    def process_item(self, item: Any) -> Optional[str]:
        self.processed += 1
        if isinstance(item, str):
            return item.upper()
        return None

    @property
    def stats(self) -> Dict[str, int]:
        return {"processed_count": self.processed}


def calculate_age(birthdate: datetime.date) -> int:
    today = datetime.date.today()
    age = today.year - birthdate.year
    if today.month < birthdate.month or (today.month == birthdate.month and today.day < birthdate.day):
        age -= 1
    return age


def complex_operation(
    data: Dict[str, Any],
    threshold: float,
    filters: Optional[List[str]] = None,
    *args: Any,
    **kwargs: Any
) -> Tuple[List[Any], Dict[str, Any]]:
    results = []
    metadata = {}
    
    for key, value in data.items():
        if filters and key not in filters:
            continue
        if isinstance(value, (int, float)) and value > threshold:
            results.append(value)
            metadata[key] = {"original": value, "processed": True}
    
    return results, metadata


class MathOperations:
    @staticmethod
    def fibonacci(n: int) -> List[int]:
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        
        sequence = [0, 1]
        while len(sequence) < n:
            sequence.append(sequence[-1] + sequence[-2])
        return sequence

    @classmethod
    def prime_factors(cls, n: int) -> List[int]:
        factors = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d * d > n:
                if n > 1:
                    factors.append(n)
                break
        return factors


def recursive_search(
    data: Dict[str, Any], 
    target: Any, 
    path: Optional[List[str]] = None
) -> Optional[List[str]]:
    if path is None:
        path = []
    
    for key, value in data.items():
        current_path = path + [key]
        
        if value == target:
            return current_path
        elif isinstance(value, dict):
            result = recursive_search(value, target, current_path)
            if result:
                return result
    
    return None


def validate_config(
    config: Dict[str, Any],
    required_fields: List[str],
    optional_fields: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    errors = []
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Check optional fields
    if optional_fields:
        for field, default in optional_fields.items():
            if field not in config:
                config[field] = default
    
    return len(errors) == 0, errors


class CustomError(Exception):
    pass


def process_with_retries(
    operation: callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    retries = 0
    while retries < max_retries:
        try:
            return operation()
        except Exception as e:
            retries += 1
            if retries == max_retries:
                raise CustomError(f"Operation failed after {max_retries} retries") from e
            time.sleep(delay * (backoff_factor ** (retries - 1)))


def generate_matrix(
    rows: int,
    cols: int,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: Optional[int] = None
) -> List[List[float]]:
    if seed is not None:
        random.seed(seed)
    
    return [
        [random.uniform(min_val, max_val) for _ in range(cols)]
        for _ in range(rows)
    ] 