# tests/conftest.py
import sqlite3
import pytest


@pytest.fixture
def tiny_db(tmp_path):
    """A 2-table shop DB. Built read-write here; code under test opens it read-only."""
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
        INSERT INTO customers VALUES (1,'Ann','NYC'),(2,'Bob','LA'),(3,'Cy','NYC');
        INSERT INTO orders VALUES (1,1,10.0),(2,1,5.0),(3,2,20.0);
        """
    )
    conn.commit()
    conn.close()
    return str(path)
