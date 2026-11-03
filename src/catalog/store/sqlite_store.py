import sqlite3
from dataclasses import dataclass

import numpy as np
import sqlite_vec

@dataclass
class Product:
    id: int
    image_path: str
    title: str | None
    category: str
    confidence: float
    encoder: str

class SQLiteStore:
    def __init__(self, path, dim=768):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.load_extension(False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY, image_path TEXT NOT NULL, title TEXT, 
            category TEXT NOT NULL, confidence REAL NOT NULL, encoder TEXT NOT NULL)""")
        self.db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS product_vectors
            USING vec0(embedding float[{dim}], category text)""")

    def add(self, products, embeddings):
        with self.db:
            for p, e in zip(products, np.asarray(embeddings, dtype=np.float32)):
                self.db.execute("INSERT INTO products VALUES (?,?,?,?,?,?)",
                                (int(p.id), p.image_path, p.title, p.category, float(p.confidence), p.encoder))
                self.db.execute("INSERT INTO product_vectors (rowid, embeddings, category) VALUES (?, ?, ?)", 
                                (int(p.id), e.tobytes(), p.category))
    
    def get(self, product_id):
        row = self.db.execute("SELECT * FROM products WHERE id = ?",
                              (int(product_id),)).fetchone()
        return Product(**dict(row)) if row else None

    def delete(self, product_id):
        with self.db:
            cur = self.db.execute("DELETE FROM products WHERE id=?", (int(product_id), ))
            self.db.execute("DELETE FROM product_vectors WHERE rowid=?", (int(product_id), ))
        return cur.rowcount > 0

    def search(self, query_emb, k=10, category=None):
        sql = "SELECT rowid, distance FROM product_vectors WHERE embedding MATCH ? AND k = ?"
        params = [np.asarray(query_emb, dtype=np.float32).tobytes(), k]
        if category:
            sql += " AND category = ?"
            params.append(category)
        rows = self.db.execute(sql + " ORDER BY distance", params).fetchall()
        return [(self.get(r["rowid"]), 1 - r["distance"] ** 2 / 2) for r in rows]

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    def close(self):
        self.db.close()