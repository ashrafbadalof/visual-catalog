import numpy as np

from catalog.store.sqlite_store import Product, SQLiteStore


def make_store(tmp_path, n=200, dim=16):
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(n, dim)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    products = [Product(id=i + 1, image_path=f"{i + 1}.jpg", title=None,
                        category=["Shoes", "Bags"][i % 2], confidence=0.9, encoder="test")
                for i in range(n)]
    store = SQLiteStore(tmp_path / "test.db", dim=dim)
    store.add(products, embs)
    return store, embs


def test_search_matches_numpy(tmp_path):
    store, embs = make_store(tmp_path)
    for q in embs[:20]:
        expected = (np.argsort(-(embs @ q))[:10] + 1).tolist()  # ids start at 1
        assert [p.id for p, _ in store.search(q)] == expected


def test_filter_and_delete(tmp_path):
    store, embs = make_store(tmp_path)
    assert all(p.category == "Bags" for p, _ in store.search(embs[0], category="Bags"))
    assert store.delete(1)
    assert store.get(1) is None
    assert 1 not in [p.id for p, _ in store.search(embs[0])]