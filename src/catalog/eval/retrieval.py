import numpy as np
import pandas as pd


def _first_hit_ranks(sims, relevant_fn):
    order = np.argsort(-sims, axis=1)
    hits = relevant_fn(order)
    return hits.argmax(axis=1)


def text_to_image_retrieval(query_embs, query_ids, query_titles,
                            gallery_embs, gallery_ids, gallery_titles,
                            ks=(1, 5, 10)):
    sims = query_embs @ gallery_embs.T

    id_pos = pd.Series(np.arange(len(gallery_ids)), index=gallery_ids)
    target = id_pos.loc[query_ids].to_numpy()
    exact = _first_hit_ranks(sims, lambda order: order == target[:, None])

    codes, _ = pd.factorize(np.concatenate([query_titles, gallery_titles]))
    q_codes, g_codes = codes[:len(query_titles)], codes[len(query_titles):]
    title = _first_hit_ranks(sims, lambda order: g_codes[order] == q_codes[:, None])

    results = {"n_queries": int(len(query_ids)), "gallery_size": int(len(gallery_ids))}
    for name, ranks in (("exact", exact), ("title", title)):
        for k in ks:
            results[f"{name}_recall@{k}"] = float((ranks < k).mean())
        results[f"{name}_mrr"] = float((1.0 / (ranks + 1)).mean())
    return results