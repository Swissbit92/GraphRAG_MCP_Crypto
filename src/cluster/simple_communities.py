from typing import List, Dict, Tuple
import itertools
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def kmeans_text_clusters(chunks: List[Dict], k: int = 6) -> Dict:
    texts = [c["text"] for c in chunks]
    if len(texts) < k:
        k = max(2, len(texts))
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    groups = {}
    for lab, ch in zip(labels, chunks):
        groups.setdefault(int(lab), []).append(ch["chunk_id"])
    # top terms per cluster
    top_terms = {}
    for lab in groups:
        idxs = [i for i,(ll,_) in enumerate(zip(labels, chunks)) if int(ll)==lab]
        centroid = km.cluster_centers_[lab]
        terms = vec.get_feature_names_out()
        top = [terms[i] for i in centroid.argsort()[-10:][::-1]]
        top_terms[lab] = top
    return {"clusters": [{"cluster_id": f"k_{lab}", "members": groups[lab], "label_terms": top_terms[lab]} for lab in groups]}

def co_mention_graph(labels: List[Dict]) -> Dict:
    """
    Build graph from entities co-mentioned in same chunk.
    labels[i] must contain entities.{token,protocol,component,organization}
    """
    G = nx.Graph()
    for lab in labels:
        ent_lists = lab.get("entities", {})
        ents = []
        for k in ("token","protocol","component","organization"):
            ents.extend([f"{k}:{e.strip()}" for e in ent_lists.get(k, []) if e and isinstance(e, str)])
        ents = list(dict.fromkeys(ents))
        for a,b in itertools.combinations(ents, 2):
            G.add_edge(a,b, weight=G.get_edge_data(a,b,{}).get("weight",0)+1)
    # simple connected components as 'communities'
    communities = []
    for i, comp in enumerate(nx.connected_components(G)):
        sub = G.subgraph(comp).copy()
        # top nodes by degree
        central = sorted(sub.degree, key=lambda x: x[1], reverse=True)[:8]
        communities.append({
            "community_id": f"g_{i}",
            "entities": sorted(list(comp))[:50],
            "top_entities": [n for n,_ in central]
        })
    return {"communities": communities}
