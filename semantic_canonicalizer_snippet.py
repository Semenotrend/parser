
import json, re
from typing import Dict, Any, List

def load_ontology(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _norm_token(tok: str, cfg: Dict[str, Any]) -> str:
    if tok is None:
        return ""
    s = str(tok).strip()
    if cfg.get("yo2e"):
        s = s.replace("ё", "е").replace("Ё", "Е")
    if cfg.get("strip_hashtags"):
        s = re.sub(r"#", "", s)
    if cfg.get("strip_at"):
        s = re.sub(r"^@", "", s)
    if cfg.get("trim_punct"):
        s = re.sub(r"[^\w\s\-+./]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
    if cfg.get("lowercase", True):
        s = s.lower()
    return s

def _tokenize_list(s: str, delims: List[str]) -> List[str]:
    if not s:
        return []
    rx = "|".join([re.escape(d) for d in delims])
    parts = re.split(rx, s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p)
    return out

def _map_token(tag: str, tok: str, ont: Dict[str, Any]) -> str:
    tags = ont.get("tags", {})
    if tag not in tags:
        return tok
    section = tags[tag]
    canon = set(section.get("canonical", []))
    syn = section.get("synonyms", {})
    if tok in canon:
        return tok
    if tok in syn:
        return syn[tok]
    t0 = tok.replace(" ", "")
    for c in canon:
        if c.replace(" ", "") == t0:
            return c
    return tok

def canonicalize_ai_fields(ai: Dict[str, Any], ont: Dict[str, Any]) -> Dict[str, Any]:
    if not ont:
        return ai
    normcfg = ont.get("normalization", {})
    delims = normcfg.get("split_delimiters", [","])
    stop = set(normcfg.get("stopwords", []))
    fields = ["format","tonality","patterns","ad_fit","audience","channel_topic"]
    used_global = set()
    for tag in fields:
        content = ((ai.get(tag) or {}).get("content") or "").strip()
        tokens = _tokenize_list(content, delims)
        normed = []
        for t in tokens:
            t = _norm_token(t, normcfg)
            if not t or t in stop:
                continue
            t = _map_token(tag, t, ont)
            if t in stop:
                continue
            if t in used_global:
                continue
            used_global.add(t)
            normed.append(t)
        if tag == "ad_fit":
            need = 10 - len(normed)
            if need > 0:
                fb = ont.get("tags", {}).get("ad_fit", {}).get("fallback_top", []) or \
                     ont.get("defaults", {}).get("ad_fit_fallback", [])
                for f in fb:
                    if f not in normed and f not in used_global:
                        normed.append(f)
                        used_global.add(f)
                        need -= 1
                        if need <= 0:
                            break
            normed = normed[:10]
        if tag == "channel_topic":
            normed = normed[:3]
        ai.setdefault(tag, {})
        ai[tag]["content"] = ", ".join(normed)
    return ai

def canonicalize_generic_tags(tokens, ont):
    if not ont:
        return list(dict.fromkeys([str(t).strip().lower() for t in tokens if t and str(t).strip()]))
    cfg = ont.get("normalization", {})
    section = ont.get("tags", {}).get("tags", {})
    canon = set(section.get("canonical", []))
    syn = section.get("synonyms", {})
    used = set()
    out = []
    for t in tokens or []:
        s = _norm_token(t, cfg)
        if not s:
            continue
        if s in syn:
            s = syn[s]
        s0 = s.replace(" ", "")
        if s not in canon:
            for c in canon:
                if c.replace(" ", "") == s0:
                    s = c
                    break
        if s in used:
            continue
        used.add(s)
        out.append(s)
    return out

def _canon_match(token, canon: list, syn: dict):
    if token in canon: return token
    if token in syn: return syn[token]
    t0 = token.replace(" ", "").replace("-", "")
    for c in canon:
        if c.replace(" ", "").replace("-", "") == t0:
            return c
    return None

def taxonomy_map(domain: str, tokens, ont: dict, node: str):
    if not ont: return []
    cfg = ont.get("normalization", {})
    section = (ont.get("taxonomy", {}) or {}).get(domain, {})
    if not section: return []
    node_def = section.get(node, {})
    canon = node_def.get("canonical", [])
    syn = node_def.get("synonyms", {})
    out, seen = [], set()
    for t in tokens or []:
        s = _norm_token(t, cfg)
        if not s: continue
        m = _canon_match(s, canon, syn)
        if not m: continue
        if m in seen: continue
        seen.add(m)
        out.append(m)
    return out

def taxonomy_children(domain: str, parent_value: str, ont: dict, node: str):
    if not ont: return []
    section = (ont.get("taxonomy", {}) or {}).get(domain, {})
    children = section.get(node, {})
    return children.get(parent_value, []) if isinstance(children, dict) else []

def classify_text_to_taxonomy(domain: str, text_tokens, ont: dict):
    cfg = ont.get("normalization", {})
    toks = [_norm_token(t, cfg) for t in (text_tokens or []) if t]
    result = {}

    if domain == "cinema":
        genres = taxonomy_map("cinema", toks, ont, "genres")
        sub = []
        for g in genres:
            sub += [s for s in toks if s in taxonomy_children("cinema", g, ont, "subgenres")]
        regions = taxonomy_map("cinema", toks, ont, "regions")
        if genres: result["genres"] = genres
        if sub: result["subgenres"] = sorted(list(dict.fromkeys(sub)))
        if regions: result["regions"] = regions

    if domain == "auto":
        brands = taxonomy_map("auto", toks, ont, "brands")
        classes = taxonomy_map("auto", toks, ont, "classes")
        power = taxonomy_map("auto", toks, ont, "powertrain")
        segm = taxonomy_map("auto", toks, ont, "segments")
        if brands: result["brands"] = brands
        if classes: result["classes"] = classes
        if power: result["powertrain"] = power
        if segm: result["segments"] = segm

    if domain == "travel":
        types = taxonomy_map("travel", toks, ont, "types")
        countries_def = (ont.get("taxonomy", {}).get("travel", {}).get("countries", {}) or {})
        countries = []
        canon = countries_def.get("canonical", [])
        syn = countries_def.get("synonyms", {})
        for t in toks:
            m = _canon_match(t, canon, syn)
            if m and m not in countries: countries.append(m)
        cities_def = (ont.get("taxonomy", {}).get("travel", {}).get("cities", {}) or {})
        by_country = cities_def.get("by_country", {})
        city_syn = cities_def.get("synonyms", {})
        cities = []
        for t in toks:
            tt = t
            if tt in city_syn: tt = city_syn[tt]
            for co, lst in by_country.items():
                if tt in lst and tt not in cities:
                    cities.append(tt)
        if types: result["types"] = types
        if countries: result["countries"] = countries
        if cities: result["cities"] = cities

    if domain == "psychology":
        topics = taxonomy_map("psychology", toks, ont, "topics")
        mods = taxonomy_map("psychology", toks, ont, "modalities")
        forms = taxonomy_map("psychology", toks, ont, "formats")
        if topics: result["topics"] = topics
        if mods: result["modalities"] = mods
        if forms: result["formats"] = forms

    if domain == "e_commerce":
        mp_def = (ont.get("taxonomy", {}).get("e_commerce", {}).get("marketplaces", {}) or {})
        syn = (ont.get("taxonomy", {}).get("e_commerce", {}).get("synonyms", {}) or {})
        flat = {}
        for region, lst in mp_def.items():
            if not isinstance(lst, list): 
                continue
            for s in lst:
                flat[s] = region
        mps = []
        regions = []
        for t in toks:
            t0 = t
            if t0 in syn: t0 = syn[t0]
            if t0 in flat:
                mps.append(t0)
                regions.append(flat[t0])
        if mps: result["marketplaces"] = sorted(list(dict.fromkeys(mps)))
        if regions: result["regions"] = sorted(list(dict.fromkeys(regions)))
    return result
