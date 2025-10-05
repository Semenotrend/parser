# parser.py — single-account (.env), TSV-чанки по 1000 строк.
# Колонки как в ТЗ, теги без "semantics".
# link_out_ratio_72h, reposts_72h, posting_* (включая posting_median_interval_min) и контент-микс, geo_city_guess через AI,
# normalize_patterns и enforce_topic_three_words.
#
# === ДОБАВЛЕНО ===
# - Интеграция онтологии: semantic_ontology_v2_1.json + semantic_canonicalizer_snippet.py
# - Канонизация тегов: format/tonality/patterns/ad_fit/audience/channel_topic
# - Таксономии (2-й уровень): cinema/*, auto/*, travel/*, psychology/*, e_commerce/*, music.genres
# - Запись taxonomy.* как дополнительные строки TSV (не ломая существующую схему)
# - вы

import os
import re
import time
import random
import logging
import csv
import json
import glob
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv, find_dotenv
from telethon.sync import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.errors import UsernameNotOccupiedError, FloodWaitError, RPCError
from socks import SOCKS5
from urllib.parse import urlparse
from statistics import median
from openai import OpenAI

LOG_DIR = "Logs"
DATA_DIR = "Data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(LOG_DIR, "processed")
LOG_PARSER_DIR = os.path.join(LOG_DIR, "parser")

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_PARSER_DIR, exist_ok=True)

# === ОНТОЛОГИЯ: импорт хелпера (ДОБАВЛЕНО) ===
from semantic_canonicalizer_snippet import (
    load_ontology,
    canonicalize_ai_fields,
    classify_text_to_taxonomy,
)

# ---------------------- ENV & CONFIG ----------------------

def load_env():
    env_path = os.getenv("ENV_FILE")

    candidates = []
    if env_path:
        env_candidate = Path(env_path)
        if env_candidate.is_absolute():
            candidates.append(env_candidate)
        else:
            candidates.append(Path.cwd() / env_candidate)
            candidates.append(SCRIPT_DIR / env_candidate)
    else:
        candidates.append(SCRIPT_DIR / ".env")
        found = find_dotenv(usecwd=True)
        if found:
            candidates.append(Path(found))

    for candidate in candidates:
        if candidate and candidate.exists():
            load_dotenv(candidate, override=True)
            break
    else:
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=True)

    session = os.getenv("TELEGRAM_SESSION") or os.getenv("SESSION")
    api_id = os.getenv("TELEGRAM_API_ID") or os.getenv("API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH") or os.getenv("API_HASH")
    if not session or not api_id or not api_hash:
        raise RuntimeError("TELEGRAM_SESSION / TELEGRAM_API_ID / TELEGRAM_API_HASH не заданы в .env")

    session_path = str((SCRIPT_DIR / session).resolve())

    proxy_enabled = os.getenv("PROXY_ENABLED", "0").strip() in ("1","true","True")
    proxy = None
    if proxy_enabled:
        host = os.getenv("PROXY_HOST")
        port = int(os.getenv("PROXY_PORT", "1080"))
        rdns = os.getenv("PROXY_RDNS", "1").strip() in ("1","true","True")
        user = os.getenv("PROXY_USER") or None
        pwd  = os.getenv("PROXY_PASS") or None
        proxy = (SOCKS5, host, port, rdns, user, pwd)

    return {
        "session": session,
        "session_path": session_path,
        "api_id": int(api_id),
        "api_hash": api_hash,
        "proxy": proxy,
        "openai_key": os.getenv("OPENAI_API_KEY"),
        "MSG_LIMIT": int(os.getenv("MSG_LIMIT", "100")),
        "CHANNELS_FILE": os.getenv("CHANNELS_FILE", "channels.txt"),
        "OUTPUT_BASE": os.getenv("OUTPUT_BASE", "channels_analysis"),
        "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", "1000")),
        "CHUNK_PAD": int(os.getenv("CHUNK_PAD", "5")),
        "AI_MODEL": os.getenv("AI_MODEL", "gpt-4o"),
        "AI_POSTS_LIMIT": int(os.getenv("AI_POSTS_LIMIT", "50")),
        # === путь к онтологии (ДОБАВЛЕНО) ===
        "SEMANTIC_ONTOLOGY_PATH": os.getenv("SEMANTIC_ONTOLOGY_PATH", "semantic_ontology_v2_1.json"),
    }


def configure_logging(session_name: str) -> str:
    log_path = os.path.join(LOG_PARSER_DIR, f"{session_name}.log")
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return log_path


SLEEP_MIN, SLEEP_MAX = 3, 7
RESOLVE_BACKOFFS = [5, 15, 45]

# ---------------------- AI PROMPT ----------------------

AI_SYSTEM_PROMPT = """
 Ты — опытный SMM-аналитик и медиабайер. Отвечай строго по схеме.
Верни ТОЛЬКО ВАЛИДНЫЙ JSON UTF-8, без markdown, без комментариев, без лишних полей.

{
  "format":          {"content":"ключи через запятую"},
  "tonality":        {"content":"ключи через запятую"},
  "patterns":        {"content":"ключи через запятую"},
  "ad_fit":          {"content":"ровно 10 ключей через запятую"},
  "brand_mentions":  {"content":"реальные бренды/артисты/лейблы/фестивали из текста через запятую"},
  "audience":        {"content":"ключи через запятую"},
  "channel_topic":   {"content":"ровно 3 ключа через запятую"},
  "channel_summary": {"content":"2–4 коротких предложения без воды"},
  "geo_city_guess":  {"city":"...", "country":"...", "confidence":0.0},
  "brand_candidates":{"list":["бренд1","бренд2","..."]}
}

ПРАВИЛА ФОРМИРОВАНИЯ «СЕМАНТИЧЕСКИХ ЯДЕР» (content) ДЛЯ format / tonality / patterns / ad_fit / audience / channel_topic:
1) Формат ключей:
   — Только смысловые ключи 1–2 слова (существительные или устойчивые словосочетания); без глаголов, без эмодзи.
   — Лемматизируй и пиши в нижнем регистре (кроме brand_mentions).
   — Разделитель: запятая + пробел. Никаких точек, кавычек, тире в конце.
2) Объёмы:
   — format: 4–6 ключей (про подачу: «мемы, афиши, сторителлинг, шортсы, опросы, подборки…»).
   — tonality: 3–6 ключей (тон/манера: «сарказм, ирония, дерзость, провокация, неформальность…»; допускаются прилагательные-существительные, приводить к форме существительного, напр. «сарказм», «ирония»).
   — patterns: 6–10 ключей (повторяющиеся паттерны контента: «анонсы, мемы, личные заметки, афиши, репосты, шутки, плейлисты…»).
   — ad_fit: РОВНО 10 категорий (только ниши/категории, не бренды): «фестивали, клубы, тикетинг, алкоголь, звукотехника, мерч, одежда, такси, бары, стриминги…».
   — audience: 4–8 ключей (сегменты/психографика: «молодёжь, студенты, клубная аудитория, меломаны, тусовщики…»).
   — channel_topic: РОВНО 3 устойчивых темы (не даты/ивенты): «клубная культура, юмор, отношения».
3) Глобальная уникальность (критично):
   — Удали повторы внутри каждого тега.
   — Удали повторы МЕЖДУ тегами из набора {format, tonality, patterns, ad_fit, audience, channel_topic}.
   — Если ключ уже использован в предыдущем теге из этого набора — замени на близкий синоним, которого ещё нет; если синонима нет, пропусти.
   — Не используй ключи из brand_mentions в остальных тегах.
4) brand_mentions:
   — Только реальные бренды/артисты/лейблы/фестивали/медиа/площадки, которые ЯВНО встречаются в TITLE/BIO/POSTS.
   — Сохраняй оригинальный регистр/написание (латиница/кириллица), разделитель — запятая.
   — 0–30 позиций, никаких служебных/общих слов («пятница», «девочка» и т.п.) и выдумок.
   — Поле brand_candidates.list дублирует эти бренды массивом (без дублей) и используется как вспомогательное.
5) channel_summary:
   — 2–4 предложения по делу: чем канал «продаёт внимание», ядро тем и пользы; без эпитетов и воды.
6) geo_city_guess:
   — Если локаль угадывается — заполни city/country и confidence (0..1), иначе верни пустые строки и 0.0.

ВХОД:
Я даю тебе:
TITLE: <заголовок канала>
BIO: <описание канала>
POSTS (sample): <фрагменты постов>

СГЕНЕРИРУЙ РОВНО ОДИН JSON СТРОГО ПО СХЕМЕ ВЫШЕ.
"""

AI_USER_PROMPT_TEMPLATE = """TITLE: {title}
BIO: {bio}
POSTS (sample):
{posts}
"""

def ai_analyze(title, bio, posts_text, openai_api_key, ai_model):
    if not (title or bio or posts_text):
        return {}
    try:
        client_ai = OpenAI(api_key=openai_api_key)
        user_block = f"TITLE:\n{title}\n\nBIO:\n{bio}\n\nPOSTS (sample):\n{posts_text}"
        resp = client_ai.chat.completions.create(
            model=ai_model,
            messages=[{"role": "system", "content": AI_SYSTEM_PROMPT},
                      {"role": "user", "content": user_block}],
            temperature=0.2
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        data = json.loads(raw)
        if not isinstance(data, dict): return {}
        out = {}
        for k in ("format","tonality","patterns","ad_fit","audience","channel_summary","channel_topic","geo_city_guess","brand_candidates"):
            v = data.get(k, {})
            out[k] = v if isinstance(v, dict) else {}
        return out
    except Exception as e:
        logging.error(f"AI JSON error: {e}")
        return {}

# ---------------------- NORMALIZERS ----------------------

def normalize_patterns(text: str) -> str:
    """Из списка с тире/переносами делает одну строку: пункты через запятую."""
    if not text:
        return ""
    parts = []
    for line in re.split(r'[\n;]+', str(text)):
        item = re.sub(r'^\s*[–—-]\s*', '', line).strip()
        if not item:
            continue
        item = re.sub(r'\s+', ' ', item).strip(' .;,-')
        if item:
            parts.append(item)
    if not parts:
        return re.sub(r'\s+', ' ', text).strip(' .;,-')
    return ', '.join(parts)

def enforce_topic_three_words(text: str) -> str:
    """Возвращает ровно 3 слова через запятую (уникальные, в порядке появления)."""
    words = re.findall(r'[A-Za-zА-Яа-яЁё]{2,24}', text or '')
    seen, uniq = set(), []
    for w in words:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw); uniq.append(lw)
    defaults = ['вечеринки', 'отношения', 'юмор']
    for d in defaults:
        if len(uniq) >= 3:
            break
        if d not in seen:
            uniq.append(d); seen.add(d)
    return ', '.join(uniq[:3])

# ---------------------- HELPERS ----------------------

def username_from_link(link: str):
    s = (link or "").strip()
    if not s: return None
    if s.startswith('http'):
        p = urlparse(s); path = (p.path or "").lstrip('/')
    else:
        path = s.lstrip('@')
    if path.startswith('+') or 'joinchat' in s or '/c/' in s or '/' in path:
        return None
    return path.split('?')[0] if path else None

def resolve_entity_safe(client, uname: str):
    last = None
    for i in range(len(RESOLVE_BACKOFFS) + 1):
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        try:
            return client.get_entity(uname)
        except FloodWaitError as e:
            wait_s = int(getattr(e, "seconds", 60)) + 1
            logging.warning(f"[resolve] floodwait @{uname}: sleep {wait_s}s")
            time.sleep(wait_s)
        except UsernameNotOccupiedError as e:
            last = e
            if i < len(RESOLVE_BACKOFFS):
                backoff = RESOLVE_BACKOFFS[i]
                logging.warning(f"[resolve] @{uname} not occupied, retry in {backoff}s")
                time.sleep(backoff); continue
            raise
        except RPCError as e:
            last = e
            if i < len(RESOLVE_BACKOFFS):
                backoff = RESOLVE_BACKOFFS[i]
                logging.warning(f"[resolve] RPC @{uname}: {e.__class__.__name__}, retry in {backoff}s")
                time.sleep(backoff); continue
            raise
        except Exception as e:
            last = e
            if i < len(RESOLVE_BACKOFFS):
                time.sleep(RESOLVE_BACKOFFS[i]); continue
            raise
    if last: raise last

def extract_contacts(text: str):
    if not text:
        return '', '', ''
    usernames = ', '.join(re.findall(r'@\w+', text))
    emails    = ', '.join(re.findall(r'[\w\.-]+@[\w\.-]+', text))
    links     = ', '.join(re.findall(r'https?://\S+', text))
    return usernames, emails, links

def message_has_external_link(msg, self_username: str = "") -> bool:
    text = (getattr(msg, "text", None) or "")
    if re.search(r'(https?://\S+|t\.me/\S+)', text, flags=re.IGNORECASE):
        return True
    btns = getattr(msg, "buttons", None)
    if btns:
        for row in btns:
            for btn in row:
                url = getattr(btn, "url", None)
                if url and (url.startswith("http://") or url.startswith("https://") or "t.me/" in url):
                    return True
    mentions = re.findall(r'@([A-Za-z0-9_]{4,})', text)
    mentions = [m.lower() for m in mentions]
    if mentions:
        if self_username and self_username.lower() in mentions and len(mentions) == 1:
            return False
        return True
    return False

def classify_message_type(msg):
    try:
        if getattr(msg, "video_note", None): return "round"
    except Exception:
        pass
    if getattr(msg, "video", None) is not None: return "video"
    if getattr(msg, "photo", None) is not None: return "image"
    if getattr(msg, "voice", None) is not None: return "audio"
    if getattr(msg, "audio", None) is not None: return "audio"
    doc = getattr(msg, "document", None)
    if doc:
        mime = (getattr(doc, "mime_type", "") or "").lower()
        if mime.startswith("image/"): return "image"
        if mime.startswith("video/"): return "video"
        if mime.startswith("audio/"): return "audio"
        for attr in getattr(doc, "attributes", []) or []:
            if hasattr(attr, "round_message") and getattr(attr, "round_message"): return "round"
            if attr.__class__.__name__ in ("DocumentAttributeAnimated",): return "video"
    if getattr(msg, "sticker", None) is not None: return "image"
    text = (getattr(msg, "text", "") or "").strip()
    return "text" if text else "text"

def compute_posting_cadence_components(messages):
    """Возвращает: posts_per_day, median_interval_min, stability, peak_window"""
    if not messages: return 0.0, 0, "none", "—"
    dates = [m.date for m in messages if getattr(m, 'date', None)]
    if len(dates) < 2: return 0.0, 0, "none", "—"
    dates_sorted = sorted(dates)
    deltas = [(dates_sorted[i] - dates_sorted[i-1]).total_seconds() for i in range(1, len(dates_sorted))]
    if not deltas: return 0.0, 0, "none", "—"
    med_interval_sec = median(deltas)
    median_interval_min = int(round(med_interval_sec / 60))
    days_span = max(1, (dates_sorted[-1] - dates_sorted[0]).days + 1)
    posts_per_day = round(len(dates_sorted) / days_span, 2)
    abs_dev = [abs(d - med_interval_sec) for d in deltas]
    mad = median(abs_dev) if abs_dev else 0
    cv_like = (mad / med_interval_sec) if med_interval_sec else 1
    stability = "high" if cv_like < 0.25 else ("medium" if cv_like < 0.6 else "low")
    hours = [d.hour for d in dates_sorted]
    top_hour = max(set(hours), key=hours.count) if hours else None
    peak_window = f"{top_hour:02d}:00–{(top_hour+1)%24:02d}:00" if top_hour is not None else "—"
    return posts_per_day, median_interval_min, stability, peak_window

# === ДОП. нормализация под таксономии (ДОБАВЛЕНО) ===
def _norm_for_taxonomy(text: str, cfg: dict) -> list:
    if not text:
        return []
    s = text
    if cfg.get("yo2e"):
        s = s.replace("ё","е").replace("Ё","Е")
    if cfg.get("strip_hashtags"):
        s = s.replace("#"," ")
    if cfg.get("strip_at"):
        s = re.sub(r"(^|[\s,;])@", r"\1", s)
    if cfg.get("trim_punct"):
        s = re.sub(r"[^\w\s\-+./]", " ", s)
    if cfg.get("lowercase", True):
        s = s.lower()
    tokens = re.split(r"[\s,;|/]+", s)
    return [t.strip() for t in tokens if t and t.strip()]

# ---------------------- TSV CHUNK MANAGEMENT ----------------------

def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def chunk_filename(base: str, session: str, idx: int, pad: int) -> str:
    return f"{base}_{session}_{str(idx).zfill(pad)}.tsv"

def count_rows_in_tsv(path: str) -> int:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f, start=1):
            pass
        return max(0, i - 1) if 'i' in locals() else 0

def init_chunk_state(output_base, session, fieldnames, chunk_size, pad):
    pattern = f"{output_base}_{session}_" + "[0-9]"*pad + ".tsv"
    files = sorted(glob.glob(pattern))
    if not files:
        idx = 1
        path = chunk_filename(output_base, session, idx, pad)
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8", newline='') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
        return {"index": idx, "rows": 0, "chunk_size": chunk_size, "pad": pad}
    last = files[-1]
    try:
        idx = int(last.rsplit("_", 1)[1].replace(".tsv", ""))
    except Exception:
        idx = 1
    rows = count_rows_in_tsv(last)
    return {"index": idx, "rows": rows, "chunk_size": chunk_size, "pad": pad}

def rotate_if_needed(state, output_base, session, fieldnames):
    if state['rows'] >= state['chunk_size']:
        state['index'] += 1
        state['rows'] = 0
        path = chunk_filename(output_base, session, state['index'], state['pad'])
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8", newline='') as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

def write_row_chunked(row: dict, state, output_base, session, fieldnames):
    if state['rows'] >= state['chunk_size']:
        rotate_if_needed(state, output_base, session, fieldnames)
    path = chunk_filename(output_base, session, state['index'], state['pad'])
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8", newline='') as tsvfile:
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writerow(row)
    state['rows'] += 1

# ---------------------- CORE ----------------------

def process_channels(cfg):
    session_name = cfg["session"]
    session_path = cfg.get("session_path", session_name)
    api_id = cfg["api_id"]
    api_hash = cfg["api_hash"]
    proxy = cfg["proxy"]
    openai_api_key = cfg["openai_key"]
    MSG_LIMIT = cfg["MSG_LIMIT"]
    CHANNELS_FILE = cfg["CHANNELS_FILE"]
    OUTPUT_BASE = cfg["OUTPUT_BASE"]
    CHUNK_SIZE = cfg["CHUNK_SIZE"]
    CHUNK_PAD = cfg["CHUNK_PAD"]
    AI_MODEL = cfg["AI_MODEL"]
    AI_POSTS_LIMIT = cfg["AI_POSTS_LIMIT"]

    output_base_path = os.path.join(RAW_DIR, OUTPUT_BASE)

    # === Загрузка онтологии (ДОБАВЛЕНО) ===
    ontology_path = cfg.get("SEMANTIC_ONTOLOGY_PATH", "semantic_ontology_v2_1.json")
    ontology = load_ontology(ontology_path) or {}
    norm_cfg = ontology.get("normalization", {})

    processed_file = os.path.join(PROCESSED_DIR, f"processed_{session_name}.txt")

    fieldnames = [
        "row_id","title","username","contacts","emails","links","subscribers",
        "engagement_rate_percent","avg_reach_24h","avg_reach_48h","avg_reach_72h",
        "link_out_ratio_72h","reposts_72h",
        "posting_posts_per_day","posting_median_interval_min","posting_stability","posting_peak_hour",
        "posts_text","posts_video","posts_image","posts_round","posts_audio",
        "geo_city_guess_city","geo_city_guess_country","geo_city_guess_confidence",
        "chat_discussion",
        "tag","content","account"
    ]

    chunk_state = init_chunk_state(output_base_path, session_name, fieldnames, CHUNK_SIZE, CHUNK_PAD)

    processed_map = {}
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as pf:
            for line in pf:
                key = line.strip()
                if key: processed_map[key] = True

    with open(CHANNELS_FILE, "r", encoding="utf-8") as f:
        channels = [line.strip() for line in f if line.strip()]

    # connect
    try:
        client = TelegramClient(
            session_path, api_id, api_hash, proxy=proxy,
            device_model='Samsung SM-G996B', system_version='Android 13',
            app_version='10.3.1', lang_code='ru', system_lang_code='ru-RU'
        )
        client.connect()
    except Exception as e:
        logging.error(f"[connect] {session_name} error: {e}")
        print(f"[X] fatal connect for {session_name}: {e}")
        return

    if not client.is_user_authorized():
        logging.error(f"[auth] {session_name} not authorized.")
        print(f"[!] {session_name} не авторизован - завершение.")
        client.disconnect(); return

    from telethon.errors.rpcerrorlist import ChannelPrivateError

    try:
        for index, link in enumerate(channels):
            try:
                k_link = (link or "").lower()
                if processed_map.get(k_link):
                    continue

                uname = username_from_link(link)
                if not uname:
                    with open(processed_file, "a", encoding="utf-8") as pf: pf.write(f"{k_link}\n")
                    continue

                entity = resolve_entity_safe(client, uname)
                full = client(GetFullChannelRequest(channel=entity))
                username = getattr(entity, 'username', '') or ''
                title = getattr(entity, 'title', '') or (getattr(entity, 'first_name', '') or '')
                channel_key = (username or title).strip().lower()
                if processed_map.get(channel_key):
                    continue

                subscribers = getattr(full.full_chat, 'participants_count', 0) or 0
                description = getattr(full.full_chat, 'about', '') or ''
                contacts, emails, links_found = extract_contacts(description)
                chat_discussion = bool(getattr(full.full_chat, 'linked_chat_id', None))

                all_messages = list(client.iter_messages(entity, limit=MSG_LIMIT))

                # ER + reach
                err_list, reach_24, reach_48, reach_72 = [], [], [], []
                now = datetime.now(timezone.utc)
                for msg in all_messages:
                    views = getattr(msg, 'views', None)
                    if not views: continue
                    age_hours = (now - msg.date).total_seconds() / 3600
                    err = (views / subscribers) * 100 if subscribers else 0
                    err_list.append(err)
                    if age_hours <= 24: reach_24.append(views)
                    if age_hours <= 48: reach_48.append(views)
                    if age_hours <= 72: reach_72.append(views)
                avg_err = sum(err_list) / len(err_list) if err_list else 0
                avg_reach_24 = sum(reach_24) / len(reach_24) if reach_24 else 0
                avg_reach_48 = sum(reach_48) / len(reach_48) if reach_48 else 0
                avg_reach_72 = sum(reach_72) / len(reach_72) if reach_72 else 0

                # link_out_ratio_72h + reposts_72h
                window_msgs = [m for m in all_messages if getattr(m, 'date', None) and (now - m.date) <= timedelta(hours=72)]
                link_out_ratio_72h = 0
                reposts_72h = 0
                if window_msgs:
                    with_links_72 = sum(1 for m in window_msgs if message_has_external_link(m, username))
                    link_out_ratio_72h = int(round((with_links_72 * 100) / len(window_msgs)))
                    reposts_72h = sum(1 for m in window_msgs if getattr(m, 'fwd_from', None) is not None)

                # posting cadence
                posts_per_day, median_interval_min, stability, peak_window = compute_posting_cadence_components(all_messages)

                # content mix
                c_text = c_video = c_image = c_round = c_audio = 0
                for m in all_messages:
                    t = classify_message_type(m)
                    if t == "text": c_text += 1
                    elif t == "video": c_video += 1
                    elif t == "image": c_image += 1
                    elif t == "round": c_round += 1
                    elif t == "audio": c_audio += 1

                # AI INPUT (старше 24ч, до 50 постов)
                messages_for_ai = [m for m in all_messages if getattr(m, 'text', None) and (now - m.date) > timedelta(hours=24)]
                base_msgs = messages_for_ai if messages_for_ai else [m for m in all_messages if getattr(m, 'text', None)]
                posts_texts = [m.text for m in base_msgs[-cfg["AI_POSTS_LIMIT"]:]] if base_msgs else []
                posts_text_joined = "\n\n".join(posts_texts) if posts_texts else ""

                # AI анализ (format..topic..geo + brand_candidates)
                ai = ai_analyze(title, description, posts_text_joined, openai_api_key, AI_MODEL) if (title or description or posts_text_joined) else {}

                # теги из AI
                def _get(ai_obj, key):
                    if not isinstance(ai_obj, dict): return ""
                    blk = ai_obj.get(key, {})
                    return (blk.get("content") or "").strip() if isinstance(blk, dict) else ""

                format_c   = _get(ai, "format")
                tonality_c = _get(ai, "tonality")
                patterns_c = _get(ai, "patterns")
                ad_fit_c   = _get(ai, "ad_fit")
                audience_c = _get(ai, "audience")
                summary_c  = _get(ai, "channel_summary")
                topic_c    = _get(ai, "channel_topic")

                geo_blk = ai.get("geo_city_guess", {}) if isinstance(ai, dict) else {}
                geo_city    = (geo_blk.get("city") or "").strip() if isinstance(geo_blk, dict) else ""
                geo_country = (geo_blk.get("country") or "").strip() if isinstance(geo_blk, dict) else ""
                geo_conf    = geo_blk.get("confidence", 0.0) if isinstance(geo_blk, dict) else 0.0
                try: geo_conf = float(geo_conf)
                except Exception: geo_conf = 0.0

                # Fallback, если ИИ пустой
                if not format_c:
                    if (c_video + c_image) >= c_text:
                        format_c = "Визуальные анонсы/афиши с короткими подписями; регулярные внешние ссылки."
                    else:
                        format_c = "Короткие тексты в разговорном стиле: заметки, ироничные ремарки, анонсы."
                if not tonality_c:
                    tonality_c = "Несерьёзная, ироничная, местами провокационная."
                if not patterns_c:
                    pats = ["анонсы мероприятий и клубных событий",
                            "ироничные и саркастические ремарки",
                            "личные комментарии автора"]
                    if (c_video + c_image) > 0: pats.append("афиши и визуальные посты")
                    if link_out_ratio_72h >= 20: pats.append("регулярные внешние ссылки")
                    patterns_c = ', '.join(pats)
                if not ad_fit_c:
                    ad_fit_c = ", ".join([
                        "билеты на мероприятия","клубы и вечеринки","алкогольные напитки","звуковое оборудование",
                        "мерч и одежда","стриминговые сервисы","ивенты и фестивали","бары и рестораны",
                        "энергетики","такси и трансфер"
                    ])
                if not audience_c:
                    audience_c = "Молодые горожане 18–34, увлечённые клубной/фестивальной культурой; ищут афиши, быстрые анонсы и дружеский тон."
                if not summary_c:
                    summary_c = ("Канал с дерзким, неформальным тоном: короткие тексты, афиши и анонсы. "
                                 "Фокус на клубной/фестивальной повестке; автор общается по-свойски и подталкивает к действию.")
                if not topic_c:
                    topic_c = "клубная жизнь, афиши, юмор"

                # НОРМАЛИЗАЦИЯ старой логикой
                patterns_c = normalize_patterns(patterns_c)
                topic_c    = enforce_topic_three_words(topic_c)

                # === КАНОНИЗАЦИЯ по онтологии (ДОБАВЛЕНО) ===
                ai_for_canon = {
                    "format": {"content": format_c},
                    "tonality": {"content": tonality_c},
                    "patterns": {"content": patterns_c},
                    "ad_fit": {"content": ad_fit_c},
                    "audience": {"content": audience_c},
                    "channel_summary": {"content": summary_c},
                    "channel_topic": {"content": topic_c},
                }
                try:
                    ai_canon = canonicalize_ai_fields(ai_for_canon, ontology) or {}
                    # Забираем канонизированные значения (ad_fit → ровно 10, topic → ≤3)
                    format_c   = ai_canon.get("format", {}).get("content", format_c)
                    tonality_c = ai_canon.get("tonality", {}).get("content", tonality_c)
                    patterns_c = ai_canon.get("patterns", {}).get("content", patterns_c)
                    ad_fit_c   = ai_canon.get("ad_fit", {}).get("content", ad_fit_c)
                    audience_c = ai_canon.get("audience", {}).get("content", audience_c)
                    summary_c  = ai_canon.get("channel_summary", {}).get("content", summary_c)
                    topic_c    = ai_canon.get("channel_topic", {}).get("content", topic_c)
                except Exception as e:
                    logging.warning(f"canonicalize_ai_fields failed: {e}")

                # RAW BIO
                bio_raw_content = description or ""

                base_row_id = f"{session_name}-{index+1}"
                common_cols = {
                    "title": title,
                    "username": username,
                    "contacts": contacts,
                    "emails": emails,
                    "links": links_found,
                    "subscribers": subscribers,
                    "engagement_rate_percent": round(avg_err, 2),
                    "avg_reach_24h": round(avg_reach_24, 2),
                    "avg_reach_48h": round(avg_reach_48, 2),
                    "avg_reach_72h": round(avg_reach_72, 2),
                    "link_out_ratio_72h": link_out_ratio_72h,
                    "reposts_72h": reposts_72h,
                    "posting_posts_per_day": posts_per_day,
                    "posting_median_interval_min": median_interval_min,
                    "posting_stability": stability,
                    "posting_peak_hour": peak_window,
                    "posts_text": c_text,
                    "posts_video": c_video,
                    "posts_image": c_image,
                    "posts_round": c_round,
                    "posts_audio": c_audio,
                    "geo_city_guess_city": geo_city,
                    "geo_city_guess_country": geo_country,
                    "geo_city_guess_confidence": round(geo_conf, 2),
                    "chat_discussion": str(chat_discussion).lower(),
                    "account": session_name
                }

                rows = [
                    ("format",          format_c),
                    ("tonality",        tonality_c),
                    ("patterns",        patterns_c),
                    ("ad_fit",          ad_fit_c),
                    ("audience",        audience_c),
                    ("channel_summary", summary_c),
                    ("channel_topic",   topic_c),
                    ("channel_bio_raw", bio_raw_content),
                ]

                # === ТАКСОНОМИИ (2-й уровень) — ДОБАВЛЕНО ===
                try:
                    corpus_full = " ".join([title, description, posts_text_joined])
                    tokens = _norm_for_taxonomy(corpus_full, norm_cfg)

                    # Основные домены через helper (возвращает dict по узлам)
                    tax_domains = ["cinema", "auto", "travel", "psychology", "e_commerce"]
                    for dom in tax_domains:
                        dom_res = classify_text_to_taxonomy(dom, tokens, ontology) or {}
                        if isinstance(dom_res, dict):
                            for node, vals in dom_res.items():
                                if vals:
                                    rows.append((f"{dom}.{node}", ", ".join(vals)))
                        elif isinstance(dom_res, list) and dom_res:
                            rows.append((dom, ", ".join(dom_res)))

                    # Пример: music.genres (часто для клубной тематики)
                    music_res = classify_text_to_taxonomy("music", tokens, ontology)
                    if isinstance(music_res, dict):
                        g = music_res.get("genres", [])
                        if g:
                            rows.append(("music.genres", ", ".join(g)))
                    elif isinstance(music_res, list) and music_res:
                        rows.append(("music.genres", ", ".join(music_res)))
                except Exception as e:
                    logging.warning(f"taxonomy mapping failed: {e}")

                # --- запись TSV ---
                for seq, (tag, content) in enumerate(rows, start=1):
                    row = {"row_id": f"{base_row_id}-{seq}", "tag": tag, "content": (content or "").strip(), **common_cols}
                    write_row_chunked(row, chunk_state, output_base_path, session_name, fieldnames)

                processed_map[channel_key] = True
                with open(processed_file, "a", encoding="utf-8") as pf:
                    pf.write(f"{channel_key}\n")

                print(f"[OK] {session_name} обработал: {title}")

            except ChannelPrivateError:
                with open(processed_file, "a", encoding="utf-8") as pf: pf.write(f"{(link or '').lower()}\n")
            except FloodWaitError as e:
                wait_s = int(getattr(e, "seconds", 60)) + 1
                logging.warning(f"[flood] wait {wait_s}s")
                time.sleep(wait_s)
            except (UsernameNotOccupiedError, RPCError) as e:
                logging.warning(f"[rpc] {link}: {e.__class__.__name__}")
                with open(processed_file, "a", encoding="utf-8") as pf: pf.write(f"{(link or '').lower()}\n")
            except Exception as e:
                logging.error(f"[loop] Ошибка {session_name} с {link}: {e}")
                print(f"[!] Ошибка {session_name} с {link}: {e}")

    finally:
        try: client.disconnect()
        except Exception: pass

# ---------------------- MAIN ----------------------

def main():
    cfg = load_env()
    configure_logging(cfg["session"])
    process_channels(cfg)

if __name__ == "__main__":
    main()
