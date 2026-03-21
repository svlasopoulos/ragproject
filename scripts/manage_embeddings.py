# ============================
# Import libraries and modules
# ============================

import hashlib
import json
import os
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_oracledb import OracleVS

import oracledb

print("✅ Successfully imported libaries and modules!")

# ==============
# Get OpenAI key
# ==============

load_dotenv("../config/.env")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY missing in .env")

# ===========================
# Establish Oracle connection
# ===========================

ur = os.getenv("USER")
pw = os.getenv("PASSWORD")
cs = os.getenv("CONNECT_STRING")

try:
    con26ai = oracledb.connect(user=ur, password=pw, dsn=cs)
    print("✅ Successfully connected to Oracle Database!")
    print ('Database version: ', con26ai.version)
except Exception as e:
    print("Connection to Oracle Database failed!")

# ================
# Create Variables
# ================

# Directory where pdf files are saved
PDF_DATA_DIR = os.getenv("PDF_DATA_DIR")

# JSON file in pdf files directory with extended metadata for all files in this directory
# stored metadata include timestamp for each run and mtime, size, sha256 hash, chunck_ids per file
DOC_MANIFEST = os.path.join(PDF_DATA_DIR, ".document_manifest_file.json")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIMENSION = os.getenv("EMBEDDING_DIMENSION")
CHUNK_SIZE = os.getenv("CHUNK_SIZE")
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP")
MAX_WORKERS = 8
ORACLE_TABLE_NAME = os.getenv("ORACLE_TABLE_NAME") # a vector-enabled SQL table
DOC_EXTENSIONS = {".pdf"}
print("✅ Successfully created variables!")

# =================
# Utility functions
# =================

# to detect changes in files
def norm_hash(text: str) -> str:
    """Hash of normalized whitespace."""
    # " ".join(text.split())                   → removes double/unnecessary whitespaces (cosmetic edits should not trigger re-embeddings)
    # " ".join(text.split()).encode("utf-8")   → turns text string into bytes using UTF-8
    # hashlib.sha256(" ".join(text.split()).encode("utf-8")) → runs those bytes through the SHA-256 hashing algorithm (Secure Hash Algorithm 256-bit)
    # hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest() → converts the raw hash bytes into a readable hexadecimal string (64 hex characters)
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest() 

def normalize_text(s: str) -> str:
    return " ".join(s.split()).strip()

def chunk_id(chunk_text: str) -> str:
    return hashlib.sha256(normalize_text(chunk_text).encode("utf-8")).hexdigest()

# removes the base directory from the path and
# convert Win path separators into Unix-style separators
def rel_key(p: str) -> str:
    """Uniform relative key for document manifest and bookkeeping."""
    return os.path.relpath(p, PDF_DATA_DIR).replace("\\", "/")

def file_type(rel_or_path: str) -> str:
    ext = os.path.splitext(rel_or_path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    return "other"

# To load the JSON manifest
def load_doc_manifest(path: str) -> dict:
    if not os.path.exists(path):
        return {"files": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# To save the JSON manifest for the next run
# with some mechanism to make it more stable for interruptions
# and breaks
def save_doc_manifest(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# ----- Loaders for hashing (whole-file text) -----
# mode="single" loads the whole PDF as one document instead of page-by-page
def read_text_pdf(abs_path: str) -> str:
    # Single-document extraction
    doc = PyPDFLoader(abs_path, mode="single").load()[0]
    return doc.page_content

# call the previous function only for PDF coduments
def read_text_for_hash(abs_path: str, ftype: str) -> str:
    if ftype == "pdf":
        return read_text_pdf(abs_path)
    raise ValueError(f"Unsupported file type for hashing: {ftype}")

# ----- Chunking for embeddings -----
# add_start_index=True , will add metadata telling where the chunk started in the original text.
def chunk_text_generic(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    docs = splitter.split_documents([Document(page_content=text)])
    return [(d.page_content, d.metadata.get("start_index", None)) for d in docs]

def build_docs_for_file(rel_path: str, file_hash: str, mtime_ns: int, size: int):
    """
    Build (ids, docs) for a .pdf file type.
    - PDF: load all pages, concatenate, split with RecursiveCharacterTextSplitter
    """
    full_path = os.path.abspath(os.path.join(PDF_DATA_DIR, rel_path))
    ftype = file_type(rel_path)

    ids, docs = [], []

    if ftype == "pdf":
        text = read_text_pdf(full_path)  # one whole blob
        chunks = chunk_text_generic(text)
    else:
        raise ValueError(f"Unsupported file type: {rel_path}")

    for ctext, start in chunks:
        cid = chunk_id(ctext)
        ids.append(cid)
        meta = {
            "source_path": full_path.replace("\\", "/"),
            "filename": os.path.basename(rel_path),
            "file_text_hash": file_hash,
            "chunk_text_hash": cid,   # same as id
            "start_index": start,
            "mtime": mtime_ns,
            "size": size,
            "ingest_version": f"rcs:{CHUNK_SIZE}:{CHUNK_OVERLAP}|{EMBEDDING_MODEL}",
            "type": ftype,
        }
        docs.append(Document(page_content=ctext, metadata=meta))

    return ids, docs

# ============================================================
# PART 1: Detect Changes in all .pdf files in the directory
#         (newly added, modified, deleted files) 
# This acts as a prefilter i.e. ignoring files that haven´t been 
# touched since last run
# How to identify files that changed recently?
#    by changes in file size
#    by changes in "last updated" timestamp
# It is efficient in case we need to process a large number of
# files
# ============================================================

def detect_changes():
    prev_doc_manifest = load_doc_manifest(DOC_MANIFEST)
    prev = prev_doc_manifest.get("files", {})

    # 1) Scan all .pdf files to get modification timestamp and file size
    curr_stats = {}
    for root, _, files in os.walk(PDF_DATA_DIR): # with os.walk we also get files in subdirectories
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in DOC_EXTENSIONS: # we look only for .pdf's
                continue
            if name.startswith("~$"):   # skip temp/lock files
                continue
            p_abs = os.path.join(root, name) # create the full file path of a file
            rel = rel_key(p_abs)
            try:
                # we perform a statistics system call on a file with os.stat
                # this gives us some metadata and statistics on the file that is saved in the system.
                s = os.stat(p_abs)
            except FileNotFoundError:
                continue
            # we get the time of the last change in nanosecondes (in Unix mtime) and the file size (in bytes)   
            curr_stats[rel] = {"mtime": s.st_mtime_ns, "size": s.st_size} 

    # 2) Decide which files need parsing (new or changed by stat)
    to_parse = []
    for rel, st in curr_stats.items():
        old = prev.get(rel)
        if old is None or old.get("mtime") != st["mtime"] or old.get("size") != st["size"]:
            to_parse.append(rel)

    # 3) Parse only those; compute content hash depending on file type
    def parse_and_hash(rel):
        try:
            abs_path = os.path.join(PDF_DATA_DIR, rel)
            ftype = file_type(rel)
            text_for_hash = read_text_for_hash(abs_path, ftype)
            return rel, norm_hash(text_for_hash)
        except Exception as e:
            print(f"[WARN] Failed to parse: {rel} ({e})")
            return rel, None

    # we perform parallel loading/hashing with the ThreadPoolExecutor
    # this speeds up the process if we have to pass many files
    new_hashes = {}
    if to_parse:
        # 
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(parse_and_hash, rel) for rel in to_parse]
            for fut in as_completed(futures):
                rel, h = fut.result()
                if h is not None:
                    new_hashes[rel] = h

    # 4) Classify changes
    added, modified, same_count = [], [], 0
    for rel in to_parse:
        if rel not in new_hashes:
            # failed parse: skip classification; keep previous state if exists
            continue
        old = prev.get(rel)
        if old is None: # if source does not exist -> new/added file
            added.append(rel)
        elif new_hashes[rel] != old.get("text_hash"): # if hash for existing file has changed
            modified.append(rel)
        else:
            same_count += 1 # parsed but content identical

    deleted = sorted(set(prev) - set(curr_stats)) # files in prev but not in curr_stats (deleted)
    unchanged = (len(curr_stats) - len(to_parse)) + same_count # number of completely unchanged files

    # 4b) Renames/moves by matching content hashes (added ↔ deleted)
    # create new list 'renamed' and change metadata only in embedding process 
    deleted_by_hash = defaultdict(list)
    for rel in deleted:
        h = prev[rel].get("text_hash")
        if h:
            deleted_by_hash[h].append(rel)

    renamed = []  # list[(old_rel, new_rel)]
    for rel in list(added):
        h = new_hashes.get(rel)
        olds = deleted_by_hash.get(h, [])
        if h and len(olds) == 1:
            old_rel = olds.pop()
            renamed.append((old_rel, rel))
            added.remove(rel)
            deleted.remove(old_rel)

    # 5) Write updated document manifest (stats + new text hashes when available)
    next_doc_manifest = {"files": {}}
    for rel, st in curr_stats.items():
        if rel in new_hashes:
            next_doc_manifest["files"][rel] = {
                "mtime": st["mtime"],
                "size": st["size"],
                "text_hash": new_hashes[rel],
            }
        else: 
            if rel in prev: # unchanged -> copy forward previous record
                next_doc_manifest["files"][rel] = prev[rel]
            else: # save new stats
                next_doc_manifest["files"][rel] = {
                    "mtime": st["mtime"],
                    "size": st["size"],
                }

    # save the new JSON manifest
    save_doc_manifest(DOC_MANIFEST, next_doc_manifest)

    # 6) Report
    print(f"Renamed/moved: {len(renamed)}")
    for o, n in renamed:
        print(f"  → {o}  ==>  {n}")
    print(f"Added: {len(added)}")
    for r in added:
        print("  +", r)
    print(f"Modified: {len(modified)}")
    for r in modified:
        print("  *", r)
    print(f"Deleted: {len(deleted)}")
    for r in deleted:
        print("  -", r)
    print(f"Unchanged: {unchanged}")
    print("\nDocument Manifest:", DOC_MANIFEST)

    return {
        "prev": prev,                          # old full state (previous manifest)
        "state": next_doc_manifest["files"],   # current files in actual manifest (stats + text_hash)
        "added": added,                        # added
        "modified": modified,                  # modified
        "deleted": deleted,                    # deleted
        "renamed": renamed,                    # renamed
    }

# ===================================
# PART 2: Embeddings / Oracle updates
# ===================================

def update_embeddings(change_set):
    def _clean_meta(md: dict) -> dict:
        """Drop keys whose values are None."""
        return {k: v for k, v in md.items() if v is not None}

    # Input from the previous step
    prev = change_set["prev"]              # previous manifest "files"
    state = change_set["state"]            # current manifest "files"
    added = change_set["added"]
    modified = change_set["modified"]
    deleted = change_set["deleted"]
    renamed = change_set["renamed"]

    # Setup embeddings
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        api_key=openai_api_key)

    # Setup Oracle vector store
    # By setting mutate_on_duplicate=True when a document with an existing ID is provided it will be updated.
    # Oracle will execute a merge statement to insert/update the document
    vector_store = OracleVS(
        con26ai,
        embeddings,
        table_name=ORACLE_TABLE_NAME,
        distance_strategy=DistanceStrategy.COSINE,
        mutate_on_duplicate=True
    )

    # Index of chunk IDs that belong to files classified as DELETED (old document manifest)
    deleted_ids_index = set()
    for rel in deleted:
        deleted_ids_index.update(prev.get(rel, {}).get("chunk_ids", []))

    moved_ids = set()  # track IDs we "move" so we don't delete later

    # A) ADDED → update-overlap (move) or add
    for rel in added:
        try:
            fhash = state.get(rel, {}).get("text_hash")
            if not fhash:
                print(f"[WARN] Skipping added (no text_hash): {rel}")
                continue
            mtime, size = state[rel]["mtime"], state[rel]["size"]
            ids, docs = build_docs_for_file(rel, fhash, mtime, size)
        except Exception as e:
            print(f"[WARN] Skipping added (read/build fail): {rel} ({e})")
            continue

        overlap_ids = [i for i in ids if i in deleted_ids_index]
        if overlap_ids:
            add_docs = [
                Document(page_content=doc.page_content, metadata=_clean_meta(doc.metadata))
                for i, doc in zip(ids, docs) if i in overlap_ids
            ]
            # add document to the vector store
            vector_store.add_documents(add_docs, ids=overlap_ids)
            moved_ids.update(overlap_ids)

        add_ids = [i for i in ids if i not in overlap_ids]
        if add_ids:
            add_docs = [
                Document(page_content=doc.page_content, metadata=_clean_meta(doc.metadata))
                for i, doc in zip(ids, docs) if i in add_ids
            ]
            # add document to the vector store
            vector_store.add_documents(add_docs, ids=add_ids)

        # Record full current list for document manifest (in-memory; will be saved below)
        state[rel]["chunk_ids"] = ids

    # B) MODIFIED → add new, delete removed, re-embed unchanged
    for rel in modified:
        try:
            fhash = state.get(rel, {}).get("text_hash")
            if not fhash:
                print(f"[WARN] Skipping modified (no text_hash): {rel}")
                continue
            mtime, size = state[rel]["mtime"], state[rel]["size"]
            ids, docs = build_docs_for_file(rel, fhash, mtime, size)
        except Exception as e:
            print(f"[WARN] Skipping modified (read/build fail): {rel} ({e})")
            continue

        new_ids = set(ids)
        old_ids = set(prev.get(rel, {}).get("chunk_ids", []))
        to_add = sorted(list(new_ids - old_ids))
        to_del = sorted(list(old_ids - new_ids))
        unchanged_ids = sorted(list(new_ids & old_ids))

        if to_add:
            add_docs = [
                Document(page_content=doc.page_content, metadata=_clean_meta(doc.metadata))
                for i, doc in zip(ids, docs) if i in to_add
            ]
            vector_store.add_documents(add_docs, ids=to_add)

        if to_del:
            vector_store.delete(ids=to_del)

        if unchanged_ids:
            add_docs = [
                Document(page_content=doc.page_content, metadata=_clean_meta(doc.metadata))
                for i, doc in zip(ids, docs) if i in unchanged_ids
            ]
            # add document to the vector store
            vector_store.add_documents(add_docs, ids=unchanged_ids)    
        state[rel]["chunk_ids"] = ids

    # C) RENAMED / MOVED → metadata-only update or first-time ingest
    for old_rel, new_rel in renamed:
        chunk_ids = prev.get(old_rel, {}).get("chunk_ids", [])
        if not chunk_ids:
            # First-time ingest (content existed before but wasn't embedded)
            try:
                fhash = state.get(new_rel, {}).get("text_hash")
                if not fhash:
                    print(f"[WARN] Skipping renamed (no text_hash): {new_rel}")
                    continue
                mtime, size = state[new_rel]["mtime"], state[new_rel]["size"]
                ids, docs = build_docs_for_file(new_rel, fhash, mtime, size)
                if ids:
                    add_docs = [
                        Document(page_content=doc.page_content, metadata=_clean_meta(doc.metadata))
                        for doc in docs
                    ]
                    # add document to the vector store
                    vector_store.add_documents(add_docs, ids=ids)
                state[new_rel]["chunk_ids"] = ids
            except Exception as e:
                print(f"[WARN] Skipping renamed (read/build fail): {new_rel} ({e})")
                continue
        else:
            # Metadata-only - redo the chunks
            fhash = state.get(new_rel, {}).get("text_hash")
            if not fhash:
                print(f"[WARN] Skipping rename metadata update (no text_hash): {new_rel}")
                continue
            mtime, size = state[new_rel]["mtime"], state[new_rel]["size"]
            ids, docs = build_docs_for_file(new_rel, fhash, mtime, size)
            add_docs = [
                Document(page_content=doc.page_content, metadata=_clean_meta(doc.metadata))
                for i, doc in zip(ids, docs) if i in chunk_ids
            ]
            # add document to the vector store
            vector_store.add_documents(add_docs, ids=chunk_ids)
            state[new_rel]["chunk_ids"] = chunk_ids

        if new_rel in state and old_rel in state:
            del state[old_rel]

    # D) DELETED → remove only IDs that were not "moved"
    for rel in deleted:
        old_ids = set(prev.get(rel, {}).get("chunk_ids", []))
        to_delete = sorted(list(old_ids - moved_ids))
        if to_delete:
            # delete document from vector store
            vector_store.delete(ids=to_delete)
        if rel in state:
            del state[rel]

    # Save document manifest with chunk_ids
    doc_manifest_out = {
        "version": 1,
        "scanned_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "files": state,
    }
    save_doc_manifest(DOC_MANIFEST, doc_manifest_out)

    print("\nEmbedding/metadata updates complete.")
    print("Document Manifest:", DOC_MANIFEST)

    # Verify the vectors are persisted in the database
    cursor = con26ai.cursor()
    print("#### Display Embedded Data ####:")
    for row in cursor.execute("SELECT id, text, metadata, embedding FROM " + ORACLE_TABLE_NAME):
        if row is None:
            print("No result from query!")
        print(f"id (binary): {row[0]}, text: {row[1]}, metadata: {row[2]}, embedding: vector[{len(row[3])}]")

    return doc_manifest_out, vector_store

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("=== Detecting changes in docs under:", PDF_DATA_DIR, "===\n")
    changes = detect_changes()
    print("\n=== Updating embeddings in Oracle database ===\n")
    doc_manifest_out, vector_store = update_embeddings(changes)
    print("\nDone.")
