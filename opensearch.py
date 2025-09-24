import json
from opensearchpy import OpenSearch
from typing import List, Dict, Set
from pathlib import Path

class OpenSearchClient:
    def __init__(self, host: str, index_name: str, timeout: int = 30, id_store_file: str = "processed_ids.json"):
        self.host = host
        self.index_name = index_name
        self.timeout = timeout
        self.client = self._connect()
        self.id_store_file = Path(id_store_file)
        # self.processed_ids: Set[str] = self._load_processed_ids()
        self.processed_ids: Set[str] = set() 

    def _connect(self) -> OpenSearch:
        """Create and return an OpenSearch client."""
        return OpenSearch(hosts=[self.host], timeout=self.timeout)

    def _load_processed_ids(self) -> Set[str]:
        """Load processed document IDs from file."""
        if self.id_store_file.exists():
            with open(self.id_store_file, "r") as f:
                return set(json.load(f))
        return set()
    
    def _save_processed_ids(self):
        """Save processed document IDs to file."""
        with open(self.id_store_file, "w") as f:
            json.dump(list(self.processed_ids), f)

    def get_first_document(self) -> Dict:
        """Fetch the first document from the index."""
        response = self.client.search(
            index=self.index_name,
            body={"query": {"match_all": {}}},
            size=1
        )
        hits = response.get("hits", {}).get("hits", [])
        if hits:
            self.processed_ids.add(hits[0]["_id"])
            return hits[0]["_source"]
        return {}

    def get_all_documents(self) -> List[Dict]:
        """Fetch all documents from the index using a scroll API."""
        docs = []
        page = self.client.search(
            index=self.index_name,
            body={"query": {"match_all": {}}},
            scroll="2m",
            size=100
        )
        sid = page["_scroll_id"]
        scroll_size = len(page["hits"]["hits"])

        while scroll_size > 0:
            if len(docs) > 20:
                break
            for doc in page["hits"]["hits"]:
                docs.append(doc["_source"])
                self.processed_ids.add(doc["_id"])
            page = self.client.scroll(scroll_id=sid, scroll="2m")
            sid = page["_scroll_id"]
            scroll_size = len(page["hits"]["hits"])

        # self._save_processed_ids()
        return docs

    def get_new_documents(self) -> List[Dict]:
        """Fetch only new documents that haven't been processed yet using OpenSearch query."""
        new_docs = []
        # Convert processed_ids set to list for query
        exclude_ids = list(self.processed_ids) if self.processed_ids else ["__none__"]

        page = self.client.search(
            index=self.index_name,
            body={
                "query": {
                    "bool": {
                        "must_not": {
                            "ids": {"values": exclude_ids}  # Exclude processed IDs
                        }
                    }
                }
            },
            scroll="2m",
            size=100
        )

        sid = page["_scroll_id"]
        scroll_size = len(page["hits"]["hits"])

        while scroll_size > 0:
            for doc in page["hits"]["hits"]:
                doc_id = doc["_id"]
                new_docs.append(doc["_source"])
                self.processed_ids.add(doc_id)

            page = self.client.scroll(scroll_id=sid, scroll="2m")
            sid = page["_scroll_id"]
            scroll_size = len(page["hits"]["hits"])

        # self._save_processed_ids()
        return new_docs

