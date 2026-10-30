import unittest

from rag_retrieval import ingest_document, retrieve


class RetrievalTests(unittest.TestCase):
    def test_retrieval_returns_traceable_source(self):
        document = ingest_document(b"Forest monitoring guidance: review wildlife observations with human experts.", "guidance.md")
        hits = retrieve([document], "wildlife observations require review")
        self.assertEqual(hits[0]["source"], "guidance.md")
        self.assertIn("human experts", hits[0]["excerpt"])

    def test_rejects_non_text_document(self):
        with self.assertRaises(ValueError):
            ingest_document(b"\xff\xfe", "bad.txt")
