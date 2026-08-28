"""Regression checks for the previously broken GitHub/Typora math syntax."""

import re
import unittest
from pathlib import Path


class DocumentationTests(unittest.TestCase):
    def test_math_delimiters_and_supported_syntax(self):
        text = (Path(__file__).resolve().parents[1] / "interpretation_card.md").read_text(encoding="utf-8")
        self.assertNotIn("\\operatorname", text)
        self.assertNotIn("```math", text)
        self.assertNotIn("```mermaid", text)
        self.assertNotIn("What changed from", text)
        for line in text.splitlines():
            self.assertFalse(re.fullmatch(r"\s*=+\s*", line), "Standalone equals can become a Setext heading")
            if "$$" in line:
                self.assertEqual(line.count("$$"), 2)
                self.assertTrue(line.startswith("$$") and line.endswith("$$"))


if __name__ == "__main__":
    unittest.main()
