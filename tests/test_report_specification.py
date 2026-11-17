"""Offline specification checks: standard library only, no ML dependencies needed."""
import ast
from pathlib import Path
import re
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def literal_assignment(path, name):
    tree = ast.parse((ROOT / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f'Missing source constant {name}')


def report_tables():
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(ROOT / 'NeuroFuzzy_Heart_Health_Report.docx') as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
    return [[[''.join(t.text or '' for t in cell.findall('.//w:t', ns))
               for cell in row.findall('w:tc', ns)] for row in table.findall('w:tr', ns)]
            for table in document.findall('.//w:tbl', ns)]


class ReportSpecificationTests(unittest.TestCase):
    def test_all_fifteen_triangle_parameters_match_word_table(self):
        table = next(t for t in report_tables() if len(t[0]) == 4 and 'Low (a' in t[0][1])
        parameters = literal_assignment('src/fuzzy_logic.py', 'PARAMETERS')
        for row in table[1:]:
            key = row[0].split(' (')[0]
            expected = [tuple(map(float, re.findall(r'-?\d+(?:\.\d+)?', cell.replace('−', '-')))) for cell in row[1:]]
            self.assertEqual(parameters[key], expected)
        self.assertEqual(len(parameters), len(table)-1)

    def test_all_ten_rule_ids_and_consequents_match_word_table(self):
        table = next(t for t in report_tables() if t[0][0] == '#' and t[0][-1] == 'Consequent')
        rules = literal_assignment('src/fuzzy_rules.py', 'RULES')
        self.assertEqual([(r[0], r[2]) for r in rules], [(r[0], float(r[2])) for r in table[1:]])

    def test_every_python_source_parses(self):
        files = [ROOT / 'app.py']
        for directory in ['src', 'tests', 'scripts']:
            files.extend((ROOT / directory).glob('*.py'))
        for path in files:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(), filename=str(path))


if __name__ == '__main__':
    unittest.main()
