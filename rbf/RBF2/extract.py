import zipfile
import xml.etree.ElementTree as ET

def read_docx(path):
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = ET.XML(xml_content)

    paragraphs = []
    for paragraph in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        texts = [node.text
                 for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
                 if node.text]
        if texts:
            paragraphs.append(''.join(texts))
    return '\n'.join(paragraphs)

def read_docx_tables(path):
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = ET.XML(xml_content)
    
    tables = []
    for tbl in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl'):
        table_data = []
        for row in tbl.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr'):
            row_data = []
            for cell in row.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc'):
                texts = []
                for p in cell.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                    p_text = []
                    for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                        if t.text: p_text.append(t.text)
                    texts.append("".join(p_text))
                row_data.append(" ".join(texts).strip())
            table_data.append(row_data)
        tables.append(table_data)
    return tables

path = 'c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF2/RBF2 (1).docx'

with open('c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF2/extracted.txt', 'w', encoding='utf-8') as f:
    f.write(read_docx(path))

with open('c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF2/extracted_tables.txt', 'w', encoding='utf-8') as f:
    tables = read_docx_tables(path)
    for i, t in enumerate(tables):
        f.write(f"Table {i}:\n")
        for r in t:
            f.write(str(r) + "\n")
