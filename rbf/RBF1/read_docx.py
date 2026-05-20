import zipfile
import xml.etree.ElementTree as ET
import os

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

with open('c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF1/extracted.txt', 'w', encoding='utf-8') as f:
    f.write(read_docx('c:/Users/Gi/Lab_IA_I/IA_2026.1/rbf/RBF1/RBF1 (1).docx'))
