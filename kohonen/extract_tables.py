import zipfile
import sys
import xml.etree.ElementTree as ET

def extract_tables_content(docx_path):
    with zipfile.ZipFile(docx_path) as docx:
        xml_content = docx.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        
        ns = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        }
        
        tables = tree.findall('.//w:tbl', ns)
        for i, tbl in enumerate(tables):
            print(f"--- Table {i+1} ---")
            rows = tbl.findall('.//w:tr', ns)
            for r in rows:
                row_data = []
                cells = r.findall('.//w:tc', ns)
                for c in cells:
                    texts = c.findall('.//w:t', ns)
                    text_content = "".join(t.text for t in texts if t.text)
                    row_data.append(text_content.strip())
                print("\t".join(row_data))

if __name__ == '__main__':
    extract_tables_content(sys.argv[1])
