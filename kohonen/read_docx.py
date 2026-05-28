import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # The namespace for w:p (paragraph) and w:t (text)
            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Extract text paragraph by paragraph to maintain structure
            paragraphs = tree.findall('.//w:p', namespace)
            output = []
            for p in paragraphs:
                texts = p.findall('.//w:t', namespace)
                p_text = ''.join([t.text for t in texts if t.text])
                if p_text:
                    output.append(p_text)
            
            return '\n'.join(output)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    print(extract_text_from_docx(sys.argv[1]))
