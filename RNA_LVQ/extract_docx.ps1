$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open('c:\Users\Gi\Lab_IA_I\IA_2026.1\RNA_LVQ\LVQ1.docx')
$text = $doc.Content.Text
$doc.Close()
$word.Quit()
$text | Out-File -FilePath 'c:\Users\Gi\Lab_IA_I\IA_2026.1\RNA_LVQ\LVQ1_text.txt' -Encoding UTF8
