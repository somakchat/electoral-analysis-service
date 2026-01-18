"""Read the uploaded document and analyze its content."""
import sys
sys.path.insert(0, 'D:/political-agent-sukumar/political-strategy-maker/political-strategy-maker/backend')
sys.stdout.reconfigure(encoding='utf-8')

import docx

doc = docx.Document('D:/political-agent-sukumar/political-strategy-maker/political-strategy-maker/backend/data/uploads/AI agent for WB 2026 Assembly Election1 (1).docx')

full_text = []
for para in doc.paragraphs:
    if para.text.strip():
        full_text.append(para.text)

print(f'Total paragraphs: {len(full_text)}')
print('\n=== DOCUMENT CONTENT (first 5000 chars) ===\n')
content = '\n'.join(full_text)
print(content[:5000])

# Check for Karimpur mentions
print('\n\n=== KARIMPUR MENTIONS ===')
for i, para in enumerate(full_text):
    if 'karimpur' in para.lower():
        print(f'Para {i}: {para[:500]}...')

# Check for strategy keywords
print('\n\n=== STRATEGY CONTENT SAMPLE ===')
strategy_paras = [p for p in full_text if any(w in p.lower() for w in ['strategy', 'recommend', 'action', 'focus on', 'should'])]
print(f'Paragraphs with strategy keywords: {len(strategy_paras)}')
for p in strategy_paras[:5]:
    print(f'\n- {p[:400]}...')

