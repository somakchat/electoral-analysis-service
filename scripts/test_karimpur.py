"""Test the Karimpur query to see if uploaded docs are included."""
import sys
sys.path.insert(0, 'D:/political-agent-sukumar/political-strategy-maker/political-strategy-maker/backend')
sys.stdout.reconfigure(encoding='utf-8')

from app.services.orchestrator import Orchestrator
import asyncio

orch = Orchestrator()

async def test():
    result = await orch.run('what should be strategy for BJP for constituency Karimpur')
    
    # Handle dict or object response
    if isinstance(result, dict):
        answer = result.get('answer', str(result))
        citations = result.get('citations', result.get('evidence', []))
    else:
        answer = getattr(result, 'answer', str(result))
        citations = getattr(result, 'citations', getattr(result, 'evidence', []))
    
    print('=== ANSWER ===')
    print(answer[:4000] if answer else "No answer")
    print('\n=== CITATIONS ===')
    for c in (citations or [])[:5]:
        if isinstance(c, dict):
            source = c.get('source', 'N/A')
            content = c.get('content', c.get('text', 'N/A'))
        else:
            source = getattr(c, 'source', 'N/A')
            content = getattr(c, 'content', getattr(c, 'text', 'N/A'))
        print(f'- Source: {source}')
        print(f'  Text: {str(content)[:150]}...')
        print()

asyncio.run(test())

