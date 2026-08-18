import tempfile
from pathlib import Path

from blenderpython_qa_extract import parse_html_for_qa


def test_parse_html_for_qa_uses_chat_message_format():
    html = '''
    <html><body>
      <dl class="py method">
        <dt id="bpy.ops.object.select_all">
          <span class="sig-name descname">select_all</span>
        </dt>
        <dd>
          <p>Select all objects in the scene.</p>
        </dd>
      </dl>
    </body></html>
    '''

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / 'sample.html'
        file_path.write_text(html, encoding='utf-8')

        result = parse_html_for_qa(str(file_path))

    assert len(result) == 1
    assert 'messages' in result[0]
    assert result[0]['messages'][0]['role'] == 'user'
    assert result[0]['messages'][1]['role'] == 'assistant'
    assert '```python' in result[0]['messages'][1]['content']
    assert 'select_all' in result[0]['messages'][1]['content']
