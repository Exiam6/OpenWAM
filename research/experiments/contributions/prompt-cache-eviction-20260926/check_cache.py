"""CPU-only regression: extract the exact class, never import/change running code."""
import ast, collections, datetime, hashlib, json, logging, random, sys
from pathlib import Path
source = Path(sys.argv[1])
output = Path(sys.argv[2])
text = source.read_text()
old = '            evicted_key, _ = self.popitem(last=False)'
new = '            evicted_key = next(iter(self))\n            super().__delitem__(evicted_key)'
assert text.count(old) == 1
logging.disable(logging.CRITICAL)
def load(text):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == '_BoundedPromptEmbedCache')
    namespace = dict(OrderedDict=collections.OrderedDict, Any=object, DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE=32, logger=logging.getLogger('regression'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[node.name]
original = load(text)(32)
failure = None
for i in range(33):
    try:
        original[str(i)] = ('embedding', i)
    except Exception as e:
        failure = dict(insertion=i+1, type=type(e).__name__, message=str(e))
        break
assert failure and failure['insertion'] == 33 and failure['type'] == 'KeyError', failure
fixed = load(text.replace(old, new))
runs = []
for capacity in [0, 1, 2, 32]:
    cache = fixed(capacity)
    expected = collections.OrderedDict()
    values = {str(i): object() for i in range(80)}
    rng = random.Random(20260926)
    evictions = 0
    for step in range(5000):
        key = str(rng.randrange(80))
        if step % 3 == 0 and key in expected:
            assert cache[key] is expected[key]
            expected.move_to_end(key)
        elif step % 11 == 0:
            before = list(cache)
            try:
                cache['absent']
            except KeyError:
                pass
            else:
                raise AssertionError('Missing key must raise')
            assert list(cache) == before
        elif step % 17 == 0:
            cache.clear(); expected.clear()
        else:
            cache[key] = values[key]
            expected[key] = values[key]
            expected.move_to_end(key)
            while len(expected) > max(1, capacity):
                expected.popitem(last=False); evictions += 1
        assert list(cache) == list(expected), (capacity, step)
        assert len(cache) <= max(1, capacity)
        for k in expected:
            assert collections.OrderedDict.__getitem__(cache, k) is expected[k]
    runs.append(dict(capacity=capacity, operations=5000, evictions=evictions))
output.write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(), python=sys.version, source=str(source), source_sha256=hashlib.sha256(text.encode()).hexdigest(), original_failure=failure, fixed_checks=runs, passed=True, scope='CPU cache contract only; no GPU inference or frozen source mutation'), indent=2)+'\n')
print(output.read_text())
