"""Prepare all 191 pinned official V* questions without inference or label leakage."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import sys
import time
from pathlib import Path
from PIL import Image, ImageOps
sys.path.insert(0, '/workspace/foveated-reasoning')
from scripts.download_datasets import SOURCES, download, source_url, checksum, verify_image

DATA = Path('/workspace/data/vstar')
OUT = Path('/workspace/vstar-prepared')
META = DATA / 'test_questions.jsonl'
SOURCE = SOURCES['vstar']
assert checksum(META) == SOURCE['sha256']
rows = [json.loads(s) for s in META.read_text().splitlines() if s.strip()]
assert len(rows) == 191
assert len({r['question_id'] for r in rows}) == 191
assert set(r['label'] for r in rows) <= set('ABCD')

def fetch(name):
    for attempt in range(5):
        try:
            path = DATA / name
            assert path.resolve().is_relative_to(DATA.resolve())
            download(source_url('vstar', name), path, image=True)
            return name
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)

names = sorted({r['image'] for r in rows})
with ThreadPoolExecutor(max_workers=8) as pool:
    tasks = {pool.submit(fetch, name): name for name in names}
    for i, task in enumerate(as_completed(tasks), 1):
        print(f'{i}/{len(names)} ready {task.result()}', flush=True)

cases, labels, image_audit = [], {}, []
for r in rows:
    path = (DATA / r['image']).resolve()
    verify_image(path)
    with Image.open(path) as image:
        image.load()
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        size = list(rgb.size)
        pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    image_sha = checksum(path)
    key = 'vstar_' + str(r['question_id'])
    case = dict(key=key, id=str(r['question_id']), dataset='vstar',
        question=r['text'], image_path=str(path), image_sha256=image_sha,
        original_size=size, allowed_letters=list('ABCD'), category=r['category'])
    assert not {'label', 'answer', 'gt_boxes', 'boxes', 'prediction'} & case.keys()
    cases.append(case)
    labels[key] = r['label']
    image_audit.append(dict(path=str(path), bytes=path.stat().st_size,
        image_sha256=image_sha, pixel_sha256=pixel_sha, original_size=size))

source = dict(repository=SOURCE['repo'], revision=SOURCE['revision'],
    metadata_path=str(META), metadata_sha256=checksum(META),
    upstream_url='https://huggingface.co/datasets/'+SOURCE['repo'])
manifest = dict(version=1, source=source, cases=cases)
def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
save(OUT/'manifest.json', manifest)
save(OUT/'labels.evaluation-only.json', labels)
save(OUT/'image_audit.json', image_audit)
save(DATA/'download_manifest.json', dict(**SOURCE, selected_ids=[r['question_id'] for r in rows],
    samples=None, seed=None, stratified=False, metadata_only=False))
validation=dict(question_count=len(cases), unique_ids=len({c['id'] for c in cases}),
    unique_image_paths=len(names), image_files_decoded=len(image_audit),
    images_total_bytes=sum(x['bytes'] for x in image_audit),
    categories=dict(Counter(c['category'] for c in cases)),
    all_allowed_letters='ABCD', official_question_text_preserved=True,
    manifest_inference_fields_exclude_labels_and_boxes=True,
    manifest_sha256=checksum(OUT/'manifest.json'),
    labels_evaluation_only_sha256=checksum(OUT/'labels.evaluation-only.json'), source=source)
save(OUT/'validation.json',validation)
print(json.dumps(validation,indent=2),flush=True)
