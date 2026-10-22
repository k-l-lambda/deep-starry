"""Frozen full-song SheetSage2 evaluation. Run GPU only with approved execution."""
import os
os.environ.setdefault('HF_HOME','/tmp/sheetsage2-hf-cache')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import json,time,traceback,hashlib,subprocess,shutil
from pathlib import Path
import torch
from transformers import AutoModel
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tests/output/yt-piano-sheetsage2-5-20261004'
MODELS=Path('/tmp/sheetsage2-models-20261004')
def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False))
def main():
 torch.set_num_threads(4)
 manifest=json.loads((OUT/'manifest.json').read_text())
 # Transformers 4.45.2 misses transitive local imports in this release.
 # Copy the complete unmodified, pinned source set to its dynamic-module cache.
 for name in ['SheetSage2','MERT-v2-FullSong']:
  cache=Path(os.environ['HF_HOME'])/'modules/transformers_modules'/name
  cache.mkdir(parents=True,exist_ok=True)
  for source in (MODELS/name).glob('*.py'):shutil.copy2(source,cache/source.name)
 start=time.monotonic()
 model=AutoModel.from_pretrained(str(MODELS/'SheetSage2'),base_model_path=str(MODELS/'MERT-v2-FullSong'),trust_remote_code=True,local_files_only=True).eval().to('cuda')
 print('MODEL_LOADED',time.monotonic()-start,flush=True)
 provenance={'torch':torch.__version__,'device':torch.cuda.get_device_name(),'load_seconds':time.monotonic()-start,'model_revision':manifest['model_revision'],'mert_revision':manifest['mert_revision'],'preset':'default','dtype':'bf16 autocast','max_seconds':None,'files':{str(p.relative_to(MODELS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.rglob('*.py') if '.cache' not in p.parts}}
 save(OUT/'runtime.json',provenance)
 for row in manifest['sample']:
  d=OUT/row['id'];receipt=d/'inference.json'
  if receipt.exists() and json.loads(receipt.read_text()).get('completed'): print('SKIP_COMPLETED',row['id'],flush=True);continue
  torch.cuda.reset_peak_memory_stats();t=time.monotonic();status={'id':row['id'],'audio_seconds':row['duration']}
  def progress(e):print(json.dumps({'id':row['id'],**e},default=str),flush=True)
  try:
   result=model.transcribe(row['audio'],output_dir=str(d/'sheetsage2'),progress=progress)
   status.update(completed=True,abc_ok=bool(result.get('abc')),result_keys=list(result))
  except Exception as e:
   status.update(completed=False,error=str(e),traceback=traceback.format_exc());print(status['traceback'],flush=True)
   # Official exporter may have saved partial MIDI and events before an ABC error.
   status['partial_files']=[p.name for p in (d/'sheetsage2').glob('*') if p.is_file()]
  torch.cuda.synchronize();status.update(seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  save(receipt,status);print('RESULT',json.dumps(status),flush=True)
if __name__=='__main__':main()
