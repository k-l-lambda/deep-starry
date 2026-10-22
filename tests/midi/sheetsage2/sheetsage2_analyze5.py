"""Export metrics and native-Matcher inputs; no GPU required."""
import json,math,collections
from pathlib import Path
import numpy as np
import pretty_midi,mido
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tests/output/yt-piano-sheetsage2-5-20261004'
def midi_notes(p):
 midi=pretty_midi.PrettyMIDI(str(p));out=[]
 for ti,ins in enumerate(midi.instruments):
  for n in ins.notes:
   out.append(dict(pitch=n.pitch,start=n.start,end=n.end,q=midi.time_to_tick(n.start)/midi.resolution,end_q=midi.time_to_tick(n.end)/midi.resolution,track=ti,name=ins.name))
 return sorted(out,key=lambda n:(n['start'],n['pitch'],n['end'])),midi
def stats(notes,midi):
 return dict(notes=len(notes),end_seconds=max((n['end'] for n in notes),default=0),quarter_span=max((n['end_q'] for n in notes),default=0),tracks=dict(collections.Counter(n['name'] for n in notes)),tempos=midi.get_tempo_changes()[1].tolist(),pitch_histogram=dict(collections.Counter(n['pitch'] for n in notes)))
def lcs_pairs(a,b):
 b=np.array([n['pitch'] for n in b]);dp=np.zeros((len(a)+1,len(b)+1),dtype=np.int32)
 for i,n in enumerate(a):dp[i+1,1:]=np.maximum.accumulate(np.where(b==n['pitch'],dp[i,:-1]+1,dp[i,1:]))
 i,j=len(a),len(b);pairs=[]
 while i and j:
  if a[i-1]['pitch']==b[j-1] and dp[i,j]==dp[i-1,j-1]+1:pairs.append((i-1,j-1));i-=1;j-=1
  elif dp[i-1,j]>=dp[i,j-1]:i-=1
  else:j-=1
 return pairs[::-1]
def onset_matches(a,b,tol=.05):
 # Maximum-cardinality one-to-one matching by pitch; sorted-time greedy is exact for a fixed tolerance.
 count=0
 for pitch in set(n['pitch'] for n in a)&set(n['pitch'] for n in b):
  aa=sorted(n['start'] for n in a if n['pitch']==pitch);bb=sorted(n['start'] for n in b if n['pitch']==pitch);i=j=0
  while i<len(aa) and j<len(bb):
   if abs(aa[i]-bb[j])<=tol:count+=1;i+=1;j+=1
   elif aa[i]<bb[j]:i+=1
   else:j+=1
 return dict(matched=count,precision=count/len(a) if a else 0,recall=count/len(b) if b else 0,f1=2*count/(len(a)+len(b)) if a or b else 0)
def matcher_notes(notes,score=False):return [dict(pitch=n['pitch'],start=(n['q']*.5 if score else n['start'])*1000) for n in notes]
def main():
 rows=[];matches=[];crosschecks=[]
 for sample in json.loads((OUT/'manifest.json').read_text())['sample']:
  d=OUT/sample['id'];row={'id':sample['id'],'inference':json.loads((d/'inference.json').read_text())};allnotes={}
  paths={'reference':'reference.mid','abc':'abc-roundtrip.mid','transcription':'sheetsage2/transcription.mid','melody':'sheetsage2/melody.mid','chords':'sheetsage2/chords.mid','input_amt':'input-transkun.mid'}
  for key,rel in paths.items():
   if (d/rel).exists():
    notes,midi=midi_notes(d/rel);allnotes[key]=notes;row[key]=stats(notes,midi)
  for key in ['reference','abc-roundtrip']:
   p=d/f'{key}.conversion.json'
   if p.exists():row[key+'_score']=json.loads(p.read_text())
  if 'abc' in allnotes:
   a=allnotes['abc'];b=allnotes['melody'];pairs=lcs_pairs(a,b)
   row['abc_vs_melody']={'pitch_sequence_lcs':len(pairs),'abc_coverage':len(pairs)/len(a) if a else 0,'melody_coverage':len(pairs)/len(b) if b else 0,'raw_onset_50ms':onset_matches(a,b)}
   full=allnotes['transcription'];pf=lcs_pairs(a,full)
   row['abc_vs_full_transcription']={'pitch_sequence_lcs':len(pf),'abc_coverage':len(pf)/len(a) if a else 0,'transcription_coverage':len(pf)/len(full) if full else 0,'raw_onset_50ms':onset_matches(a,full)}
   if (d/'abcjs-crosscheck.mid').exists():
    independent,im=midi_notes(d/'abcjs-crosscheck.mid')
    crosschecks.append(dict(id=sample['id'],lily_notes=len(a),abcjs_notes=len(independent),pitch_lcs=len(lcs_pairs(a,independent)),abcjs_melody_lcs=len(lcs_pairs(independent,b)),onset50=onset_matches(a,independent),abcjs_stats=stats(independent,im),abcjs_vs_full_transcription_pitch_lcs=len(lcs_pairs(independent,full))))
   if len(pairs)>=3:
    x=np.array([a[i]['start'] for i,j in pairs]);y=np.array([b[j]['start'] for i,j in pairs]);A=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(A,y,rcond=None)[0];res=y-A@coef
    adjusted=[dict(n,start=float(coef[0]+coef[1]*n['start'])) for n in a]
    row['abc_vs_melody']['affine_corrected_onset_50ms']=onset_matches(adjusted,b)
    row['abc_vs_melody']['affine_time_fit']={'intercept_sec':coef[0],'scale':coef[1],'mae_sec':float(np.mean(abs(res))),'p90_abs_sec':float(np.quantile(abs(res),.9)),'max_abs_sec':float(max(abs(res)))}
    np.savez_compressed(d/'abc-melody-time-pairs.npz',abc_seconds=x,transcription_seconds=y,residual_seconds=res,pairs=np.array(pairs))
   c=allnotes['reference'];pc=lcs_pairs(a,c);row['abc_vs_reference_pitch_sequence']={'matched':len(pc),'precision':len(pc)/len(a) if a else 0,'recall':len(pc)/len(c) if c else 0,'f1':2*len(pc)/(len(a)+len(c)) if a or c else 0}
   matches.append(dict(id=sample['id'],comparison='abc_vs_reference_score_quarters',criterion=matcher_notes(c,True),sample=matcher_notes(a,True)))
   matches.append(dict(id=sample['id'],comparison='abc_vs_transcribed_melody_seconds',criterion=matcher_notes(b),sample=matcher_notes(a)))
  if 'melody' in allnotes:
   row['melody_vs_input_amt_onset50']=onset_matches(allnotes['melody'],allnotes['input_amt'])
   beats=(d/'sheetsage2/beat.lab')
   if beats.exists():
    times=np.array([float(l.split()[0]) for l in beats.read_text().splitlines() if l.strip()]);intervals=np.diff(times)
    row['beat_intervals']={'n':len(intervals),'median_sec':float(np.median(intervals)),'p10_sec':float(np.quantile(intervals,.1)),'p90_sec':float(np.quantile(intervals,.9)),'cv':float(intervals.std()/intervals.mean())}
  (d/'midi-note-events.json').write_text(json.dumps(allnotes));rows.append(row)
 (OUT/'metrics.json').write_text(json.dumps(rows,indent=2));(OUT/'matcher-input.json').write_text(json.dumps(matches));print(json.dumps(rows,indent=2))
 (OUT/'abcjs-crosscheck.json').write_text(json.dumps(crosschecks,indent=2))
if __name__=='__main__':main()
