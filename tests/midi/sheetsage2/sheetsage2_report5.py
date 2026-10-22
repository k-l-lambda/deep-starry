"""Build Chinese report, plots, CSV, and a local side-by-side score gallery."""
import json,csv,html,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'tests/output/yt-piano-sheetsage2-5-20261004'
def main():
 rows=json.loads((OUT/'metrics.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());byid={s['id']:s for s in manifest['sample']}
 native=json.loads((OUT/'matcher-results.json').read_text());mapping={(s['id'],s['comparison']):s for s in native}
 table=[];cards=[]
 for row in rows:
  sid=row['id'];d=OUT/sid;source=byid[sid];notes=json.loads((d/'midi-note-events.json').read_text())
  if 'abc' not in row:
   cards.append(f'<section><h2>{sid}</h2><pre>{html.escape(json.dumps(row,indent=2,ensure_ascii=False))}</pre></section>');continue
  m=mapping[sid,'abc_vs_reference_score_quarters'];x=m['sample_to_criterion'];y=m['criterion_to_sample'];timing=row['abc_vs_melody'];fit=timing.get('affine_time_fit',{})
  record=dict(id=sid,title=source['title'],audio_sec=source['duration'],inference_sec=row['inference']['seconds'],reference_measures=row['reference_score']['measure_count'],abc_measures=row['abc-roundtrip_score']['measure_count'],reference_notes=row['reference']['notes'],abc_notes=row['abc']['notes'],transcription_melody_notes=row['melody']['notes'],transcription_chord_notes=row['chords']['notes'],abc_to_reference_match=x['coverage'],reference_to_abc_match=y['coverage'],abc_reference_target_reuse=x['reuse'],reference_covered_by_abc_unique=x['unique_targets']/row['reference']['notes'],abc_melody_pitch_lcs=timing['pitch_sequence_lcs'],abc_melody_pitch_lcs_fraction=timing['abc_coverage'],affine_p90_sec=fit.get('p90_abs_sec'),affine_mae_sec=fit.get('mae_sec'),abc_tempo=row['abc']['tempos'],beat_interval_cv=row.get('beat_intervals',{}).get('cv'),reference_flag='283 score bars versus 61 AMT segments; reference consistency unresolved' if sid=='MfDmUk7ie6s' else '')
  table.append(record)
  fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
  for key,color,label in [('melody','#2575b8','Transcribed melody: audio time'),('abc','#bd4932','ABC roundtrip: fixed tempo')]:
   for n in notes[key]:
    if n['start']<=30:axes[0].plot([n['start'],min(n['end'],30)],[n['pitch'],n['pitch']],color=color,linewidth=2,alpha=.65)
   axes[0].plot([],[],color=color,label=label)
  axes[0].set(xlim=(0,30),ylabel='MIDI pitch',title=source['title']+' / first 30 seconds');axes[0].legend()
  pair=np.load(d/'abc-melody-time-pairs.npz');axes[1].scatter(pair['abc_seconds'],pair['transcription_seconds'],s=5)
  xx=pair['abc_seconds'];axes[1].plot(xx,fit['intercept_sec']+fit['scale']*xx,color='#bd4932',linewidth=1)
  axes[1].set(xlabel='ABC roundtrip time (s)',ylabel='Original melody time (s)',title='Same-pitch sequence matches and one global affine fit')
  axes[2].plot(xx,pair['residual_seconds'],'.',markersize=3);axes[2].axhline(0,color='gray',linewidth=.7)
  axes[2].set(xlabel='ABC roundtrip time (s)',ylabel='Time residual (s)',title='Local timing remaining after global offset and scale')
  fig.savefig(d/'timing-comparison.png',dpi=150);plt.close(fig)
  links=' '.join(f'<a href="{sid}/{f}">{label}</a>' for f,label in [('sheetsage2/score.abc','原始ABC'),('sheetsage2.lyl','ABC转Lilylet'),('reference.lyl','数据集Lilylet'),('abc-roundtrip.mid','Lilylet回转MIDI'),('abcjs-crosscheck.mid','ABCjs交叉核验MIDI'),('sheetsage2/transcription.mid','默认转录MIDI'),('sheetsage2/melody.mid','仅旋律MIDI'),('reference.mid','参考谱MIDI')])
  allpages=''.join(f'<details><summary>{name}：全部谱页</summary>'+''.join(f'<a href="{sid}/{p.name}">第{i+1}页</a> ' for i,p in enumerate(sorted(d.glob(prefix+'.page-*.svg'),key=lambda p:int(p.stem.split('-')[-1]))))+'</details>' for name,prefix in [('数据集参考','reference'),('SheetSage2','abc-roundtrip')])
  cards.append(f'<section><h2>{html.escape(source["title"])} · {sid}</h2><p>{links}</p><p>参考/ABC小节：{record["reference_measures"]}/{record["abc_measures"]}；参考/ABC音符：{record["reference_notes"]}/{record["abc_notes"]}；默认转录旋律/伴奏：{record["transcription_melody_notes"]}/{record["transcription_chord_notes"]}。</p><p>{html.escape(record["reference_flag"])}</p><div class="scores"><div><h3>数据集 Lilylet（第1页）</h3><img src="{sid}/reference.page-1.svg"></div><div><h3>SheetSage2 ABC → Lilylet（第1页）</h3><img src="{sid}/abc-roundtrip.page-1.svg"></div></div>{allpages}<img class="chart" src="{sid}/timing-comparison.png"><details><summary>完整指标</summary><pre>{html.escape(json.dumps(row,indent=2,ensure_ascii=False))}</pre></details></section>')
 (OUT/'summary.json').write_text(json.dumps(table,indent=2,ensure_ascii=False))
 if table:
  with (OUT/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SheetSage2 · yt-piano 5首对照</title><style>body{font:16px/1.6 system-ui;margin:2em;background:#f7f8fa;color:#223}section{background:white;border:1px solid #ddd;padding:1.5em;margin:2em 0}a{margin-right:1em}.scores{display:grid;grid-template-columns:1fr 1fr;gap:1em}.scores img{width:100%}.chart{max-width:100%}pre{white-space:pre-wrap;max-height:30em;overflow:auto}h1{font-size:1.7em}</style><h1>SheetSage2：yt-piano 5首完整曲目对照</h1><p>固定种子20261004，从178对中抽取5对；原始音频输入，不先合成MIDI。ABC由本地lilylet解码后经MEI/Verovio转回MIDI。默认转录MIDI另含自动生成和弦伴奏。Matcher覆盖率允许目标复用，不是音符一一对应F1；Auld样本存在明显导航失配。Lilylet回转含延音处理差异，独立abcjs MIDI与详细审计见中文报告；音符数不能直接当模型准确率。</p><p><a href="report.zh-CN.md">中文报告</a><a href="summary.csv">汇总CSV</a><a href="manifest.json">样本及版本</a></p>'''+''.join(cards)+'</html>'
 (OUT/'index.html').write_text(page)
 print(json.dumps(table,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
