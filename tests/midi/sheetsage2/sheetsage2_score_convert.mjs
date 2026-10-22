/** Compare both notations through the same local lilylet -> MEI -> Verovio path. */
import fs from 'node:fs';
import path from 'node:path';
import * as lily from '/home/camus/work/lilylet/lib/lilylet/index.js';
import createModule from '/home/camus/work/lilylet/node_modules/verovio/dist/verovio-module.mjs';
import {VerovioToolkit} from '/home/camus/work/lilylet/node_modules/verovio/dist/verovio.mjs';
const root=path.resolve(process.argv[2] || 'tests/output/yt-piano-sheetsage2-5-20261004');
const toolkit=new VerovioToolkit(await createModule());
toolkit.setOptions({pageWidth:2100,pageHeight:2970,scale:40,adjustPageHeight:true,breaks:'auto'});
const manifest=JSON.parse(fs.readFileSync(path.join(root,'manifest.json')));
// Experiment-only copy: upstream minor-key circle-of-fifths has the sign reversed.
const decoderPath='/home/camus/work/lilylet/lib/lilylet/abcDecoder.js';
const decoderSource=fs.readFileSync(decoderPath,'utf8').replace('fifths += 3;', 'fifths -= 3;').replace(/from "(\.[^"]+)"/g,(_,p)=>`from "${new URL(p,'file://'+decoderPath).href}"`);
fs.writeFileSync(path.join(root,'code','abcDecoder-experiment.mjs'),decoderSource);
const decoder=await import('data:text/javascript;base64,'+Buffer.from(decoderSource).toString('base64'));
for(const sample of manifest.sample){
 const dir=path.join(root,sample.id);
 for(const [name,input,isAbc] of [['reference','reference.lyl',false],['abc-roundtrip','sheetsage2/score.abc',true]]){
  const file=path.join(dir,input);if(!fs.existsSync(file))continue;
  try{
   const text=fs.readFileSync(file,'utf8');
   // Local ABC parser requires numeric voice IDs and nonempty title fields.
   // Preserve model output; save the equivalent parser input separately.
   let normalized=text;
   if(isAbc){
    const ids=new Map();
    normalized=text.replace(/^T:[ \t]*$/gm,'T:SheetSage2').replace(/^V:[ \t]*(\w+)/gm,(_,id)=>{if(!ids.has(id))ids.set(id,ids.size+1);return 'V:'+ids.get(id);});
    // Decoder expects explicit per-measure voice patches, interleaved by bar.
    const header=[], voices=new Map();let body=false,current=null;
    for(const line of normalized.split('\n')){
     if(!body){header.push(line);if(line.startsWith('K:'))body=true;continue;}
     if(/^V:\d+$/.test(line)){current=line.slice(2);if(!voices.has(current))voices.set(current,[]);continue;}
     if(!line.trim()||line.startsWith('%'))continue;
     if(!current)throw Error('ABC body without voice');
     const bars=line.split('|');if(bars.pop().trim())throw Error('Unterminated ABC bar');
     for(const bar of bars){const multi=/^Z(\d+)$/.exec(bar.trim());voices.get(current).push(...(multi?Array(Number(multi[1])).fill('Z'):[bar]));}
    }
    const counts=[...voices.values()].map(x=>x.length);if(new Set(counts).size!==1)throw Error('Unequal voice measure counts');
    normalized=header.join('\n')+'\n'+Array.from({length:counts[0]},(_,i)=>[...voices].map(([v,bars])=>`[V:${v}]${bars[i]}|`).join('\n')).join('\n')+'\n';
    fs.writeFileSync(path.join(dir,'abc-parser-input.abc'),normalized);
   }
   const doc=isAbc?decoder.decode(normalized):lily.parseCode(text);
   const oldError=path.join(dir,`${name}.conversion-error.txt`);if(fs.existsSync(oldError))fs.unlinkSync(oldError);
   if(isAbc)fs.writeFileSync(path.join(dir,'sheetsage2.lyl'),lily.serializeLilyletDoc(doc));
   const onsets=lily.measureOnsets(doc);fs.writeFileSync(path.join(dir,`${name}.onsets.json`),JSON.stringify(onsets));
   fs.writeFileSync(path.join(dir,`${name}.ast.json`),JSON.stringify(doc));
   const mei=lily.meiEncoder.encode(doc);fs.writeFileSync(path.join(dir,`${name}.mei`),mei);
   if(!toolkit.loadData(mei))throw Error('Verovio rejected MEI: '+toolkit.getLog());
   fs.writeFileSync(path.join(dir,`${name}.mid`),Buffer.from(toolkit.renderToMIDI(),'base64'));
   const pages=toolkit.getPageCount();
   for(let page=1;page<=pages;page++)fs.writeFileSync(path.join(dir,`${name}.page-${page}.svg`),toolkit.renderToSVG(page));
   const stats={id:sample.id,name,measure_count:doc.measures.length,pages,note_heads:onsets.flatMap(m=>m.notes).reduce((n,e)=>n+e.midi.length,0),grace_heads:onsets.flatMap(m=>m.notes).filter(e=>e.grace).reduce((n,e)=>n+e.midi.length,0),time_signatures:[...new Set(onsets.map(m=>JSON.stringify(m.timeSig)))],log:toolkit.getLog()};
   fs.writeFileSync(path.join(dir,`${name}.conversion.json`),JSON.stringify(stats,null,2));console.log(JSON.stringify(stats));
  }catch(e){fs.writeFileSync(path.join(dir,`${name}.conversion-error.txt`),e.stack);console.error(sample.id,name,e.stack);process.exitCode=1;}
 }
}
