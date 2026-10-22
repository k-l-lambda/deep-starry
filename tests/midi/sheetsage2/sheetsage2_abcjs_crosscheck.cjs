// Independent ABC parser; suppress synthesized chord backing for melody comparison.
const fs=require('fs');
const abcjs=require('/home/camus/work/lilylet/node_modules/abcjs');
const root='tests/output/yt-piano-sheetsage2-5-20261004';
for(const s of JSON.parse(fs.readFileSync(root+'/manifest.json')).sample){
 const abc=fs.readFileSync(`${root}/${s.id}/sheetsage2/score.abc`,'utf8');
 const midi=abcjs.synth.getMidiFile(abc,{midiOutputType:'binary',chordsOff:true});
 fs.writeFileSync(`${root}/${s.id}/abcjs-crosscheck.mid`,Buffer.from(midi[0]));
}
