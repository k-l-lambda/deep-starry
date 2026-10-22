// Native Intelli-piano onset-aware Matcher. Reuse is reported, not hidden in an F1.
const fs=require('fs');
const M=require('/home/camus/work/intelli-piano/node_modules/@k-l-lambda/music-widgets/lib/source/inc/Matcher');
async function run(c,s){
 const criterion={notes:c.map((n,index)=>({...n,index})),pitchMap:{}};
 const sample={notes:s.map((n,index)=>({...n,index}))};
 for(const n of criterion.notes)(criterion.pitchMap[n.pitch]??=[]).push(n);
 M.genNotationContext(criterion);M.genNotationContext(sample);
 for(const n of sample.notes)M.makeMatchNodes(n,criterion);
 if(!c.length||!s.length)return {matched:0,total:s.length,coverage:0,unique_targets:0,reuse:0,path:[]};
 const nav=await M.runNavigation(criterion,sample);const path=nav.path();
 const pairs=path.flatMap((ci,si)=>ci>=0?[[si,ci]]:[]);for(const [si,ci] of pairs)if(s[si].pitch!==c[ci].pitch)throw Error('pitch mismatch');
 const unique=new Set(pairs.map(x=>x[1])).size;
 return {matched:pairs.length,total:s.length,coverage:pairs.length/s.length,unique_targets:unique,reuse:pairs.length?(pairs.length-unique)/pairs.length:0,path};
}
(async()=>{const input=JSON.parse(fs.readFileSync(process.argv[2]));let out=[];
 for(const p of input){const a=await run(p.criterion,p.sample);const b=await run(p.sample,p.criterion);out.push({id:p.id,comparison:p.comparison,sample_to_criterion:a,criterion_to_sample:b});console.error(p.id,p.comparison,a.coverage,b.coverage);}
 fs.writeFileSync(process.argv[3],JSON.stringify(out));})();
