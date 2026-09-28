const fs=require("fs"),path=require("path");
function size(d){let t=0,f=0;const w=p=>{let e;try{e=fs.readdirSync(p,{withFileTypes:true})}catch{return}for(const x of e){const q=path.join(p,x.name);if(x.isDirectory())w(q);else{try{t+=fs.statSync(q).size;f++}catch{}}}};w(d);return{mb:t/1048576,f}}
for(const d of ["resources/.venv","resources/bin","resources"]) console.log(`  ${d.padEnd(18)} ${size(d).mb.toFixed(0).padStart(5)} MB  ${size(d).f} 文件`);
console.log("\n=== models 是否存在 ===");
console.log("  resources/models :", fs.existsSync("resources/models"));
