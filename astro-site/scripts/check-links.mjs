import {globSync,readFileSync,existsSync} from 'node:fs';
import {parseHTML} from 'linkedom';
import { resolve } from 'node:path';
const root=resolve('dist');
let failures=[];let count=0;
for(const file of globSync(root+'/**/*.html')){
 const {document}=parseHTML(readFileSync(file,'utf8'));
 if(document.querySelector('meta[http-equiv="refresh"]'))continue;
 const route='/compressionkit/'+file.slice(root.length+1).replace(/index.html$/,'');
 for(const el of document.querySelectorAll('main a[href],main img[src]')){
 const ref=el.getAttribute('href')||el.getAttribute('src');if(!ref||/^(mailto:|data:)/.test(ref))continue;
 const u=new URL(ref,'https://local'+route);
 if (u.origin !== 'https://local' && !(u.hostname === 'ambiqai.github.io' && u.pathname.startsWith('/compressionkit/'))) continue;
 if(!u.pathname.startsWith('/compressionkit/')){failures.push([route,ref,'base']);continue;}
 let target=root+decodeURIComponent(u.pathname.slice('/compressionkit'.length));if(target.endsWith('/'))target+='index.html';
 if(!existsSync(target)){failures.push([route,ref,'missing']);continue;}
 if(u.hash && target.endsWith('.html') && !['#only-light','#only-dark'].includes(u.hash)){
 const {document:d}=parseHTML(readFileSync(target,'utf8'));const id=decodeURIComponent(u.hash.slice(1));if(!d.getElementById(id))failures.push([route,ref,'anchor']);
 }
 count++;
 }
 const body=document.querySelector('.sl-markdown-content')?.textContent||'';
 if(/:material-|\{\s*\.\w|!!! |::: compressionkit|=== "/.test(body))failures.push([route,'raw syntax']);
}
console.log(JSON.stringify({count,failures},null,2));
if(failures.length) process.exitCode=1;
