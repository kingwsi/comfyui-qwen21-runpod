'use strict';
const $ = s => document.querySelector(s);
const ACTIVE = new Set(['submitting','queued','running']);
const symbols = {text:'✦',edit:'▧',reference:'▦',faceswap:'◉'};
const assetURL = id => `api/assets/${encodeURIComponent(id)}`;
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let templates={}, selected='edit', images=[null,null,null], job=null, sending=false, uploading=false, timer=null;
let attempt=null, uploadSlot=0;
const drafts={};
function error(message=''){$('#error').textContent=message;$('#error').hidden=!message;}
async function api(path,options={}){
 const headers={...options.headers,'X-Qwen-Studio':'1'};
 const xsrf=document.cookie.split('; ').find(s=>s.startsWith('_xsrf='));
 if(xsrf)headers['X-XSRFToken']=decodeURIComponent(xsrf.slice(6));
 if(options.body&&!(options.body instanceof FormData)){headers['Content-Type']='application/json';options.body=JSON.stringify(options.body);}
 const response=await fetch(`api/${path}`,{...options,headers,credentials:'same-origin',cache:'no-store'});
 let data;try{data=await response.json();}catch{throw new Error('无法读取服务响应，请检查 ComfyUI 是否已启动，并重新登录 Jupyter。');}
 if(!response.ok)throw new Error(data.error||`请求失败（${response.status}）`);
 return data;
}
const active=()=>job&&ACTIVE.has(job.status);
function controls(){
 const busy=sending||uploading||active();
 $('#generate').disabled=!!busy||!templates[selected];
 $('#generate').textContent=uploading?'正在上传…':sending?'正在提交…':active()?'正在生成…':'生成图片 ↗';
 $('#prompt-count').textContent=`${$('#prompt').value.length} / 4000`;
 $('#templates').querySelectorAll('button').forEach(b=>b.disabled=uploading||sending);
 $('#image-slots').querySelectorAll('button').forEach(b=>b.disabled=uploading||sending);
 $('#reset-prompt').disabled=uploading||sending;
}
function renderTemplates(){
 $('#templates').innerHTML=Object.entries(templates).map(([id,t])=>`<button type="button" data-template="${esc(id)}" aria-pressed="${selected===id}"><span class="template-name"><span class="template-symbol" aria-hidden="true">${symbols[id]||'▧'}</span>${esc(t.name)}</span><small>${esc(t.description)}</small></button>`).join('');
}
function renderImages(){
 const t=templates[selected];if(!t)return;
 $('#image-section').hidden=!t.max_images;
 $('#image-count').textContent=t.max_images?(t.min_images===t.max_images?`${t.min_images} 张图片`:`${t.min_images}–${t.max_images} 张图片`):'无需上传图片';
 $('#images-label').textContent=selected==='faceswap'?'目标图片与参考头像':selected==='reference'?'主图与参考图':'上传原图';
 $('#image-hint').textContent=t.image_hint||(selected==='reference'?'按从左到右的顺序对应图 1、图 2、图 3。第一张为主图，第三张可选。':'点击或拖入图片，也可以直接粘贴图片。');
 $('#image-slots').classList.toggle('multi',t.max_images>1);
 $('#image-slots').classList.toggle('pair',t.max_images===2);
 $('#image-slots').innerHTML=Array.from({length:t.max_images},(_,i)=>{
  const label=t.image_labels?.[i]||(i===0?'主图':'参考图 '+(i+1));
  const caption=t.image_labels?`图 ${i+1} · ${label}`:i===0?'图 1 · 主图':`图 ${i+1} · 参考${i===2?'（可选）':''}`;
  return `<div class="image-slot" data-slot="${i}"><button type="button" class="choose" data-upload="${i}" aria-label="${images[i]?'更换':'上传'}${esc(label)}">${images[i]?`<img src="${assetURL(images[i])}" alt="${esc(label)}">`:`<span class="plus" aria-hidden="true">＋</span><span>上传${esc(label)}</span><small>点击或拖入图片</small>`}</button>${images[i]?`<button class="remove" type="button" data-remove="${i}" aria-label="移除图片 ${i+1}">×</button>`:''}<div class="slot-label">${esc(caption)}</div></div>`;
 }).join('');
 controls();
}
function selectTemplate(id){
 if(!templates[id]||uploading||sending)return;
 drafts[selected]=$('#prompt').value;selected=id;
 $('#prompt').value=drafts[id]??templates[id].default_prompt;
 $('#prompt').placeholder=templates[id].placeholder;
 renderTemplates();renderImages();error();
}
function showResult(){
 if(!job)return;
 $('#result-template').textContent=templates[job.template]?.name||'';
 $('#status').classList.toggle('failed',job.status==='failed');
 $('#result-actions').hidden=job.status!=='done';
 const messages={submitting:'正在提交任务…',queued:`已加入队列${job.position?` · 前面还有 ${job.position-1} 个等待任务`:''}…`,running:'正在生成图片…',done:'生成完成',failed:job.error||'生成失败，请检查输入后重新生成。'};
 $('#status').textContent=messages[job.status]||'';
 if(job.status==='done'){
  const url=assetURL(job.result);
  $('#result').innerHTML=`<button type="button" id="result-image" aria-label="查看生成图片"><img src="${url}" alt="生成结果"></button>`;
  $('#download').href=url+'?download=1';
 }else{
  $('#result').innerHTML=`<div class="empty-result"><div class="empty-icon" aria-hidden="true">${job.status==='failed'?'!':'▧'}</div><p>${job.status==='failed'?'本次生成未完成':'正在准备你的图片'}</p><span>${job.status==='failed'?'图片和要求已保留，可调整后重新生成。':'生成任务独立运行，刷新页面也可以继续查看。'}</span></div>`;
 }
 controls();
}
function poll(){
 clearTimeout(timer);if(!active())return;
 const jid=job.id;
 timer=setTimeout(async()=>{
  try{
   const next=await api(`jobs/${jid}`);
   if(job?.id!==jid)return;
   const changed=JSON.stringify(job)!==JSON.stringify(next);job=next;
   if(changed)showResult();error();
  }catch(e){error(e.message+' 将自动重新连接。');}
  if(job?.id===jid)poll();
 },2000);
}
async function generate(e){
 e.preventDefault();if(sending||uploading||active())return;
 const t=templates[selected],text=$('#prompt').value.trim();
 const chosen=images.slice(0,t.max_images);
 if(chosen.slice(0,t.min_images).some(id=>!id)){error(selected==='faceswap'?'请先上传目标图片和参考头像。':`请先上传${t.min_images>1?'主图和参考图':'原图'}。`);return;}
 if(chosen[2]&&!chosen[1]){error('请先添加图 2，再添加图 3。');return;}
 if(!text){error('请填写生成要求。');$('#prompt').focus();return;}
 const body={template:selected,prompt:text,assets:chosen.filter(Boolean)};
 const signature=JSON.stringify(body);
 if(!attempt||attempt.signature!==signature)attempt={signature,request_id:crypto.randomUUID()};
 sending=true;controls();error();
 try{
  job=await api('jobs',{method:'POST',body:{...body,request_id:attempt.request_id}});
  attempt=null;showResult();poll();
 }catch(e){error(e.message+' 输入已保留，再次生成会重试本次提交。');}
 finally{sending=false;controls();}
}
async function upload(files,start=0){
 const list=Array.from(files),max=templates[selected]?.max_images||0;
 if(!list.length||uploading||sending)return;
 if(list.length>max-start){error(`这里还能上传 ${max-start} 张图片。`);return;}
 if(list.some(f=>f.size>20*1024*1024)){error('每张图片不能超过 20 MB。');return;}
 if(list.some(f=>!['image/png','image/jpeg','image/webp'].includes(f.type))){error('请使用 PNG、JPEG 或 WebP 图片。');return;}
 uploading=true;controls();error();
 try{
  for(let i=0;i<list.length;i++){const form=new FormData();form.append('image',list[i]);const a=await api('uploads',{method:'POST',body:form});images[start+i]=a.id;}
 }catch(e){error(e.message);}
 finally{uploading=false;renderImages();$('#files').value='';}
}
function preview(){if(job?.status!=='done')return;$('#preview-image').src=assetURL(job.result);$('#preview').showModal();}
$('#templates').addEventListener('click',e=>{const b=e.target.closest('[data-template]');if(b)selectTemplate(b.dataset.template);});
$('#image-slots').addEventListener('click',e=>{
 const remove=e.target.closest('[data-remove]');if(remove){images[Number(remove.dataset.remove)]=null;renderImages();return;}
 const b=e.target.closest('[data-upload]');if(b){uploadSlot=Number(b.dataset.upload);$('#files').multiple=templates[selected].max_images-uploadSlot>1;$('#files').click();}
});
$('#image-slots').addEventListener('dragover',e=>{e.preventDefault();e.target.closest('.image-slot')?.classList.add('drag-over');});
$('#image-slots').addEventListener('dragleave',e=>e.target.closest('.image-slot')?.classList.remove('drag-over'));
$('#image-slots').addEventListener('drop',e=>{e.preventDefault();const slot=e.target.closest('.image-slot');if(slot){slot.classList.remove('drag-over');upload(e.dataTransfer.files,Number(slot.dataset.slot));}});
$('#files').addEventListener('change',()=>upload($('#files').files,uploadSlot));
$('#generate-form').addEventListener('submit',generate);
$('#prompt').addEventListener('input',controls);
$('#prompt').addEventListener('paste',e=>{const files=Array.from(e.clipboardData.items).filter(i=>i.kind==='file').map(i=>i.getAsFile());if(files.length&&templates[selected].max_images){e.preventDefault();const empty=images.slice(0,templates[selected].max_images).findIndex(a=>!a);upload(files,empty<0?0:empty);}});
$('#reset-prompt').addEventListener('click',()=>{$('#prompt').value=templates[selected].default_prompt;controls();});
$('#view-result').addEventListener('click',preview);$('#result').addEventListener('click',e=>{if(e.target.closest('#result-image'))preview();});
$('#close-preview').addEventListener('click',()=>$('#preview').close());
$('#preview').addEventListener('click',e=>{if(e.target===$('#preview'))$('#preview').close();});
(async()=>{
 try{
  const config=await api('config');templates=config.templates;
  $('#model').textContent=config.model==='turbo'?'Qwen 2.1 Turbo':'Qwen Image 2.1';
  if(config.latest_job){job=await api(`jobs/${config.latest_job}`);selected=templates[job.template]?job.template:'edit';images=job.assets.concat([null,null,null]).slice(0,3);if(selected===job.template)drafts[selected]=job.prompt;}
  $('#prompt').value=drafts[selected]??templates[selected].default_prompt;
  $('#prompt').placeholder=templates[selected].placeholder;renderTemplates();renderImages();showResult();poll();
 }catch(e){error(e.message);}
})();
