let events=[],visitors=[],latestStats=null;
let browserCameraStream=null,browserCameraActive=false,browserCameraGeneration=0;
let browserCameraStartedAt=0,browserFramesAtStart=0,latestBrowserFramesReceived=0;
let unrecognizedRenderKey=null;
const $=id=>document.getElementById(id);

function escapeHTML(v){return String(v??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;")}
function shortTime(v){if(!v)return"—";const m=String(v).match(/(\d{2}:\d{2}:\d{2})/);return m?m[1]:String(v)}
function kind(type){type=String(type||"").toUpperCase();if(type.includes("EXIT"))return"exit";if(type==="ENTRY"||type==="VISITOR_REGISTERED")return"entry";return"other"}
function symbol(type){const k=kind(type);return k==="exit"?"↓":k==="entry"?"↑":"•"}

async function fetchJSON(url){const r=await fetch(url,{cache:"no-store"});if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r.json()}

function showSourceMessage(message,error=false){
  const node=$("sourceMessage");if(!node)return;
  node.textContent=message;node.classList.toggle("error",error);
}

function showResetMessage(message,error=false){
  const node=$("resetMessage");if(!node)return;
  node.textContent=message;node.classList.toggle("error",error);
  node.classList.toggle("visible",Boolean(message));
}

function renderUnrecognizedPeople(people=[]){
  const list=$("unrecognizedPeopleList"),count=$("unrecognizedCount");
  if(!list||!count)return;
  const current=Array.isArray(people)?people:[];
  count.textContent=`${current.length} recorded`;
  const renderKey=JSON.stringify(current);
  if(renderKey===unrecognizedRenderKey)return;
  unrecognizedRenderKey=renderKey;
  if(!current.length){
    list.innerHTML=`<div class="empty">No unrecognized sightings have been recorded yet.</div>`;
    return;
  }
  list.innerHTML=current.map(person=>{
    const kind=person.kind==="person"?"person":"face";
    const photo=person.image_url
      ?`<img class="unrecognized-photo" src="${escapeHTML(person.image_url)}" alt="Saved image of unrecognized sighting ${escapeHTML(person.id)}" loading="lazy" onerror="this.hidden=true">`
      :`<span class="unrecognized-icon">${kind==="person"?"P":"F"}</span>`;
    return `<article class="unrecognized-row ${kind}">${photo}<strong>${escapeHTML(person.label||"Not identified")}</strong><small>${kind==="person"?"No face visible":"Face was not recognized at first sighting"}</small><time datetime="${escapeHTML(person.first_seen||"")}">First seen ${escapeHTML(person.first_seen||"—")}</time></article>`;
  }).join("");
}

async function fetchAPI(url,options){
  const response=await fetch(url,options);
  const result=response.status===204?{}:await response.json();
  if(!response.ok)throw Error(result.error||`${url}: HTTP ${response.status}`);
  return result;
}

async function refreshSourceStatus(){
  try{
    const status=await fetchJSON("/api/status");
    const reset=status.reset||{};
    const resetButton=$("resetDatabaseButton");
    if(resetButton)resetButton.disabled=["queued","processing"].includes(reset.state);
    if(reset.state==="complete")showResetMessage(reset.message);
    else if(reset.state==="error")showResetMessage(reset.message,true);
    else if(reset.state==="queued"||reset.state==="processing")showResetMessage(reset.message);
    else showResetMessage("");
    const state=String(status.state||"").toLowerCase();
    const unrecognized=await fetchJSON("/api/unrecognized-people");
    renderUnrecognizedPeople(unrecognized);
    const receivedFrames=Number(status.browser_camera?.frames_received||0);
    latestBrowserFramesReceived=receivedFrames;
    if(browserCameraActive&&status.source!=="Browser camera"){
      browserCameraActive=false;browserCameraGeneration++;
      if(browserCameraStream)browserCameraStream.getTracks().forEach(track=>track.stop());
      browserCameraStream=null;$("browserCameraPreview").srcObject=null;$("browserCameraPreview").hidden=true;
      browserCameraStartedAt=0;
      $("browserCameraButton").disabled=false;$("stopBrowserCameraButton").hidden=true;
    }
    showSourceMessage(status.message||`Input source: ${status.source||"not selected"}`,state==="error");
    if(browserCameraActive&&status.source==="Browser camera"){
      if(receivedFrames>browserFramesAtStart){
        const dims=status.browser_camera?.dimensions;
        showSourceMessage(`Device camera connected${dims?` (${dims[0]}×${dims[1]})`:""}; frames are reaching the tracker.`);
      }else if(Date.now()-browserCameraStartedAt>5000){
        showSourceMessage(
          "Camera opened here, but the tracker received no frames. For another device, use a trusted HTTPS URL and ensure the tracker is reachable on the network.",
          true
        );
      }
    }
    if(state==="running"){$("engineState").textContent="Connected";$("feedMessage").style.display="none"}
    else if(state==="paused"){ $("engineState").textContent="Camera stopped";$("feedMessage").textContent=status.message;$("feedMessage").style.display="grid" }
    else if(state==="waiting"){ $("engineState").textContent="Waiting for camera";$("feedMessage").textContent=status.message;$("feedMessage").style.display="grid" }
    else if(state==="reconnecting"){ $("engineState").textContent="Reconnecting";$("feedMessage").textContent=status.message;$("feedMessage").style.display="grid" }
    else if(state==="completed"){ $("engineState").textContent="Video complete";$("feedMessage").textContent="Video finished. Upload another video or connect a live source.";$("feedMessage").style.display="grid" }
    else if(state==="error"||state==="starting"||state==="queued"){ $("engineState").textContent=state==="error"?"Source error":"Waiting for source";$("feedMessage").textContent=status.message||"Waiting for input source…";$("feedMessage").style.display="grid" }
  }catch(error){
    console.error("Input status API error:",error);
    showSourceMessage("Could not read input status.",true);
  }
}

function renderEvents(){
  const list=$("eventList"),table=$("eventTable");if(!list||!table)return;
  if(!events.length){list.innerHTML=`<div class="empty">No events recorded in this session.</div>`;table.innerHTML=`<tr><td colspan="5" class="empty">No events recorded.</td></tr>`;return}
  list.innerHTML=events.slice(0,8).map(e=>{
    const k=kind(e.event_type),v=e.visitor_id!=null?`Visitor ${escapeHTML(e.visitor_id)}`:"Pipeline";
    return `<div class="event-row"><div class="event-icon ${k}">${symbol(e.event_type)}</div><div class="event-info"><strong>${escapeHTML(e.event_type||"EVENT")}</strong><small>${v}</small></div><span class="event-time">${escapeHTML(shortTime(e.timestamp))}</span></div>`
  }).join("");
  table.innerHTML=events.slice(0,100).map(e=>{
    const k=kind(e.event_type),v=e.visitor_id!=null?`Visitor ${escapeHTML(e.visitor_id)}`:"—";
    return `<tr><td><span class="tag ${k}">${escapeHTML(e.event_type||"EVENT")}</span></td><td>${v}</td><td>${escapeHTML(e.timestamp||"—")}</td><td>${e.image_path?"Image saved":"—"}</td><td>${k==="exit"?"Closed":"Recorded"}</td></tr>`
  }).join("");
  $("eventCountLabel").textContent=`${events.length} record${events.length===1?"":"s"}`;
}

function renderVisitors(filter=""){
  const grid=$("visitorGrid");if(!grid)return;
  const arr=visitors.filter(v=>String(v.id).includes(filter));
  if(!arr.length){grid.innerHTML=`<div class="card empty">No visitors found.</div>`;return}
  grid.innerHTML=arr.map(v=>`<article class="visitor-card" data-id="${escapeHTML(v.id)}"><div class="visitor-photo-wrap">${v.face_image_url?`<img class="visitor-photo" src="${escapeHTML(v.face_image_url)}" alt="Face image for visitor ${escapeHTML(v.id)}" loading="lazy" onerror="this.hidden=true;this.nextElementSibling.hidden=false">`:""}<div class="avatar visitor-photo-placeholder" ${v.face_image_url?'hidden':""}>V${String(v.id).padStart(2,"0")}</div></div><strong>Visitor ${escapeHTML(v.id)}</strong><small>Registered identity</small><div class="seen">● Last seen ${escapeHTML(v.last_seen||"—")}</div></article>`).join("");
  document.querySelectorAll(".visitor-card").forEach(c=>c.onclick=()=>openVisitor(c.dataset.id));
}

function openVisitor(id){
  const v=visitors.find(x=>String(x.id)===String(id));if(!v)return;
  $("modalTitle").textContent=`Visitor ${v.id}`;$("modalAvatar").textContent=String(v.id).padStart(2,"0");
  const photo=$("modalVisitorPhoto"),placeholder=$("modalAvatar");
  photo.hidden=!v.face_image_url;
  placeholder.hidden=Boolean(v.face_image_url);
  photo.alt=`Face image for visitor ${v.id}`;
  photo.onerror=()=>{photo.hidden=true;placeholder.hidden=false};
  if(v.face_image_url)photo.src=v.face_image_url;
  document.querySelector("#visitorModal .detail-grid").innerHTML=`<div><span>First seen</span><strong>${escapeHTML(v.first_seen||"—")}</strong></div><div><span>Last seen</span><strong>${escapeHTML(v.last_seen||"—")}</strong></div><div><span>Visitor ID</span><strong>${escapeHTML(v.id)}</strong></div><div><span>Recognition</span><strong>InsightFace / ArcFace</strong></div>`;
  $("visitorModal").classList.add("show");
}

function renderSummary(s){
  const entries=Number(s.entries||0),exits=Number(s.exits||0),active=Number(s.active_visitors||0);
  $("netOccupancy").textContent=Math.max(0,entries-exits);
  $("exitRate").textContent=`${entries?Math.round(exits/entries*100):0}%`;
  $("registeredCount").textContent=visitors.length;
  const vals=[entries,active,exits,Number(s.unique_visitors||0),Math.max(0,entries-exits),entries+exits],max=Math.max(...vals,1);
  $("activityBars").innerHTML=vals.map((v,i)=>`<div class="bar" style="height:${Math.max(7,v/max*58)}%"><span>${i+1}</span></div>`).join("");
}

async function refreshDashboard(){
  try{
    const [stats,ev,vs]=await Promise.all([fetchJSON("/api/stats"),fetchJSON("/api/events"),fetchJSON("/api/visitors")]);
    latestStats=stats;events=ev||[];visitors=vs||[];
    $("uniqueCount").textContent=stats.unique_visitors??0;$("activeCount").textContent=stats.active_visitors??0;$("entryCount").textContent=stats.entries??0;$("exitCount").textContent=stats.exits??0;
    $("trackSummary").textContent=`${stats.active_visitors??0} active track${Number(stats.active_visitors||0)===1?"":"s"}`;
    renderEvents();renderVisitors($("visitorSearch")?.value.trim()||"");renderSummary(stats);
  }catch(e){console.error("Dashboard API error:",e);$("engineState").textContent="Backend unavailable"}
}

function switchView(view){
  document.querySelectorAll(".view").forEach(v=>v.classList.remove("active-view"));$(view)?.classList.add("active-view");
  document.querySelectorAll(".nav-item").forEach(n=>n.classList.toggle("active",n.dataset.view===view));
  const titles={overview:"Visitor monitoring",visitors:"Visitor directory",events:"Event log",system:"System status"};
  $("pageTitle").textContent=titles[view]||titles.overview;document.querySelector(".crumb").textContent=`Operations / ${titles[view]||titles.overview}`;
}

document.querySelectorAll(".nav-item").forEach(n=>n.onclick=()=>switchView(n.dataset.view));
document.querySelectorAll("[data-view-target]").forEach(b=>b.onclick=()=>switchView(b.dataset.viewTarget));
$("visitorSearch")?.addEventListener("input",e=>renderVisitors(e.target.value.trim()));
$("modalClose")?.addEventListener("click",()=>$("visitorModal").classList.remove("show"));
$("visitorModal")?.addEventListener("click",e=>{if(e.target.id==="visitorModal")e.currentTarget.classList.remove("show")});

$("uploadForm")?.addEventListener("submit",async e=>{
  e.preventDefault();
  const file=$("videoFile").files[0];
  if(!file){showSourceMessage("Choose a video file first.",true);return}
  const button=e.currentTarget.querySelector("button[type=submit]");
  button.disabled=true;showSourceMessage(`Uploading ${file.name}…`);
  try{
    const body=new FormData();body.append("video",file);
    const result=await fetchAPI("/api/upload",{method:"POST",body});
    $("videoFile").value="";
    showSourceMessage(result.message);
    await refreshSourceStatus();
  }catch(error){showSourceMessage(error.message,true)}
  finally{button.disabled=false}
});

$("sourceForm")?.addEventListener("submit",async e=>{
  e.preventDefault();
  const button=e.currentTarget.querySelector("button[type=submit]");
  button.disabled=true;showSourceMessage("Connecting to live input…");
  try{
    const result=await fetchAPI("/api/source",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({source:$("sourceInput").value})
    });
    showSourceMessage(result.message);
    await refreshSourceStatus();
  }catch(error){showSourceMessage(error.message,true)}
  finally{button.disabled=false}
});

$("resetDatabaseButton")?.addEventListener("click",async e=>{
  const button=e.currentTarget;
  const confirmed=window.confirm(
    "Permanently delete all registered face identities, embeddings, entry/exit history, and saved face evidence? The live tracker will start registering visitors again."
  );
  if(!confirmed)return;
  button.disabled=true;
  showResetMessage("Queueing database reset…");
  try{
    const result=await fetchAPI("/api/reset",{method:"POST"});
    showResetMessage(result.message||"Database reset queued.");
    await refreshDashboard();
    await refreshSourceStatus();
  }catch(error){
    showResetMessage(error.message,true);
    button.disabled=false;
  }
});

async function sendBrowserCameraFrames(generation){
  const video=$("browserCameraPreview"),canvas=$("browserCameraCanvas");
  const context=canvas.getContext("2d",{alpha:false});
  while(browserCameraActive&&generation===browserCameraGeneration){
    if(video.readyState>=HTMLMediaElement.HAVE_CURRENT_DATA&&video.videoWidth>0){
      const scale=Math.min(1,1280/video.videoWidth);
      canvas.width=Math.round(video.videoWidth*scale);
      canvas.height=Math.round(video.videoHeight*scale);
      context.drawImage(video,0,0,canvas.width,canvas.height);
      const blob=await new Promise(resolve=>canvas.toBlob(resolve,"image/jpeg",0.78));
      if(blob){
        try{
          await fetchAPI("/api/browser-frame",{
            method:"POST",
            headers:{"Content-Type":"image/jpeg"},
            body:blob
          });
        }catch(error){
          if(browserCameraActive)showSourceMessage(error.message,true);
        }
      }
    }
    await new Promise(resolve=>setTimeout(resolve,90));
  }
}

async function startBrowserCamera(){
  if(!window.isSecureContext){
    throw Error(
      "Browser camera access requires localhost on this computer or trusted HTTPS. A plain HTTP network address is blocked by the browser."
    );
  }
  if(!navigator.mediaDevices?.getUserMedia){
    throw Error("This browser cannot access a camera. Open the dashboard on localhost or use HTTPS.");
  }
  try{
    browserCameraStream=await navigator.mediaDevices.getUserMedia({
      audio:false,
      video:{facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}}
    });
  }catch(error){
    const messages={
      NotAllowedError:"Camera permission was denied. Allow camera access for this site, then try again.",
      NotFoundError:"No camera was found on this device.",
      NotReadableError:"The camera is already in use by another application.",
      OverconstrainedError:"This camera cannot satisfy the requested video settings."
    };
    throw Error(messages[error.name]||`Could not start the device camera: ${error.message||error.name}`);
  }
  const video=$("browserCameraPreview");
  video.srcObject=browserCameraStream;
  video.hidden=false;
  await video.play();
  await fetchAPI("/api/source",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({source:"browser"})
  });
  browserCameraActive=true;
  browserCameraStartedAt=Date.now();
  browserFramesAtStart=latestBrowserFramesReceived;
  browserCameraStream.getVideoTracks().forEach(track=>{
    track.onended=()=>{
      if(browserCameraActive)stopBrowserCamera();
    };
  });
  const generation=++browserCameraGeneration;
  $("browserCameraButton").disabled=true;
  $("stopBrowserCameraButton").hidden=false;
  showSourceMessage("Using this device’s camera. Processing frames from the browser.");
  sendBrowserCameraFrames(generation);
}

async function stopBrowserCamera(){
  browserCameraActive=false;
  browserCameraGeneration++;
  if(browserCameraStream){
    browserCameraStream.getTracks().forEach(track=>track.stop());
    browserCameraStream=null;
  }
  $("browserCameraPreview").srcObject=null;
  $("browserCameraPreview").hidden=true;
  $("browserCameraButton").disabled=false;
  $("stopBrowserCameraButton").hidden=true;
  try{
    await fetchAPI("/api/browser-stop",{method:"POST"});
    showSourceMessage("Device camera stopped.");
    await refreshSourceStatus();
  }catch(error){
    showSourceMessage(error.message,true);
  }
}

$("browserCameraButton")?.addEventListener("click",async()=>{
  $("browserCameraButton").disabled=true;
  showSourceMessage("Requesting permission to use this device’s camera…");
  try{await startBrowserCamera()}
  catch(error){
    $("browserCameraButton").disabled=false;
    if(browserCameraStream)browserCameraStream.getTracks().forEach(track=>track.stop());
    browserCameraStream=null;
    $("browserCameraPreview").srcObject=null;
    $("browserCameraPreview").hidden=true;
    showSourceMessage(error.message,true);
  }
});
$("stopBrowserCameraButton")?.addEventListener("click",stopBrowserCamera);
window.addEventListener("beforeunload",()=>{
  if(browserCameraActive)navigator.sendBeacon("/api/browser-stop");
  browserCameraActive=false;
  if(browserCameraStream)browserCameraStream.getTracks().forEach(track=>track.stop());
});

$("liveFeed").onerror=()=>{$("feedMessage").style.display="grid";$("feedMessage").textContent="Camera stream unavailable"};
function updateClock(){$("clock").textContent=new Date().toLocaleTimeString([],{hour12:false})}

updateClock();refreshDashboard();refreshSourceStatus();setInterval(updateClock,1000);setInterval(refreshDashboard,2000);setInterval(refreshSourceStatus,2000);
