(() => {
  const map = new MapLite.Map('controlMap', {
    center:[10.78,106.69], zoom:12, minZoom:10, maxZoom:18,
    maxBounds:[[10.45,106.20],[11.30,107.30]]
  });

  let selectedVehicleId='V1';
  let fitted=false;
  const roadCache=new Map();
  const roadInFlight=new Set();
  let routingFallback=false;
  let routingUsed=false;
  let lastRoutingProvider='';
  let renderVersion=0;

  const routeKey=(a,b)=>[a[0],a[1],b[0],b[1]].map(v=>Number(v).toFixed(5)).join('|');
  const samePoint=(a,b)=>Math.abs(a[0]-b[0])<1e-7&&Math.abs(a[1]-b[1])<1e-7;
  const haversine=(a,b)=>{
    const R=6371000,rad=Math.PI/180,p1=a[0]*rad,p2=b[0]*rad,dp=(b[0]-a[0])*rad,dl=(b[1]-a[1])*rad;
    const h=Math.sin(dp/2)**2+Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)**2;return 2*R*Math.asin(Math.sqrt(h));
  };
  const pointAlong=(points,progress)=>{
    if(!points?.length)return null;if(points.length===1)return points[0];
    const p=Math.max(0,Math.min(1,Number(progress)||0));const ds=[];let total=0;
    for(let i=1;i<points.length;i++){const d=haversine(points[i-1],points[i]);ds.push(d);total+=d;}
    if(total<=0)return points[points.length-1];let target=total*p,acc=0;
    for(let i=1;i<points.length;i++){const d=ds[i-1];if(acc+d>=target){const f=d?((target-acc)/d):0;return [points[i-1][0]+(points[i][0]-points[i-1][0])*f,points[i-1][1]+(points[i][1]-points[i-1][1])*f];}acc+=d;}
    return points[points.length-1];
  };

  function updateRoutingStatus(){
    const badge=document.getElementById('routingStatus'),mini=document.getElementById('routingMini');
    let text='',cls='route-health';
    if(roadInFlight.size){text='Đang tính tuyến đường…';cls+=' checking';}
    else if(routingUsed&&!routingFallback){text=`Đã kết nối đường bộ${lastRoutingProvider?` · ${lastRoutingProvider}`:''}`;cls+=' good';}
    else if(routingUsed&&routingFallback){text='Một số tuyến đang dùng dự phòng';cls+=' warn';}
    else if(routingFallback){text='Chưa kết nối được định tuyến';cls+=' bad';}
    else{text='Sẵn sàng kiểm tra định tuyến';cls+=' neutral';}
    if(badge){badge.textContent=text;badge.className=cls;}
    if(mini)mini.textContent=text.replace('Đã kết nối đường bộ','Đường bộ sẵn sàng');
  }

  function ensureRoad(a,b){
    if(!a||!b)return null;
    if(samePoint(a,b))return {ok:true,fallback:false,points:[a,b],source:'Cùng địa điểm',distance_m:0,duration_s:0};
    const key=routeKey(a,b);
    if(roadCache.has(key))return roadCache.get(key);
    if(!roadInFlight.has(key)){
      roadInFlight.add(key);updateRoutingStatus();
      Demo.roadRoute(a,b).then(result=>{
        roadInFlight.delete(key);roadCache.set(key,result);routingUsed=true;
        if(result.fallback)routingFallback=true;else lastRoutingProvider=(result.provider||'OSRM').replace(/^https?:\/\//,'').split('/')[0];
        updateRoutingStatus();
        if(Demo.data)renderAll(Demo.data,false);
      }).catch(()=>{roadInFlight.delete(key);routingFallback=true;updateRoutingStatus();});
    }
    return null;
  }
  const routeResult=(a,b)=>roadCache.get(routeKey(a,b))||ensureRoad(a,b);

  function targetForOrder(o){
    if(o.next_operation==='CUSTOMER_PICKUP')return o.pickup;
    if(o.next_operation==='CUSTOMER_DELIVERY')return o.delivery;
    return Demo.data?.depot||null;
  }
  function targetForDecision(data,d){const o=data.orders.find(x=>x.order_id===d.order_id);return o?targetForOrder(o):null;}
  const flowChip=(n)=>`<span class="flow-chip f${n}">F${n}</span>`;
  const sourceChip=(origin)=>`<span class="source-chip ${String(origin||'STATIC').toLowerCase()}">${Demo.origin(origin||'STATIC')}</span>`;
  function activeLegOf(data,vid){return (data.active_legs||[]).find(x=>x.vehicle_id===vid)||null;}
  function visualVehiclePosition(data,v){
    const active=activeLegOf(data,v.vehicle_id);if(!active)return [v.lat,v.lon];
    const a=[active.start_lat,active.start_lon],b=[active.target_lat,active.target_lon],rr=routeResult(a,b);
    return rr?.points?.length>1?(pointAlong(rr.points,active.progress)||[v.lat,v.lon]):[v.lat,v.lon];
  }
  // Hướng đầu xe theo đoạn đường đang chạy; xe đứng yên giữ hướng cũ.
  const lastHeading=new Map();
  function visualVehicleHeading(data,v){
    const active=activeLegOf(data,v.vehicle_id);let h=null;
    if(active){const a=[active.start_lat,active.start_lon],b=[active.target_lat,active.target_lon],rr=routeResult(a,b);h=rr?.points?.length>1?Demo.headingAlong(rr.points,active.progress,pointAlong):Demo.bearing(a,b);}
    if(h==null)h=lastHeading.get(v.vehicle_id)??0;lastHeading.set(v.vehicle_id,h);return h;
  }

  function stopMarkerHtml(n,type,dynamic=false){return `<div class="route-stop-marker ${type} ${dynamic?'dynamic':''}"><b>${n}</b></div>`;}

  function syncMap(data){
    map.clearPolylines();map.clearMarkers();const all=[];
    const depot=[data.depot.lat,data.depot.lon];
    map.addMarker('depot',depot[0],depot[1],'<div class="depot-marker ops-depot">KHO<b>SGNA</b></div>',`${data.depot.name} · ${data.depot.address}`);all.push(depot);

    // Tất cả điểm order hiển thị nhẹ; tuyến xe đang chọn sẽ được đánh số nổi bật phía sau.
    data.orders.filter(o=>!o.cancelled&&!o.completed).forEach(o=>{
      const short=o.order_id.replace('DL-','').replace('PU-','').replace('DY-','M');
      if(['PICKUP_DELIVERY','OUTBOUND'].includes(o.flow_type)){
        const m=map.addMarker(`ord-p-${o.order_id}`,o.pickup.lat,o.pickup.lon,Demo.orderHtml('pickup',short,o.order_class==='DYNAMIC'?'dynamic':''),`${o.order_id} · Lấy hàng · ${o.pickup.address}`);
        m.onclick=()=>{selectedVehicleId=selectedVehicleId||'V1';renderAll(data,false);};all.push([o.pickup.lat,o.pickup.lon]);
      }
      if(['PICKUP_DELIVERY','INBOUND','INBOUND_PRELOADED'].includes(o.flow_type)){
        map.addMarker(`ord-d-${o.order_id}`,o.delivery.lat,o.delivery.lon,Demo.orderHtml('delivery',short),`${o.order_id} · Giao hàng · ${o.delivery.address}`);all.push([o.delivery.lat,o.delivery.lon]);
      }
    });

    // Lịch sử ngắn của từng xe.
    data.vehicles.forEach(v=>{
      if(v.route_history?.length>1){
        const from=Math.max(0,v.route_history.length-5);
        for(let i=from;i<v.route_history.length-1;i++){
          const a=v.route_history[i],b=v.route_history[i+1],rr=routeResult(a,b);
          if(rr?.points?.length>1)map.addPolyline(`hist-${v.vehicle_id}-${i}`,rr.points,{color:v.color,weight:2,opacity:.22,dash:'3 6'});
        }
      }
    });

    // Chặng đang chạy của tất cả xe: line đậm theo màu xe.
    (data.active_legs||[]).forEach(leg=>{
      const v=data.vehicles.find(x=>x.vehicle_id===leg.vehicle_id),a=[leg.start_lat,leg.start_lon],b=[leg.target_lat,leg.target_lon],rr=routeResult(a,b);
      if(rr?.points?.length>1)map.addPolyline(`active-${leg.vehicle_id}`,rr.points,{color:v?.color||'#111827',weight:6,opacity:.92});
    });

    // Biểu tượng xe đi dọc theo hình học đường bộ.
    data.vehicles.forEach(v=>{
      const pos=visualVehiclePosition(data,v);const m=map.addMarker(`veh-${v.vehicle_id}`,pos[0],pos[1],Demo.vehicleHtml(v,visualVehicleHeading(data,v),v.vehicle_id===selectedVehicleId?'selected':''),`${v.vehicle_id} · ${v.zone} · ${Demo.vehicleStatus(v.status)} · ${v.load}/${v.capacity} kg`);
      m.onclick=()=>{selectedVehicleId=v.vehicle_id;document.getElementById('vehicleSelect').value=v.vehicle_id;renderAll(data,false);};all.push(pos);
    });

    // Tuyến còn lại của xe đang chọn.
    const sv=data.vehicles.find(v=>v.vehicle_id===selectedVehicleId)||data.vehicles[0];
    if(sv){
      const active=activeLegOf(data,sv.vehicle_id);let cursor=active?[active.target_lat,active.target_lon]:visualVehiclePosition(data,sv);
      let plan=(sv.operational_plan||[]).slice();
      if(active)plan=plan.filter((x,i)=>!(i===0&&x.order_id===active.order_id));
      plan.slice(0,8).forEach((stop,i)=>{
        const t=[stop.target.lat,stop.target.lon],rr=routeResult(cursor,t);
        if(rr?.points?.length>1)map.addPolyline(`plan-${sv.vehicle_id}-${i}`,rr.points,{color:sv.color,weight:4,opacity:.6,dash:'10 5'});
        const type=stop.operation==='CUSTOMER_PICKUP'?'pickup':'delivery';
        const marker=map.addMarker(`plan-stop-${i}`,t[0],t[1],stopMarkerHtml(i+1,type,stop.dynamic),`${i+1}. ${stop.order_id} · ${Demo.operation(stop.operation)} · ${stop.target.address||''}`);
        marker.onclick=()=>{document.querySelector(`[data-plan-order="${stop.order_id}"]`)?.scrollIntoView({behavior:'smooth',block:'center'});};
        cursor=t;all.push(t);
      });
    }

    // Khuyến nghị mô hình: dashed white/blue overlay đến điểm được đề xuất.
    (data.pending_decisions||[]).forEach(d=>{
      if(d.status!=='OK'||!d.vehicle_id)return;const v=data.vehicles.find(x=>x.vehicle_id===d.vehicle_id),t=targetForDecision(data,d);if(!v||!t)return;
      const a=visualVehiclePosition(data,v),b=[t.lat,t.lon],rr=routeResult(a,b);
      if(rr?.points?.length>1)map.addPolyline(`model-${d.decision_id}`,rr.points,{color:'#0b1220',weight:3,opacity:.85,dash:'2 6'});
    });

    if(!fitted&&all.length){map.fitBounds(all,42);fitted=true;}
    updateRoutingStatus();
  }

  function localizedText(text=''){
    let s=String(text||'');const replacements={
      'CUSTOMER_PICKUP_COMPLETE':'Hoàn tất lấy hàng tại khách','CUSTOMER_DELIVERY_COMPLETE':'Hoàn tất giao hàng cho khách',
      'DEPOT_LOAD_COMPLETE':'Hoàn tất chất hàng tại kho','DEPOT_UNLOAD_COMPLETE':'Hoàn tất dỡ hàng tại kho',
      'VEHICLE_BREAKDOWN':'Xe hỏng','VEHICLE_RECOVERY':'Xe phục hồi','TRAFFIC_UPDATE':'Cập nhật giao thông',
      'NEW_ORDER':'Đơn mới','CANCEL_ORDER':'Hủy đơn','SHIFT_START':'Bắt đầu ca','MANUAL_REPLAN':'Yêu cầu quyết định thủ công',
      'PESDVS Actor':'PESDVS','environment':'môi trường','operation':'tác vụ','location':'địa điểm','argmin accumulated time':'chọn xe có thời gian tích lũy nhỏ nhất'
    };
    for(const [a,b] of Object.entries(replacements))s=s.split(a).join(b);return Demo.validatorNotes(s);
  }

  // ====================================================================================
  // PHÂN TÍCH TUYẾN
  // ====================================================================================
  async function segmentStats(a,b){const rr=routeResult(a,b);if(!rr)return null;return {distanceKm:Number(rr.distance_m||0)/1000,durationMin:Number(rr.duration_s||0)/60,fallback:!!rr.fallback};}

  function renderVehicleSelector(data){
    const sel=document.getElementById('vehicleSelect');
    if(!data.vehicles.some(v=>v.vehicle_id===selectedVehicleId))selectedVehicleId=data.vehicles[0]?.vehicle_id||'';
    const html=data.vehicles.map(v=>`<option value="${v.vehicle_id}" ${v.vehicle_id===selectedVehicleId?'selected':''}>${v.vehicle_id} · ${Demo.escapeHtml(v.zone)} · ${Demo.vehicleStatus(v.status)}</option>`).join('');
    if(sel.innerHTML!==html)sel.innerHTML=html;sel.value=selectedVehicleId;
  }

  function renderRouteAnalysis(data){
    const v=data.vehicles.find(x=>x.vehicle_id===selectedVehicleId)||data.vehicles[0];if(!v)return;
    const active=activeLegOf(data,v.vehicle_id);const plan=(v.operational_plan||[]).slice();
    const summary=document.getElementById('routeSummary');
    summary.innerHTML=`
      <div class="route-summary-title"><div><b>${v.vehicle_id}</b><span>${Demo.escapeHtml(v.zone)}</span></div><span class="status-pill ${v.status==='EN_ROUTE'?'running':v.status==='SCHEDULED'?'scheduled':'idle'}">${Demo.vehicleStatus(v.status)}</span></div>
      <div class="route-metric-grid">
        <div><span>Xuất kho</span><b>${v.departure_clock}</b></div><div><span>Điểm dừng còn lại</span><b>${plan.length}</b></div>
        <div><span>Tải hiện tại</span><b>${Demo.fmt(v.load,0)} kg</b></div><div><span>Sức chứa còn lại</span><b>${Demo.fmt(v.remaining_capacity,0)} kg</b></div>
        <div><span>Quãng đường đã ghi nhận</span><b>${Demo.fmt(v.distance,1)} km</b></div><div><span>Tác vụ hoàn tất</span><b>${v.n_services}</b></div>
      </div>
      <div class="route-total-line"><span id="routeRoadMetric">Đang tính quãng đường còn lại theo mạng đường…</span><span>${active?`Đang tới ${Demo.escapeHtml(active.order_id)}`:'Sẵn sàng cho điểm dừng kế tiếp'}</span></div>`;

    const rail=document.getElementById('routeRail');
    if(!plan.length){rail.innerHTML='<div class="empty">Xe không còn điểm dừng trong kế hoạch hiện tại.</div>';return;}
    rail.innerHTML=plan.map((s,i)=>{
      const opClass=s.operation==='CUSTOMER_PICKUP'?'pickup':'delivery';
      const tags=[`<span class="route-tag flow${s.flow_number||0}">F${s.flow_number||'?' } · ${Demo.escapeHtml(s.flow_short||'')}</span>`,s.onboard?'<span class="route-tag onboard">Hàng trên xe</span>':'',s.origin==='AUTO'?'<span class="route-tag auto">Động tự động</span>':'',s.origin==='MANUAL'?'<span class="route-tag manual">Động thủ công</span>':'',s.committed?'<span class="route-tag committed">Mô hình cam kết</span>':''].join('');
      return `<article class="route-node ${opClass}" data-plan-order="${s.order_id}">
        <div class="route-node-index">${String(i+1).padStart(2,'0')}</div><div class="route-node-line"></div>
        <div class="route-node-main"><div class="route-node-top"><b>${Demo.escapeHtml(s.order_id)}</b><span>${Demo.operation(s.operation)}</span></div>
          <div class="route-node-place">${Demo.escapeHtml(s.target.name||s.target.address||'Điểm phục vụ')}</div>
          <div class="route-node-meta"><span>TW ≤ ${s.latest_clock||'—'}</span><span>${Demo.fmt(s.weight,0)} kg</span><span id="seg-${v.vehicle_id}-${i}">đang tính đoạn đường</span></div>
          <div class="route-node-tags">${tags}</div>
        </div></article>`;
    }).join('');

    const version=++renderVersion;
    (async()=>{
      let cursor=active?[active.target_lat,active.target_lon]:visualVehiclePosition(data,v),totalKm=0,totalMin=0,used=0;
      let effective=plan;if(active&&effective[0]?.order_id===active.order_id)effective=effective.slice(1);
      for(let i=0;i<effective.length;i++){
        const s=effective[i],t=[s.target.lat,s.target.lon],rr=routeResult(cursor,t);if(!rr){ensureRoad(cursor,t);cursor=t;continue;}
        const km=Number(rr.distance_m||0)/1000,min=Number(rr.duration_s||0)/60;totalKm+=km;totalMin+=min;used++;
        const originalIndex=plan.indexOf(s),el=document.getElementById(`seg-${v.vehicle_id}-${originalIndex}`);
        if(el)el.textContent=rr.fallback?'chờ tuyến đường bộ':`${km.toFixed(1)} km · ~${Math.round(min)} phút`;
        cursor=t;
      }
      if(version!==renderVersion)return;
      const metric=document.getElementById('routeRoadMetric');if(metric&&used)metric.textContent=`Còn khoảng ${totalKm.toFixed(1)} km · ${Math.round(totalMin)} phút theo tuyến đường bộ`;
    })();
  }

  async function renderPickupAnalysis(data){
    const box=document.getElementById('pickupAnalysis'),count=document.getElementById('pickupCount');
    const v=data.vehicles.find(x=>x.vehicle_id===selectedVehicleId)||data.vehicles[0];if(!v)return;
    const active=activeLegOf(data,v.vehicle_id);const base=active?[active.target_lat,active.target_lon]:visualVehiclePosition(data,v);
    let plan=(v.operational_plan||[]).slice();if(active&&plan[0]?.order_id===active.order_id)plan=plan.slice(1);
    const next=plan[0]?.target?[plan[0].target.lat,plan[0].target.lon]:null;
    const candidates=data.orders.filter(o=>o.released&&!o.cancelled&&!o.completed&&o.next_operation==='CUSTOMER_PICKUP');
    const rows=candidates.map(o=>({o,air:haversine(base,[o.pickup.lat,o.pickup.lon])})).sort((a,b)=>a.air-b.air).slice(0,4);
    count.textContent=`${rows.length} điểm`;
    if(!rows.length){box.innerHTML='<div class="empty">Không có đơn lấy hàng nào đang chờ trong trạng thái hệ thống.</div>';return;}
    const modelMap=new Map((data.pending_decisions||[]).map(d=>[`${d.vehicle_id}|${d.order_id}`,d]));
    box.innerHTML=rows.map(({o,air},i)=>{
      const feasibleLoad=v.load+o.weight<=v.capacity+1e-9,chosen=modelMap.has(`${v.vehicle_id}|${o.order_id}`);
      return `<article class="pickup-card ${chosen?'model-chosen':''}" id="pickup-${o.order_id}">
        <div class="pickup-card-head"><div><b>${Demo.escapeHtml(o.order_id)}</b><span>${Demo.escapeHtml(o.pickup.name||o.pickup.address)}</span></div>${chosen?'<span class="model-flag">Mô hình đang chọn</span>':'<span class="mini-badge">Đang phân tích</span>'}</div>
        <div class="pickup-metrics"><div><span>Cách xe/điểm kế tiếp</span><b>${(air/1000).toFixed(1)} km</b></div><div><span>Tải sau khi lấy</span><b class="${feasibleLoad?'ok-text':'bad-text'}">${Demo.fmt(v.load+o.weight,0)} / ${Demo.fmt(v.capacity,0)} kg</b></div><div><span>Cửa sổ lấy hàng</span><b>${o.pickup.begin_clock||'—'}–${o.pickup.end_clock||'—'}</b></div><div><span>Độ đi vòng theo đường bộ</span><b id="detour-${o.order_id}">đang tính…</b></div></div>
        <div class="pickup-verdict ${chosen?'chosen':feasibleLoad?'candidate':'reject'}">${chosen?'Khuyến nghị mô hình: chen điểm lấy hàng vào phần tuyến còn lại':feasibleLoad?'Khả thi về tải; đang đánh giá độ đi vòng/cửa sổ thời gian':'Không khả thi do tải trọng'}</div>
      </article>`;
    }).join('');

    for(const {o} of rows){
      if(v.load+o.weight>v.capacity+1e-9)continue;
      const p=[o.pickup.lat,o.pickup.lon],r1=routeResult(base,p);if(!r1){ensureRoad(base,p);continue;}
      let detour=Number(r1.distance_m||0)/1000;
      if(next){const r2=routeResult(p,next),r0=routeResult(base,next);if(!r2){ensureRoad(p,next);}if(!r0){ensureRoad(base,next);}if(r2&&r0)detour=Math.max(0,(Number(r1.distance_m||0)+Number(r2.distance_m||0)-Number(r0.distance_m||0))/1000);}
      const el=document.getElementById(`detour-${o.order_id}`);if(el)el.textContent=r1.fallback?'chưa có tuyến':'+'+detour.toFixed(1)+' km';
    }
  }

  function renderFlowBoard(data){
    const fs=data.flow_summary||{};
    for(const n of [1,2,3]){
      const x=fs[String(n)]||{};
      const set=(id,val)=>{const el=document.getElementById(id);if(el)el.textContent=val;};
      set(`f${n}Active`,x.active??0);set(`f${n}Done`,x.completed??0);set(`f${n}Mix`,`${x.static??0} / ${x.dynamic??0}`);set(`f${n}Onboard`,x.onboard??0);
    }
    const ad=data.auto_dynamic||{};
    const state=document.getElementById('autoDemandState'),next=document.getElementById('autoDemandNext');
    if(state)state.innerHTML=`<span class="auto-pulse"></span> Đơn động tự động ${ad.enabled?'đang BẬT':'đang TẮT'} · đã phát sinh ${ad.released??0}/${ad.scheduled_total??0}`;
    if(next)next.textContent=ad.remaining?`Lần phát sinh kế tiếp khoảng ${ad.next_release_clock||'—'} · còn ${ad.remaining} order trong lịch mô phỏng`:'Đã phát sinh hết lịch nhu cầu động của ca';
  }

  // ====================================================================================
  // DECISION / VEHICLES / ORDERS
  // ====================================================================================
  function decisionCard(d,data){
    const ok=d.status==='OK'&&d.feasible,active=activeLegOf(data,d.vehicle_id),queued=!!active;
    return `<article class="ops-decision ${ok?'':'error'}">
      <div class="ops-decision-head"><div><span>${Demo.eventType(d.replan_reason)||localizedText(d.replan_reason||'Quyết định trực tuyến')}</span><b>${Demo.escapeHtml(d.decision_id)}</b></div><span class="status-pill ${ok?'running':'error'}">${Demo.decisionStatus(d.status)}</span></div>
      ${ok?`<div class="ops-decision-pair"><b>${Demo.escapeHtml(d.vehicle_id)}</b><span>→</span><b>${Demo.escapeHtml(d.order_id)}</b></div>
      <div class="ops-decision-op">${Demo.operation(d.operation)}<small>${Demo.flow(d.flow_type)}</small></div>
      <div class="ops-decision-grid"><div><span>ETA</span><b>${d.expected_arrival_clock||'—'}</b></div><div><span>Hoàn tất</span><b>${d.expected_completion_clock||'—'}</b></div><div><span>Xác suất</span><b>${Demo.fmt(d.policy_probability,3)}</b></div><div><span>Độ trễ</span><b>${Demo.fmt(d.decision_latency_ms,1)} ms</b></div></div>
      <div class="ops-decision-note"><b>Xe khả thi:</b> ${Demo.escapeHtml(d.k_tilde||'—')}<br>${Demo.escapeHtml(Demo.validatorNotes(d.validator_notes||''))}</div>
      <button class="btn btn-primary full" data-execute="${d.decision_id}">${queued?'Chấp nhận cho ranh giới kế tiếp':'Thực hiện tác vụ đề xuất'}</button>`:
      `<div class="ops-decision-note">${Demo.escapeHtml(Demo.validatorNotes(d.validator_notes||'Không có tác vụ khả thi.'))}</div>`}
    </article>`;
  }
  function renderDecisions(data){
    const box=document.getElementById('decisions'),arr=data.pending_decisions||[];box.innerHTML=arr.length?arr.map(d=>decisionCard(d,data)).join(''):'<div class="empty">Không có khuyến nghị đang chờ. Xe tiếp tục phần tuyến hiện tại.</div>';
    box.querySelectorAll('[data-execute]').forEach(b=>b.addEventListener('click',async()=>{b.disabled=true;try{await Demo.api('/api/decision/execute','POST',{decision_id:b.dataset.execute});}catch(e){alert(e.message)}finally{b.disabled=false;}}));
  }

  function renderVehicles(data){
    const box=document.getElementById('vehicleList');box.innerHTML=data.vehicles.map(v=>{
      const planCount=v.operational_plan?.length||0,active=activeLegOf(data,v.vehicle_id);
      return `<article class="ops-vehicle-card ${v.vehicle_id===selectedVehicleId?'selected':''} ${v.broken?'broken':''}" data-vehicle-card="${v.vehicle_id}">
        <div class="ops-vehicle-head"><div><i style="background:${v.color}"></i><b>${v.vehicle_id}</b><span>${Demo.vehicleType(v.vehicle_type)}</span></div><span class="status-pill ${v.status==='EN_ROUTE'?'running':v.status==='SCHEDULED'?'scheduled':v.broken?'error':'idle'}">${Demo.vehicleStatus(v.status)}</span></div>
        <div class="ops-zone">${Demo.escapeHtml(v.zone)} · xuất kho ${v.departure_clock}${active?` · đang tới ${Demo.escapeHtml(active.order_id)}`:''}</div>
        <div class="load-row"><span>Tải ${Demo.fmt(v.load,0)} / ${Demo.fmt(v.capacity,0)} kg</span><b>${Demo.fmt(v.load_pct,0)}%</b></div><div class="loadbar"><i style="width:${Math.min(100,v.load_pct)}%;background:${v.color}"></i></div>
        <div class="ops-vehicle-stats"><span><b>${planCount}</b> điểm dừng còn lại</span><span><b>${v.n_services}</b> đã hoàn tất</span><span><b>${Demo.fmt(v.distance,1)}</b> km</span></div>
        <div class="ops-vehicle-actions">${v.broken?`<button class="btn btn-secondary btn-small" data-recover="${v.vehicle_id}">Phục hồi</button>`:`<button class="btn btn-ghost btn-small" data-break="${v.vehicle_id}">Mô phỏng hỏng xe</button>`}</div>
      </article>`;
    }).join('');
    box.querySelectorAll('[data-vehicle-card]').forEach(c=>c.addEventListener('click',e=>{if(e.target.closest('button'))return;selectedVehicleId=c.dataset.vehicleCard;document.getElementById('vehicleSelect').value=selectedVehicleId;renderAll(data,false);}));
    box.querySelectorAll('[data-break]').forEach(b=>b.addEventListener('click',async()=>{try{await Demo.api(`/api/events/breakdown/${b.dataset.break}`,'POST')}catch(e){alert(e.message)}}));
    box.querySelectorAll('[data-recover]').forEach(b=>b.addEventListener('click',async()=>{try{await Demo.api(`/api/events/recovery/${b.dataset.recover}`,'POST')}catch(e){alert(e.message)}}));
  }

  function renderOrders(data){
    const body=document.getElementById('ordersBody');body.innerHTML=data.orders.map(o=>{
      const owner=o.assigned_vehicle_id?`Trên ${o.assigned_vehicle_id}`:(o.committed_vehicle_id?`Cam kết ${o.committed_vehicle_id}`:'Chưa gán');
      const flowNo=o.flow_number||'?';
      return `<tr class="${o.completed?'done-row':''}"><td><b>${Demo.escapeHtml(o.order_id)}</b><small>${Demo.orderClass(o.order_class)}</small></td><td>${flowChip(flowNo)}<small>${Demo.escapeHtml(o.flow_short||Demo.flow(o.flow_type))}</small></td><td>${o.completed?'Đã hoàn tất':Demo.operation(o.next_operation)}<small>${Demo.escapeHtml(targetForOrder(o)?.name||'')}</small></td><td>${Demo.escapeHtml(owner)}</td><td>${o.latest_delivery_clock?`≤ ${o.latest_delivery_clock}`:'—'}${o.late?`<small class="late">Trễ +${Demo.fmt(o.lateness,0)} phút</small>`:''}</td><td>${Demo.fmt(o.weight,0)} kg</td><td>${sourceChip(o.origin)}</td></tr>`;
    }).join('');
  }

  function renderEvents(data){
    const e=document.getElementById('events');e.innerHTML=data.events.length?data.events.slice(0,28).map(x=>`<div class="ops-log-row"><span>${x.clock}</span><b>${Demo.eventKind(x.kind)}</b><p>${Demo.escapeHtml(localizedText(x.text))}</p></div>`).join(''):'<div class="empty">Chưa có sự kiện.</div>';
  }

  function renderKpis(data){
    document.getElementById('kEnRoute').textContent=data.summary.vehicles_en_route;
    document.getElementById('kDeparted').textContent=`${data.summary.vehicles_departed} / ${data.vehicles.length} xe đã xuất kho`;
    document.getElementById('kActive').textContent=data.summary.active;document.getElementById('kOnboard').textContent=`${data.summary.onboard} đơn có hàng trên xe`;
    document.getElementById('kCompleted').textContent=data.summary.completed;document.getElementById('kLate').textContent=`${data.summary.late} đơn trễ`;
    document.getElementById('kReplans').textContent=data.summary.replans;document.getElementById('kDecisions').textContent=`${data.summary.decisions} quyết định`;
    document.getElementById('kActivity').textContent=`${Demo.fmt(data.summary.vehicle_activity_pct,0)}%`;
    document.getElementById('kDemandMix').textContent=`${data.summary.static_orders||0} tĩnh · ${data.summary.dynamic_orders||0} động`;
    document.getElementById('kTraffic').textContent=`${Number(data.summary.traffic_factor).toFixed(2)}×`;
    document.getElementById('modelCheckpoint').textContent=data.model.checkpoint_loaded?'Đã nạp trọng số':'Chưa nạp trọng số';document.getElementById('modelState').textContent=`${data.model.checkpoint} · Z${data.model.z_dim}/U${data.model.u_dim}/D${data.model.d_dim}`;
    document.getElementById('scenarioSeed').textContent=`Mã ngẫu nhiên ${data.scenario.seed}`;
    document.getElementById('btnStart').disabled=!!data.clock.running;document.getElementById('btnPause').disabled=!data.clock.running;
    const autoBtn=document.getElementById('btnAutoDynamic');if(autoBtn){const on=!!data.auto_dynamic?.enabled;autoBtn.textContent=`Đơn động tự động: ${on?'BẬT':'TẮT'}`;autoBtn.classList.toggle('off',!on);autoBtn.classList.toggle('on',on);}
  }

  // ====================================================================================
  // BIỂU ĐỒ DÒNG THỜI GIAN
  // ====================================================================================
  function drawTimeline(data){
    const canvas=document.getElementById('opsTimeline'),wrap=canvas.parentElement,dpr=window.devicePixelRatio||1,w=Math.max(780,wrap.clientWidth),h=260;
    canvas.width=w*dpr;canvas.height=h*dpr;canvas.style.width=w+'px';canvas.style.height=h+'px';const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
    const left=72,right=22,top=28,bottom=28,plotW=w-left-right,laneH=(h-top-bottom)/data.vehicles.length;
    const now=Number(data.clock.minutes),start=475,end=Math.max(720,Math.ceil((now+45)/30)*30),x=t=>left+(t-start)/(end-start)*plotW;
    ctx.font='11px system-ui';ctx.fillStyle='#667085';ctx.strokeStyle='#e4e7ec';ctx.lineWidth=1;
    for(let t=480;t<=end;t+=30){const xx=x(t);ctx.beginPath();ctx.moveTo(xx,top);ctx.lineTo(xx,h-bottom);ctx.stroke();ctx.fillText(Demo.minutesToClock(t),xx-16,h-8);}
    data.vehicles.forEach((v,i)=>{const y=top+i*laneH+laneH/2;ctx.fillStyle='#101828';ctx.font='600 11px system-ui';ctx.fillText(v.vehicle_id,16,y+4);ctx.fillStyle='#98a2b3';ctx.font='9px system-ui';ctx.fillText(v.zone,16,y+16);ctx.strokeStyle='#f2f4f7';ctx.beginPath();ctx.moveTo(left,y+laneH/2-2);ctx.lineTo(w-right,y+laneH/2-2);ctx.stroke();
      const xd=x(v.departure_time);ctx.fillStyle=v.color;ctx.beginPath();ctx.moveTo(xd,y-10);ctx.lineTo(xd-5,y-18);ctx.lineTo(xd+5,y-18);ctx.closePath();ctx.fill();
      if(!v.departed){ctx.fillStyle='#98a2b3';ctx.fillRect(xd,y-4,Math.max(2,x(now)-xd),8);}
    });
    (data.operation_history||[]).forEach(seg=>{const i=data.vehicles.findIndex(v=>v.vehicle_id===seg.vehicle_id);if(i<0)return;const y=top+i*laneH+laneH/2;const v=data.vehicles[i],x1=x(seg.start_time),x2=x(seg.completed?seg.completion_time:Math.min(now,seg.completion_time));ctx.fillStyle=seg.completed?v.color:'#0b1220';ctx.globalAlpha=seg.completed?.62:.95;ctx.fillRect(x1,y-6,Math.max(3,x2-x1),12);ctx.globalAlpha=1;if(x2-x1>45){ctx.fillStyle='#fff';ctx.font='9px system-ui';ctx.fillText(seg.order_id,x1+5,y+3);}});
    const nx=x(now);ctx.strokeStyle='#111827';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(nx,top-8);ctx.lineTo(nx,h-bottom);ctx.stroke();ctx.fillStyle='#111827';ctx.font='700 10px system-ui';ctx.fillText(`BÂY GIỜ ${data.clock.clock}`,Math.min(w-105,nx+5),top-12);
  }

  function renderAll(data,fit=true){renderVehicleSelector(data);syncMap(data);renderFlowBoard(data);renderRouteAnalysis(data);renderPickupAnalysis(data);renderDecisions(data);renderVehicles(data);renderOrders(data);renderEvents(data);renderKpis(data);drawTimeline(data);}

  // Guide
  const guide=document.getElementById('guideOverlay');const showGuide=()=>{guide.classList.remove('guide-hidden');document.body.classList.add('guide-open');};const hideGuide=()=>{guide.classList.add('guide-hidden');document.body.classList.remove('guide-open');};
  document.getElementById('btnEnterGuide').addEventListener('click',hideGuide);document.getElementById('btnGuide').addEventListener('click',showGuide);showGuide();

  // Controls
  document.getElementById('vehicleSelect').addEventListener('change',e=>{selectedVehicleId=e.target.value;if(Demo.data)renderAll(Demo.data,false);});
  document.getElementById('btnStart').addEventListener('click',async()=>{try{await Demo.api('/api/simulation/start','POST')}catch(e){alert(e.message)}});
  document.getElementById('btnPause').addEventListener('click',async()=>{try{await Demo.api('/api/simulation/pause','POST')}catch(e){alert(e.message)}});
  document.getElementById('btnAutoDynamic').addEventListener('click',async()=>{try{const enabled=!Boolean(Demo.data?.auto_dynamic?.enabled);await Demo.api('/api/simulation/auto-dynamic','POST',{enabled});}catch(e){alert(e.message)}});
  document.getElementById('speedSelect').addEventListener('change',async e=>{try{await Demo.api('/api/simulation/speed','POST',{minutes_per_tick:Number(e.target.value)})}catch(err){alert(err.message)}});
  document.getElementById('btnNewScenario').addEventListener('click',async()=>{if(confirm('Tạo một ca mô phỏng mới với giờ xuất phát ngẫu nhiên mới?')){fitted=false;roadCache.clear();Demo.clearRouteCache();await Demo.api('/api/simulation/new','POST');}});
  document.getElementById('btnReplan').addEventListener('click',async()=>{try{await Demo.api('/api/decision/replan','POST')}catch(e){alert(e.message)}});
  document.querySelectorAll('[data-traffic]').forEach(b=>b.addEventListener('click',async()=>{try{await Demo.api('/api/events/traffic','POST',{factor:Number(b.dataset.traffic),area:'ALL'})}catch(e){alert(e.message)}}));
  document.getElementById('btnRoutingTest').addEventListener('click',async()=>{
    const b=document.getElementById('btnRoutingTest');b.disabled=true;b.textContent='Đang kiểm tra…';roadCache.clear();routingFallback=false;routingUsed=false;lastRoutingProvider='';updateRoutingStatus();
    try{const r=await Demo.routingDiagnostic();routingUsed=true;routingFallback=!!r.fallback;if(!r.fallback)lastRoutingProvider=(r.provider||'OSRM').replace(/^https?:\/\//,'').split('/')[0];alert(r.fallback?`Chưa kết nối được tuyến đường bộ.\n${r.warning||''}`:`Kết nối thành công.\nNguồn: ${r.provider}\nTuyến kiểm tra: ${(r.distance_m/1000).toFixed(1)} km`);}catch(e){alert(e.message)}finally{b.disabled=false;b.textContent='Kiểm tra định tuyến';updateRoutingStatus();if(Demo.data)renderAll(Demo.data,false);}
  });
  window.addEventListener('resize',()=>{if(Demo.data)drawTimeline(Demo.data)});

  Demo.connect(data=>renderAll(data));
})();
