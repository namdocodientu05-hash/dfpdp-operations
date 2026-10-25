(() => {
'use strict';
const $ = id => document.getElementById(id);
const fmt = n => Number(n||0).toLocaleString('vi-VN',{maximumFractionDigits:1});
const pct = (a,b=100) => `${(100*Number(a||0)/Math.max(1,Number(b||1))).toFixed(1)}%`;
const colorMap = {completed:'#22c55e', late:'#ef4444', active:'#2563eb', cancelled:'#94a3b8', future:'#f59e0b'};
let currentState = null;

function statusText(run){return ({completed:'Hoàn tất toàn bộ',horizon_end:'Kết thúc horizon',running:'Đang chạy',paused:'Tạm dừng',ready:'Sẵn sàng',blocked:'Cần xử lý',waiting:'Chờ đơn'}[run] || run || '—');}
function orderStatus(o){return o.cancelled?'Đã hủy':o.completed?'Hoàn tất':o.late?'Hoàn tất trễ':o.released?'Đang hoạt động':'Chưa phát sinh';}

function buildSummary(s){
  const orders=s.orders||[], veh=s.vehicles||[], tel=(s.analytics?.telemetry||[]).slice(-72);
  const completed=orders.filter(o=>o.completed).length;
  const cancelled=orders.filter(o=>o.cancelled).length;
  const late=orders.filter(o=>o.late).length;
  const active=orders.filter(o=>o.released && !o.completed && !o.cancelled).length;
  const future=s.auto_dynamic?.remaining||0;
  const total=s.run?.total||orders.length+future;
  const dynamic=orders.filter(o=>o.order_class==='DYNAMIC').length;
  const statics=orders.filter(o=>o.order_class==='STATIC').length;
  const byFlow={};
  for(const o of orders){
    const key=`F${o.flow_number} · ${o.flow_short}`;
    const x=byFlow[key] ||= {name:key,total:0,completed:0,late:0,active:0};
    x.total++; if(o.completed)x.completed++; if(o.late)x.late++; if(o.released&&!o.completed&&!o.cancelled)x.active++;
  }
  const vehicleStats=veh.map(v=>({id:v.label||v.vehicle_id,distance:Number(v.distance||0),services:Number(v.n_services||0),activity:Number(v.travel_time||0)+Number(v.waiting_time||0)})).sort((a,b)=>b.distance-a.distance);
  const daily=(s.calendar?.daily_summaries||[]).map(x=>({...x}));
  if(!daily.find(x=>x.day_number===s.clock?.day_number)) daily.push({day_number:s.clock?.day_number,completed_cumulative:completed,late_cumulative:late,carryover_count:s.summary?.carryover||0,distance_km:s.run?.distance_km||0});
  daily.sort((a,b)=>a.day_number-b.day_number);
  const trafficAvg = tel.length ? tel.reduce((a,b)=>a+Number(b.traffic_speed_kmh||0),0)/tel.length : Number(s.summary?.traffic_speed_kmh||0);
  return {orders,veh,tel,total,completed,cancelled,late,active,future,dynamic,statics,byFlow:Object.values(byFlow),vehicleStats,daily,trafficAvg};
}

function renderKpis(s,d){
  const cards=[
    ['Tỷ lệ hoàn tất', pct(d.completed,d.total), `${d.completed}/${d.total} đơn`, 100*d.completed/Math.max(1,d.total)],
    ['Đơn giao trễ', fmt(d.late), `${pct(d.late,d.total)} trên tổng đơn`, 100*d.late/Math.max(1,d.total)],
    ['Carry-over cuối kỳ', fmt(s.summary?.carryover), 'đơn chưa hoàn tất chuyển ngày / tồn cuối horizon', 100*(s.summary?.carryover||0)/Math.max(1,d.total)],
    ['Quãng đường tích lũy', `${fmt(s.run?.distance_km)} km`, `${s.vehicles.length} xe đang được theo dõi`, Math.min(100,(s.run?.distance_km||0)/(d.total*5||1)*100)],
    ['Mức hoạt động đội xe', pct(s.summary?.vehicle_activity_pct,100), `${s.summary?.vehicles_en_route||0}/${s.vehicles.length} xe đang chạy`, s.summary?.vehicle_activity_pct||0],
    ['Traffic trung bình', `${fmt(d.trafficAvg)} km/h`, `vùng ${s.traffic_profile?.selected?.region_label||'—'}`, Math.min(100,d.trafficAvg)]
  ];
  $('kpiGrid').innerHTML=cards.map(([name,val,desc,p])=>`<article class="card kpi-card"><span>${name}</span><b>${val}</b><small>${desc}</small><div class="kpi-bar"><i style="width:${Math.max(4,Math.min(100,p||0))}%"></i></div></article>`).join('');
}

function donutSegment(cx,cy,r0,r1,a0,a1,color){
  const polar=(a,r)=>[cx+r*Math.cos(a),cy+r*Math.sin(a)];
  const [x0,y0]=polar(a0,r1), [x1,y1]=polar(a1,r1), [x2,y2]=polar(a1,r0), [x3,y3]=polar(a0,r0);
  const large=(a1-a0)>Math.PI?1:0;
  return `<path d="M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r1} ${r1} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)} L ${x2.toFixed(2)} ${y2.toFixed(2)} A ${r0} ${r0} 0 ${large} 0 ${x3.toFixed(2)} ${y3.toFixed(2)} Z" fill="${color}"/>`;
}
function renderStatus(s,d){
  const parts=[['Hoàn tất',d.completed,'completed'],['Trễ',d.late,'late'],['Đang hoạt động',d.active,'active'],['Đã hủy',d.cancelled,'cancelled'],['Chưa phát sinh',d.future,'future']].filter(x=>x[1]>0);
  const total=Math.max(1,parts.reduce((a,b)=>a+b[1],0));
  let a=-Math.PI/2, svg=`<rect x="0" y="0" width="280" height="220" rx="18" fill="#fbfdff"/>`;
  for(const [label,val,key] of parts){const ang=2*Math.PI*(val/total); svg+=donutSegment(110,110,46,76,a,a+ang,colorMap[key]); a+=ang;}
  svg+=`<circle cx="110" cy="110" r="40" fill="#fff"/><text x="110" y="102" text-anchor="middle" class="chart-title">Hoàn tất</text><text x="110" y="126" text-anchor="middle" class="chart-val" style="font-size:24px">${pct(d.completed,d.total)}</text><text x="110" y="146" text-anchor="middle" class="chart-title">${d.completed}/${d.total} đơn</text>`;
  $('statusDonut').innerHTML=svg;
  $('statusLegend').innerHTML=parts.map(([label,val,key])=>`<div class="legend-row"><i style="background:${colorMap[key]}"></i><span>${label}</span><b>${fmt(val)} · ${pct(val,total)}</b></div>`).join('');
}

function renderInsights(s,d){
  const items=[];
  const completion=d.completed/Math.max(1,d.total);
  items.push({t:'Mức hoàn tất',b: completion>=0.9 ? 'Kịch bản cho kết quả tốt, phần lớn đơn đã hoàn tất trong horizon mô phỏng.' : completion>=0.75 ? 'Tỷ lệ hoàn tất ở mức khá nhưng vẫn còn dư địa tối ưu.' : 'Tỷ lệ hoàn tất còn thấp, cần xem lại quy mô đội xe hoặc logic điều phối.'});
  items.push({t:'Áp lực cuối ngày / cuối horizon',b:(s.summary?.carryover||0)>0 ? `Còn ${s.summary?.carryover} đơn carry-over/tồn cuối horizon, cho thấy hệ thống chịu áp lực tại các khung giờ cao điểm hoặc thiếu năng lực phục vụ.` : 'Không còn carry-over cuối kỳ, cho thấy luồng điều phối được hấp thụ tốt trong thời lượng mô phỏng.'});
  items.push({t:'Chất lượng phục vụ',b:d.late>0 ? `Có ${d.late} đơn trễ (${pct(d.late,d.total)}). Nên xem thêm các đơn trong bảng “Đơn cần chú ý” để xác định vùng/ngày có nguy cơ.` : 'Không ghi nhận đơn trễ, mức độ đáp ứng time window đang rất tích cực.'});
  items.push({t:'Gợi ý vận hành',b:(s.summary?.vehicle_activity_pct||0)>85 ? 'Đội xe đang khai thác rất cao; nên cân nhắc tăng xe hoặc giảm demand động nếu muốn giảm rủi ro trễ.' : (s.summary?.vehicle_activity_pct||0)<45 ? 'Đội xe chưa khai thác hết công suất; có thể tăng đơn hoặc giảm số xe để cải thiện hiệu quả tài nguyên.' : 'Mức sử dụng đội xe tương đối cân bằng; trọng tâm tiếp theo là tối ưu traffic và giảm carry-over.'});
  $('insightCards').innerHTML=items.map(x=>`<div class="insight-card"><b>${x.t}</b><p>${x.b}</p></div>`).join('');
}

function renderBarChart(svgId, rows, opts={}){
  const el=$(svgId); if(!el) return;
  const width=svgId==='vehicleChart'||svgId==='telemetryMiniChart'?420:760, height=290, pad={l:54,r:20,t:18,b:42};
  const innerW=width-pad.l-pad.r, innerH=height-pad.t-pad.b;
  const max=Math.max(1,...rows.map(r=>r.value));
  const step=innerW/Math.max(1,rows.length), barW=step*.62;
  let out=`<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#ffffff"/>`;
  for(let g=0;g<=4;g++){const y=pad.t+innerH*g/4; const val=((4-g)/4*max); out+=`<line x1="${pad.l}" y1="${y}" x2="${width-pad.r}" y2="${y}" stroke="#e2e8f0" stroke-dasharray="5 5"/><text x="${pad.l-8}" y="${y+4}" text-anchor="end" class="axis-txt">${val.toFixed(0)}</text>`;}
  rows.forEach((r,i)=>{const h=innerH*(r.value/max); const x=pad.l+step*i+(step-barW)/2, y=pad.t+innerH-h; out+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="12" fill="${r.color||'#2563eb'}" opacity="0.95"/>`; if(r.value2!=null){const h2=innerH*(r.value2/max); const y2=pad.t+innerH-h2; out+=`<rect x="${(x+barW*.22).toFixed(1)}" y="${y2.toFixed(1)}" width="${(barW*.56).toFixed(1)}" height="${h2.toFixed(1)}" rx="10" fill="#f8fafc" opacity="0.85"/>`;} out+=`<text x="${(x+barW/2).toFixed(1)}" y="${(y-7).toFixed(1)}" text-anchor="middle" class="chart-val">${fmt(r.value)}</text><text x="${(x+barW/2).toFixed(1)}" y="${height-12}" text-anchor="middle" class="axis-txt">${r.label}</text>`;});
  el.innerHTML=out;
}

function renderFlow(d){
  const colors=['#2563eb','#14b8a6','#f59e0b','#7c3aed'];
  const rows=d.byFlow.map((x,i)=>({label:x.name.replace(' · ','\n'),value:x.total,color:colors[i%colors.length],value2:x.completed}));
  renderBarChart('flowChart', rows);
}
function renderVehicles(d){
  const rows=d.vehicleStats.slice(0,8).map((x,i)=>({label:x.id.replace('Xe ','X'),value:x.distance,color:['#1d4ed8','#14b8a6','#8b5cf6','#f59e0b','#ef4444','#0ea5e9','#22c55e','#64748b'][i%8]}));
  renderBarChart('vehicleChart', rows);
}

function renderDaily(d){
  const el=$('dailyChart'); const rows=d.daily; const width=760,height=290,pad={l:50,r:26,t:18,b:42}; const innerW=width-pad.l-pad.r, innerH=height-pad.t-pad.b; if(!rows.length){el.innerHTML='';return;} const max=Math.max(1,...rows.map(r=>Math.max(r.completed_cumulative||0,r.late_cumulative||0,r.carryover_count||0))); const xs=i=>pad.l+(innerW*(rows.length===1?0:i/(rows.length-1))); const y=v=>pad.t+innerH-innerH*v/max; const path=(arr,col)=>`<path d="${arr.map((v,i)=>`${i?'L':'M'} ${xs(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')}" fill="none" stroke="${col}" stroke-width="3" stroke-linecap="round"/>`;
  let out=`<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#fff"/>`; for(let g=0;g<=4;g++){const yy=pad.t+innerH*g/4; out+=`<line x1="${pad.l}" y1="${yy}" x2="${width-pad.r}" y2="${yy}" stroke="#e2e8f0" stroke-dasharray="5 5"/><text x="${pad.l-8}" y="${yy+4}" text-anchor="end" class="axis-txt">${((4-g)/4*max).toFixed(0)}</text>`;}
  out+=path(rows.map(r=>r.completed_cumulative||0),'#22c55e'); out+=path(rows.map(r=>r.late_cumulative||0),'#ef4444'); out+=path(rows.map(r=>r.carryover_count||0),'#f59e0b');
  rows.forEach((r,i)=>{const x=xs(i); out+=`<circle cx="${x.toFixed(1)}" cy="${y(r.completed_cumulative||0).toFixed(1)}" r="4" fill="#22c55e"/><circle cx="${x.toFixed(1)}" cy="${y(r.late_cumulative||0).toFixed(1)}" r="4" fill="#ef4444"/><circle cx="${x.toFixed(1)}" cy="${y(r.carryover_count||0).toFixed(1)}" r="4" fill="#f59e0b"/><text x="${x.toFixed(1)}" y="${height-12}" text-anchor="middle" class="axis-txt">Ngày ${r.day_number}</text>`;});
  out+=`<text x="${width-220}" y="24" class="axis-txt">● Hoàn tất</text><circle cx="${width-230}" cy="20" r="4" fill="#22c55e"/><text x="${width-130}" y="24" class="axis-txt">● Trễ</text><circle cx="${width-140}" cy="20" r="4" fill="#ef4444"/><text x="${width-70}" y="24" class="axis-txt">● Carry-over</text><circle cx="${width-82}" cy="20" r="4" fill="#f59e0b"/>`;
  el.innerHTML=out;
}

function renderTelemetry(d){
  const el=$('telemetryMiniChart'); const rows=d.tel.slice(-36); const width=420,height=290,pad={l:44,r:16,t:20,b:34}; const innerW=width-pad.l-pad.r, innerH=height-pad.t-pad.b; if(!rows.length){el.innerHTML='';return;} const max=Math.max(1,...rows.map(r=>Math.max(r.active_orders||0,r.busy_vehicles||0))); const xs=i=>pad.l+(innerW*(rows.length===1?0:i/(rows.length-1))); const y=v=>pad.t+innerH-innerH*v/max; const path=(arr,col)=>`<path d="${arr.map((v,i)=>`${i?'L':'M'} ${xs(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ')}" fill="none" stroke="${col}" stroke-width="3" stroke-linecap="round"/>`;
  let out=`<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#fff"/>`; for(let g=0;g<=4;g++){const yy=pad.t+innerH*g/4; out+=`<line x1="${pad.l}" y1="${yy}" x2="${width-pad.r}" y2="${yy}" stroke="#e2e8f0" stroke-dasharray="5 5"/>`;}
  out+=path(rows.map(r=>r.active_orders||0),'#2563eb'); out+=path(rows.map(r=>r.busy_vehicles||0),'#14b8a6'); out+=path(rows.map(r=>r.carryover||0),'#f59e0b');
  out+=`<text x="26" y="18" class="axis-txt">Đơn hoạt động</text><text x="148" y="18" class="axis-txt">Xe chạy</text><text x="232" y="18" class="axis-txt">Carry-over</text>`;
  el.innerHTML=out;
}

function renderTables(s,d){
  $('dailyTable').innerHTML=d.daily.map(x=>`<tr><td>Ngày ${x.day_number}</td><td>${fmt(x.completed_cumulative)}</td><td>${fmt(x.late_cumulative)}</td><td>${fmt(x.carryover_count)}</td><td>${fmt(x.distance_km)} km</td></tr>`).join('') || '<tr><td colspan="5" class="report-muted">Chưa có dữ liệu ngày.</td></tr>';
  const watch=[...d.orders].filter(o=>o.late || (o.released&&!o.completed&&!o.cancelled)).sort((a,b)=>(b.lateness||0)-(a.lateness||0)).slice(0,10);
  $('orderAttentionTable').innerHTML=watch.map(o=>`<tr><td><b>${o.order_id}</b></td><td>${o.flow_short}</td><td><span class="chip ${o.late?'red':(o.completed?'green':'amber')}">${orderStatus(o)}</span></td><td>${o.late?fmt(o.lateness)+' phút':'—'}</td></tr>`).join('') || '<tr><td colspan="4" class="report-muted">Không có đơn cần chú ý.</td></tr>';
}

function render(state){
  currentState = state;
  const d=buildSummary(state);
  $('reportTitle').textContent = `Báo cáo vận hành · ${state.traffic_profile?.selected?.region_label||'TP.HCM'} · ${state.settings?.days||1} ngày mô phỏng`;
  $('reportSubtitle').textContent = `${state.vehicles.length} xe · ${d.total} đơn · ${state.intelligence?.algorithm_label||state.settings.algorithm} · seed ${state.settings.seed}. Mô phỏng ${state.run?.status==='completed'||state.run?.status==='horizon_end'?'đã kết thúc':'có thể vẫn đang tiếp diễn'} và báo cáo này luôn cập nhật theo trạng thái mới nhất.`;
  $('heroStatus').textContent = statusText(state.run?.status);
  $('heroClock').textContent = `Ngày ${state.clock?.day_number||1}/${state.clock?.days_total||1} · ${state.clock?.clock||'08:00'}`;
  renderKpis(state,d); renderStatus(state,d); renderInsights(state,d); renderFlow(d); renderVehicles(d); renderDaily(d); renderTelemetry(d); renderTables(state,d);
}

$('printButton').onclick=()=>window.print();
$('jsonButton').onclick=()=>{if(!currentState)return;const blob=new Blob([JSON.stringify(currentState,null,2)],{type:'application/json'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='eureka-v9.2-report-snapshot.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
Demo.connect(render);
})();
