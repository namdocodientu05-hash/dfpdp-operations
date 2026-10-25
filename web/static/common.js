window.Demo = (() => {
  const state = { data:null, ws:null, listeners:new Set(), reconnect:null };

  const MAP = {
    flow: {
      PICKUP_DELIVERY:'Lấy tại khách → giao cho khách',
      OUTBOUND:'Lấy tại khách → dỡ tại kho',
      INBOUND:'Chất tại kho → giao cho khách',
      INBOUND_PRELOADED:'Đã chất sẵn tại kho → giao cho khách'
    },
    operation: {
      CUSTOMER_PICKUP:'Lấy hàng tại khách',
      CUSTOMER_DELIVERY:'Giao hàng cho khách',
      DEPOT_LOAD:'Chất hàng tại kho',
      DEPOT_UNLOAD:'Dỡ hàng tại kho'
    },
    stage: {
      WAIT_CUSTOMER_PICKUP:'Chờ lấy hàng tại khách',
      WAIT_CUSTOMER_DELIVERY:'Chờ giao hàng cho khách',
      WAIT_DEPOT_LOAD:'Chờ chất hàng tại kho',
      WAIT_DEPOT_UNLOAD:'Chờ dỡ hàng tại kho',
      COMPLETED:'Đã hoàn tất'
    },
    cargo: { NOT_LOADED:'Chưa chất lên xe', ON_VEHICLE:'Đang ở trên xe', UNLOADED:'Đã dỡ khỏi xe' },
    vehicleStatus: {
      IDLE:'Đang rảnh', EN_ROUTE:'Đang trên đường', SCHEDULED:'Chờ giờ xuất phát',
      BROKEN:'Đang hỏng', OFF_SHIFT:'Hết ca'
    },
    orderClass: { STATIC:'Đơn tĩnh', DYNAMIC:'Đơn động' },
    origin: { STATIC:'Tĩnh đầu ca', AUTO:'Động tự động', MANUAL:'Động thủ công' },
    decisionStatus: {
      OK:'Hợp lệ', NO_ACTION:'Không có tác vụ', NO_FEASIBLE_ACTION:'Không có tác vụ khả thi',
      DEFERRED_TO_NEXT_EVENT:'Dời sang sự kiện kế tiếp',
      NO_VALID_ACTION_AFTER_RESELECT:'Không tìm được tác vụ hợp lệ sau khi chọn lại',
      REJECTED:'Bị từ chối', ERROR:'Lỗi'
    },
    eventKind: {
      SYSTEM:'HỆ THỐNG', MANIFEST:'DANH SÁCH HÀNG', DECISION:'QUYẾT ĐỊNH', EVENT:'SỰ KIỆN',
      EXECUTE:'THỰC HIỆN', NEW_ORDER:'ĐƠN MỚI', TRAFFIC_UPDATE:'GIAO THÔNG',
      DEPARTURE:'XUẤT KHO', SERVICE:'HOÀN TẤT'
    },
    eventType: {
      SHIFT_START:'Bắt đầu ca', NEW_ORDER:'Đơn mới', CANCEL_ORDER:'Hủy đơn',
      ORDER_PRIORITY_UPDATE:'Cập nhật ưu tiên', TIME_WINDOW_UPDATE:'Cập nhật cửa sổ thời gian',
      TRAFFIC_UPDATE:'Cập nhật giao thông', GPS_UPDATE:'Cập nhật GPS',
      VEHICLE_BREAKDOWN:'Xe hỏng', VEHICLE_RECOVERY:'Xe phục hồi', SERVICE_COMPLETE:'Hoàn tất phục vụ',
      CUSTOMER_PICKUP_COMPLETE:'Hoàn tất lấy hàng tại khách', CUSTOMER_DELIVERY_COMPLETE:'Hoàn tất giao hàng cho khách',
      DEPOT_LOAD_COMPLETE:'Hoàn tất chất hàng tại kho', DEPOT_UNLOAD_COMPLETE:'Hoàn tất dỡ hàng tại kho',
      PICKUP_COMPLETE:'Hoàn tất lấy hàng', DELIVERY_COMPLETE:'Hoàn tất giao hàng',
      MANUAL_REPLAN:'Yêu cầu quyết định thủ công'
    },
    vehicleType: { VAN:'Xe van', TRUCK:'Xe tải' },
    decode: { greedy:'Chọn xác suất cao nhất', sampling:'Lấy mẫu' }
  };

  const translate=(dict,value,fallback=value)=>MAP[dict]?.[value] ?? fallback ?? '—';
  const flow=v=>translate('flow',v);
  const operation=v=>translate('operation',v);
  const stageLabel=v=>translate('stage',v);
  const cargo=v=>translate('cargo',v);
  const vehicleStatus=v=>translate('vehicleStatus',v);
  const orderClass=v=>translate('orderClass',v);
  const origin=v=>translate('origin',v,v||'—');
  const decisionStatus=v=>translate('decisionStatus',v);
  const eventKind=v=>translate('eventKind',v);
  const eventType=v=>translate('eventType',v);
  const vehicleType=v=>translate('vehicleType',v);
  const decode=v=>translate('decode',v,v);

  function validatorNotes(v=''){
    let s=String(v||'');
    const pairs=[
      ['no vehicle has a feasible order','Không có xe nào còn đơn khả thi'],
      ['first-step paper initialisation','Khởi tạo bước đầu theo logic bài báo'],
      ['argmin accumulated time','Chọn xe có tổng thời gian tích lũy nhỏ nhất'],
      ['validator disabled','Bộ kiểm tra khả thi đang tắt'],
      ['empty order slot','Ô đơn hàng trống'],
      ['vehicle unavailable (breakdown / off shift)','Xe không sẵn sàng (hỏng hoặc hết ca)'],
      ['order cancelled','Đơn đã bị hủy'],['order already completed','Đơn đã hoàn tất'],
      ['service location unavailable','Địa điểm phục vụ không sẵn sàng'],
      ['time window infeasible','Không đáp ứng được cửa sổ thời gian'],
      ['no available service location','Không có địa điểm phục vụ khả thi'],
      ['requested but cargo is not loaded yet','được yêu cầu nhưng hàng chưa được chất lên xe'],
      ['location_rule=min_travel_plus_wait','quy tắc địa điểm = ít thời gian di chuyển + chờ nhất'],
      ['min_travel_plus_wait','ít thời gian di chuyển + chờ nhất']
    ];
    for(const [a,b] of pairs)s=s.split(a).join(b);
    s=s.replace(/ok \(([^)]+)\)/gi,'Hợp lệ ($1)');
    s=s.replace(/departure (\d{2}:\d{2}) is past the next event at (\d{2}:\d{2})/gi,'Thời điểm khởi hành $1 đã sau sự kiện kế tiếp lúc $2');
    s=s.replace(/capacity violated for ([^:]+): w=([0-9.]+) > remaining=([0-9.]+)/gi,'Vượt tải cho $1: khối lượng $2 > sức chứa còn lại $3');
    s=s.replace(/must be done by ([^,]+),/gi,'phải do $1 thực hiện,');
    return s.replace(/^\s*\|\s*/,'').replace(/\s*\|\s*/g,' · ');
  }

  function clockBind(data){
    document.querySelectorAll('[data-clock]').forEach(el=>el.textContent=data?.clock?.clock||'--:--');
    document.querySelectorAll('[data-speed]').forEach(el=>el.textContent=data?.clock?.speed_rule||'');
  }
  function notify(data){state.data=data;clockBind(data);state.listeners.forEach(fn=>{try{fn(data)}catch(e){console.error(e)}});}
  function openSocket(){
    clearTimeout(state.reconnect);
    const proto=location.protocol==='https:'?'wss:':'ws:';
    const ws=new WebSocket(`${proto}//${location.host}/ws`);state.ws=ws;
    ws.onopen=()=>{try{ws.send('giu-ket-noi')}catch(e){}};
    ws.onmessage=e=>{try{notify(JSON.parse(e.data))}catch(err){console.error(err)}};
    ws.onclose=()=>{state.reconnect=setTimeout(openSocket,1200)};
  }
  function connect(onState){state.listeners.add(onState);if(!state.ws||state.ws.readyState>1)openSocket();fetch('/api/state').then(r=>r.json()).then(notify).catch(()=>{});return()=>state.listeners.delete(onState);}
  setInterval(()=>{if(state.ws?.readyState===1){try{state.ws.send('giu-ket-noi')}catch(e){}}},15000);

  async function api(url,method='GET',body=null){
    const res=await fetch(url,{method,headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
    let data={};try{data=await res.json()}catch(e){}
    if(!res.ok)throw new Error(data.detail||'Yêu cầu thất bại');return data;
  }

  // ====================================================================================
  // ĐỊNH TUYẾN ĐƯỜNG BỘ V7
  // Trình duyệt gọi trực tiếp 2 máy chủ OSRM trước. Cách này tránh trường hợp Python/Windows
  // bị proxy/chứng chỉ/DNS chặn trong khi Chrome vẫn truy cập Internet bình thường.
  // Sau đó mới thử điểm truy cập máy chủ cục bộ; chỉ cuối cùng mới vẽ đường nối dự phòng.
  // ====================================================================================
  const routeCache=new Map();
  const directProviders=[
    'https://router.project-osrm.org',
    'https://routing.openstreetmap.de/routed-car'
  ];
  const routeKey=(a,b)=>[a[0],a[1],b[0],b[1]].map(v=>Number(v).toFixed(5)).join('|');

  async function fetchWithTimeout(url,ms=5500){
    const ctl=new AbortController();const timer=setTimeout(()=>ctl.abort(),ms);
    try{return await fetch(url,{signal:ctl.signal,mode:'cors',cache:'no-store'});}finally{clearTimeout(timer);}
  }
  async function directOsrm(base,a,b){
    const url=`${base}/route/v1/driving/${Number(a[1]).toFixed(6)},${Number(a[0]).toFixed(6)};${Number(b[1]).toFixed(6)},${Number(b[0]).toFixed(6)}?overview=full&geometries=geojson&steps=false&alternatives=false`;
    const r=await fetchWithTimeout(url,5500);if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();if(d.code!=='Ok'||!d.routes?.length)throw new Error(d.message||d.code||'Không có tuyến');
    const route=d.routes[0];const points=(route.geometry?.coordinates||[]).map(([lon,lat])=>[lat,lon]);
    if(points.length<2)throw new Error('Không có hình học tuyến');
    return {ok:true,fallback:false,source:base,provider:base,points,distance_m:Number(route.distance||0),duration_s:Number(route.duration||0),via:'trình duyệt'};
  }
  async function roadRoute(a,b){
    const key=routeKey(a,b);if(routeCache.has(key))return routeCache.get(key);
    const promise=(async()=>{
      const attempts=[];
      for(const base of directProviders){
        try{const out=await directOsrm(base,a,b);out.attempts=[...attempts,{provider:base,ok:true}];return out;}
        catch(e){attempts.push({provider:base,ok:false,error:String(e?.message||e)});}
      }
      try{
        const params=new URLSearchParams({start_lat:a[0],start_lon:a[1],end_lat:b[0],end_lon:b[1]});
        const r=await fetch(`/api/road-route?${params.toString()}`,{cache:'no-store'});let d={};try{d=await r.json()}catch(e){}
        if(r.ok&&d?.points?.length>1){d.via='máy chủ cục bộ';d.attempts=[...attempts,...(d.attempts||[])];return d;}
      }catch(e){attempts.push({provider:'localhost',ok:false,error:String(e?.message||e)});}
      return {ok:false,fallback:true,source:'Đường nối dự phòng',provider:'fallback',via:'dự phòng',warning:'Không kết nối được dịch vụ định tuyến công khai. Kiểm tra Internet/firewall rồi bấm “Kiểm tra định tuyến”.',points:[a,b],distance_m:0,duration_s:0,attempts};
    })();
    routeCache.set(key,promise);return promise;
  }
  function clearRouteCache(){routeCache.clear();}
  async function routingDiagnostic(){
    const a=[10.8188,106.6519],b=[10.7296,106.7180];clearRouteCache();return roadRoute(a,b);
  }

  function escapeHtml(s=''){return String(s).replace(/[&<>'"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#039;','"':'&quot;'}[m]));}
  // Hướng di chuyển (độ, theo chiều kim đồng hồ từ hướng Bắc) giữa hai điểm [lat,lon].
  function bearing(a,b){if(!a||!b)return null;const dy=b[0]-a[0],dx=(b[1]-a[1])*Math.cos(a[0]*Math.PI/180);return (Math.abs(dx)+Math.abs(dy))>1e-9?Math.atan2(dx,dy)*180/Math.PI:null;}
  function headingAlong(points,progress,along){if(!points?.length||points.length<2)return null;const p=Math.max(0,Math.min(1,Number(progress)||0)),e=.004;return bearing(along(points,Math.max(0,p-e)),along(points,Math.min(1,p+e)));}
  // Minh họa nhìn ngang (đầu xe hướng sang phải): van thân trắng sọc màu xe, xe tải thùng trắng cabin màu xe.
  const wheel=(cx,cy=29)=>`<circle cx="${cx}" cy="${cy}" r="5.2" fill="#111827"/><circle cx="${cx}" cy="${cy}" r="2.6" fill="#cbd5e1"/><circle cx="${cx}" cy="${cy}" r="1" fill="#64748b"/>`;
  function vanSvg(color,cls){return `<svg class="${cls} van" viewBox="0 0 72 36" aria-hidden="true"><ellipse cx="36" cy="32.5" rx="31" ry="2.6" fill="rgba(0,0,0,.28)"/><path d="M5 9Q5 5 9 5H44Q48 5 51 8L60 17Q66 18 67 22V27Q67 29 65 29H6Q5 29 5 27Z" fill="#f8fafc" stroke="#94a3b8" stroke-width=".8"/><path d="M5.4 23H66.6V27Q66.6 28.6 65 28.6H6Q5.4 28.6 5.4 27Z" fill="#dfe5ec"/><path d="M5.4 18.5H62.5L64.5 21.3H5.4Z" fill="${color}"/><path d="M45 8.5H49.5L57.5 17H45Z" fill="#334155"/><path d="M46.4 9.6H48.6L52.2 13.6H46.4Z" fill="#94a3b8" opacity=".55"/><path d="M43 5.6V28.4M25 5.6V28.4" stroke="#cbd5e1" stroke-width=".7"/><rect x="38.5" y="13" width="3" height="1.2" rx=".6" fill="#94a3b8"/><rect x="56.4" y="13.6" width="2" height="3.2" rx=".6" fill="#334155"/><path d="M64.4 20.6H66.8V23.2H64.9Z" fill="#fde68a"/><rect x="5" y="9.5" width="1.6" height="5" fill="#ef4444"/><rect x="62.5" y="26" width="5.5" height="2.6" rx="1" fill="#475569"/><rect x="3.8" y="26" width="4.2" height="2.6" rx="1" fill="#475569"/>${wheel(16)}${wheel(55)}</svg>`;}
  function truckSvg(color,cls,long){const L=long?54:42,c=L+3,W=c+23;return `<svg class="${cls} truck ${long?'long':''}" viewBox="0 0 ${W} 36" aria-hidden="true"><ellipse cx="${W/2}" cy="32.5" rx="${W/2-3}" ry="2.6" fill="rgba(0,0,0,.28)"/><rect x="3" y="25.5" width="${W-6}" height="3.2" fill="#334155"/><rect x="2" y="2.5" width="${L}" height="23" rx="2" fill="#f8fafc" stroke="#94a3b8" stroke-width=".8"/><rect x="2.4" y="20.5" width="${L-.8}" height="4.6" fill="#dfe5ec"/><rect x="2.4" y="15.5" width="${L-.8}" height="3" fill="${color}"/><path d="M5 4.5V23.5" stroke="#cbd5e1" stroke-width=".7"/><path d="M${c} 9H${c+12}Q${c+14} 9 ${c+15} 11L${c+19} 17Q${c+22} 18 ${c+22} 21V27Q${c+22} 28.5 ${c+20.5} 28.5H${c}Z" fill="${color}" stroke="rgba(15,23,42,.35)" stroke-width=".8"/><path d="M${c+3} 11H${c+12.5}L${c+16.5} 17H${c+3}Z" fill="#334155"/><path d="M${c+4.2} 12H${c+7}L${c+9.5} 15H${c+4.2}Z" fill="#94a3b8" opacity=".55"/><rect x="${c+19.8}" y="20" width="2.2" height="2.6" fill="#fde68a"/><rect x="${c+17.5}" y="25.8" width="5.5" height="2.7" rx="1" fill="#475569"/>${wheel(12)}${long?wheel(24):''}${wheel(c+14)}</svg>`;}
  function vehicleArt(v,cls='truck-art'){const color=v.color||'#168269';return v.vehicle_type==='TRUCK'?truckSvg(color,cls,/20/.test(v.model||'')):vanSvg(color,cls);}
  // Bản đồ: nhìn từ trên xuống, đầu xe hướng lên (Bắc), xoay theo heading.
  function vanTopSvg(color){return `<svg class="top-art van" viewBox="0 0 24 48" aria-hidden="true"><rect x="4.5" y="5" width="18" height="42" rx="4" fill="rgba(0,0,0,.3)"/><rect x="1.2" y="11" width="2.6" height="3.4" rx=".8" fill="#334155"/><rect x="20.2" y="11" width="2.6" height="3.4" rx=".8" fill="#334155"/><rect x="3" y="3" width="18" height="42" rx="4" fill="#f8fafc" stroke="#94a3b8" stroke-width=".8"/><path d="M4.2 8Q4.2 4 8 3.8H16Q19.8 4 19.8 8V9.5H4.2Z" fill="#e2e8f0"/><path d="M4.6 10.2H19.4L18.2 15.6H5.8Z" fill="#334155"/><path d="M6.3 11H10L9.3 14.6H6.6Z" fill="#94a3b8" opacity=".5"/><rect x="5.4" y="16.6" width="13.2" height="26.6" rx="1.6" fill="#fff" stroke="#e2e8f0" stroke-width=".6"/><path d="M5.6 22.5H18.4M5.6 28.5H18.4M5.6 34.5H18.4M5.6 40H18.4" stroke="#e2e8f0" stroke-width=".7"/><rect x="3" y="16.6" width="2.2" height="26.6" fill="${color}"/><rect x="18.8" y="16.6" width="2.2" height="26.6" fill="${color}"/><rect x="5" y="3.4" width="3.2" height="1.4" rx=".7" fill="#fde68a"/><rect x="15.8" y="3.4" width="3.2" height="1.4" rx=".7" fill="#fde68a"/><rect x="4.6" y="43.6" width="3" height="1.2" rx=".6" fill="#ef4444"/><rect x="16.4" y="43.6" width="3" height="1.2" rx=".6" fill="#ef4444"/></svg>`;}
  function truckTopSvg(color,long){const L=long?40:32,H=16+L;return `<svg class="top-art truck ${long?'long':''}" viewBox="0 0 26 ${H}" aria-hidden="true"><rect x="3.5" y="4" width="22" height="${H-3}" rx="3" fill="rgba(0,0,0,.3)"/><rect x="0.6" y="5.5" width="2.8" height="3.4" rx=".8" fill="#334155"/><rect x="22.6" y="5.5" width="2.8" height="3.4" rx=".8" fill="#334155"/><rect x="4" y="2" width="18" height="11.5" rx="3.2" fill="${color}" stroke="rgba(15,23,42,.35)" stroke-width=".8"/><path d="M5.6 6.2H20.4L19.4 10H6.6Z" fill="#334155"/><path d="M7.2 6.8H10.6L10 9.4H7.6Z" fill="#94a3b8" opacity=".5"/><rect x="6" y="2.4" width="3" height="1.3" rx=".6" fill="#fde68a"/><rect x="17" y="2.4" width="3" height="1.3" rx=".6" fill="#fde68a"/><rect x="9" y="13" width="8" height="2.5" fill="#334155"/><rect x="2" y="15" width="22" height="${L}" rx="1.6" fill="#f8fafc" stroke="#94a3b8" stroke-width=".8"/><rect x="2.4" y="15.4" width="21.2" height="2.6" fill="${color}"/>${Array.from({length:Math.floor(L/6)},(_,i)=>`<path d="M3 ${21+i*6}H23" stroke="#e2e8f0" stroke-width=".7"/>`).join('')}<rect x="3.5" y="${H-2}" width="3.2" height="1.2" rx=".6" fill="#ef4444"/><rect x="19.3" y="${H-2}" width="3.2" height="1.2" rx=".6" fill="#ef4444"/></svg>`;}
  function vehicleHtml(v,heading=0,extra=''){const color=v.color||'#168269',art=v.vehicle_type==='TRUCK'?truckTopSvg(color,/20/.test(v.model||'')):vanTopSvg(color);return `<div class="map-veh ${v.broken?'broken':''} ${extra}"><span class="veh-body" style="transform:rotate(${Number(heading)||0}deg)">${art}</span><small style="background:${color}">${escapeHtml(v.label||v.vehicle_id.replace('V','Xe '))}</small></div>`;}
  // Đơn hàng: thùng carton (điểm lấy) và cửa hàng khách (điểm giao); kho có mái đỏ.
  function boxArt(cls='parcel-art'){return `<svg class="${cls}" viewBox="0 0 40 40" aria-hidden="true"><ellipse cx="20" cy="36.5" rx="15" ry="2.4" fill="rgba(0,0,0,.3)"/><path d="M20 6L36 13L20 20L4 13Z" fill="#e6c08a"/><path d="M4 13L20 20V36L4 29Z" fill="#c98f52"/><path d="M36 13V29L20 36V20Z" fill="#a8743f"/><path d="M12 9.5L28 16.5V32.5" fill="none" stroke="#f2dfb8" stroke-width="3"/><path d="M4 13L20 20L36 13M20 20V36" fill="none" stroke="#8a5d2e" stroke-width=".6" opacity=".6"/><path d="M7 19L14.5 22.3V27.5L7 24.2Z" fill="#fff" opacity=".9"/><path d="M8.3 21.4L13 23.5M8.3 23.3L11.5 24.7" stroke="#94a3b8" stroke-width=".7"/></svg>`;}
  function storeArt(cls='parcel-art'){const aw=[0,1,2,3,4,5,6].map(i=>`<rect x="${4+i*4}" y="13.5" width="4" height="5" fill="${i%2?'#fff':'#e2453c'}"/><path d="M${4+i*4} 18.5Q${6+i*4} 21 ${8+i*4} 18.5Z" fill="${i%2?'#fff':'#e2453c'}"/>`).join('');return `<svg class="${cls}" viewBox="0 0 40 40" aria-hidden="true"><ellipse cx="21" cy="36.5" rx="16" ry="2.4" fill="rgba(0,0,0,.3)"/><path d="M31 13L37 9.5V32L31 35.5Z" fill="#3c6aa3"/><path d="M5 13L11 9.5H37L31 13Z" fill="#9cc0ea"/><rect x="5" y="13" width="26" height="22.5" fill="#5b8fd0"/>${aw}<rect x="8" y="23" width="7.5" height="6.5" fill="#dbeafe" stroke="#2f5f99" stroke-width=".8"/><rect x="19.5" y="23.5" width="8" height="12" fill="#1e3a5f"/><circle cx="25.8" cy="30" r=".7" fill="#fbbf24"/></svg>`;}
  function depotArt(cls='depot-art'){return `<svg class="${cls}" viewBox="0 0 52 42" aria-hidden="true"><ellipse cx="26" cy="38.5" rx="23" ry="2.8" fill="rgba(0,0,0,.3)"/><path d="M41 17L47.5 13V33.5L41 37.5Z" fill="#d6bd92"/><rect x="5" y="17" width="36" height="20.5" fill="#f3e4c6"/><path d="M2.5 18L9.5 7.5H44.5L49.5 13L43.5 18Z" fill="#d9483b"/><path d="M2.5 18H43.5L49.5 13" fill="none" stroke="#a9322a" stroke-width="1.6"/><path d="M9.5 7.5L15 18M17 7.5L22 18M24.5 7.5L29 18M32 7.5L36 18M39.5 7.5L43 18" stroke="#c23d31" stroke-width=".7"/><rect x="13" y="22.5" width="19" height="15" fill="#9ca3af"/><path d="M13 25.5H32M13 28.5H32M13 31.5H32M13 34.5H32" stroke="#6b7280" stroke-width=".8"/><rect x="34.5" y="31" width="5.5" height="5" fill="#c98f52"/><rect x="35.5" y="26.5" width="4.5" height="4.5" fill="#e0b57c"/><rect x="6.5" y="21" width="4.5" height="4" fill="#bfdbfe" stroke="#94a3b8" stroke-width=".6"/></svg>`;}
  function orderHtml(kind,label,extra=''){return `<div class="map-parcel art ${kind} ${extra}" title="${escapeHtml(label||'')}">${kind==='pickup'?boxArt(''):storeArt('')}<em>${kind==='pickup'?'↑':'↓'}</em></div>`;}
  function statusClass(s){if(s==='COMPLETED'||s==='DELIVERED')return'done';if(String(s).includes('WAIT')||s==='SCHEDULED')return'pending';if(s==='CANCELLED'||s==='REJECTED')return'reject';return'active';}
  function fmt(n,d=1){const x=Number(n);return Number.isFinite(x)?x.toFixed(d):'—';}
  function minutesToClock(m){m=Number(m);if(!Number.isFinite(m))return'—';const h=Math.floor(m/60)%24,mi=Math.floor(m%60);return `${String(h).padStart(2,'0')}:${String(mi).padStart(2,'0')}`;}

  return {
    connect,api,roadRoute,clearRouteCache,routingDiagnostic,vehicleHtml,vehicleArt,boxArt,storeArt,depotArt,bearing,headingAlong,orderHtml,statusClass,escapeHtml,fmt,minutesToClock,
    flow,operation,stageLabel,cargo,vehicleStatus,orderClass,origin,decisionStatus,eventKind,eventType,vehicleType,decode,validatorNotes,
    get data(){return state.data;}
  };
})();
