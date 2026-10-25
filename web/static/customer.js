(() => {
  const map=new MapLite.Map('customerMap',{center:[10.79,106.69],zoom:12,minZoom:10,maxZoom:17,maxBounds:[[10.55,106.35],[11.15,107.10]]});
  let samples={};
  const selected={pickup:null,delivery:null};
  let submitting=false, budget=100;

  function show(text,type='success'){
    const m=document.getElementById('msg');m.className=type;m.textContent=text;
    setTimeout(()=>{m.className='';m.textContent=''},5000);
  }

  function sampleButtons(role){
    const uuTien=['HIM_LAM','LOTTE_Q7','SC_VIVO','CRESCENT','PHU_MY_HUNG','NHA_BE','TRUNG_SON','QUAN_8'];
    const ordered=[...uuTien.filter(id=>samples[id]).map(id=>[id,samples[id]]),...Object.entries(samples).filter(([id])=>!uuTien.includes(id))];
    return ordered.slice(0,8).map(([id,p])=>`<button type="button" class="quick-place ${selected[role]===id?'selected':''}" data-role="${role}" data-id="${id}"><strong>${Demo.escapeHtml(p.name)}</strong><span>${Demo.escapeHtml(p.district||'TP.HCM')}</span></button>`).join('');
  }

  function renderSamples(data){
    samples=data.samples||{};
    document.getElementById('pickupSamples').innerHTML=sampleButtons('pickup');
    document.getElementById('deliverySamples').innerHTML=sampleButtons('delivery');
    document.querySelectorAll('[data-role][data-id]').forEach(b=>b.addEventListener('click',()=>{
      selected[b.dataset.role]=b.dataset.id;
      document.getElementById(`${b.dataset.role}Sample`).value=b.dataset.id;
      renderSamples(data);refreshMarkers();
    }));
  }

  function refreshMarkers(){
    map.clearMarkers('sel-');
    for(const role of ['pickup','delivery']){
      const id=selected[role];
      if(id&&samples[id]){
        const p=samples[id];
        map.addMarker(`sel-${role}`,p.lat,p.lon,Demo.orderHtml(role==='pickup'?'pickup':'delivery',role==='pickup'?'L':'G'),`${role==='pickup'?'Điểm lấy hàng':'Điểm giao hàng'} · ${p.address}`);
      }
    }
  }

  function syncMap(data){
    map.clearMarkers('veh-');
    data.vehicles.forEach(v=>map.addMarker(`veh-${v.vehicle_id}`,v.lat,v.lon,Demo.vehicleHtml(v),`${v.vehicle_id} · ${v.model||Demo.vehicleType(v.vehicle_type)} ·${v.load}/${v.capacity} kg`));
    refreshMarkers();
  }

  function flowVisibility(){
    const f=document.getElementById('flowType').value;
    document.getElementById('pickupGroup').style.display=f==='INBOUND'?'none':'';
    document.getElementById('deliveryGroup').style.display=f==='OUTBOUND'?'none':'';
  }
  document.getElementById('flowType').addEventListener('change',flowVisibility);flowVisibility();

  function renderOrders(data){
    const box=document.getElementById('myOrders');
    const arr=data.orders.filter(o=>o.origin==='MANUAL').slice(-18).reverse();
    box.innerHTML=arr.length?arr.map(o=>{
      const owner=o.assigned_vehicle_id?`Hàng trên ${o.assigned_vehicle_id}`:(o.committed_vehicle_id?`Đã cam kết ${o.committed_vehicle_id}`:'Chưa gán xe');
      return `<article class="mobile-order"><div class="top"><div><div class="id">${Demo.escapeHtml(o.order_id)}</div><div class="muted small">F${o.flow_number||'?'} · ${Demo.escapeHtml(o.flow_short||Demo.flow(o.flow_type))} · ${o.weight} kg · ${Demo.origin(o.origin)}</div></div><span class="badge ${o.completed?'done':'active'}">${o.completed?'ĐÃ HOÀN TẤT':Demo.escapeHtml(Demo.stageLabel(o.stage))}</span></div><div class="route">Tác vụ kế tiếp: <b>${Demo.escapeHtml(Demo.operation(o.next_operation))}</b><br>Hàng: ${Demo.escapeHtml(Demo.cargo(o.cargo_state))} · ${Demo.escapeHtml(owner)}</div>${(!o.completed&&!o.cancelled&&!o.assigned_vehicle_id)?`<button class="btn btn-secondary" data-cancel="${o.order_id}">Hủy đơn</button>`:''}</article>`;
    }).join(''):'<div class="empty-mobile">Chưa có đơn phát sinh.</div>';
    box.querySelectorAll('[data-cancel]').forEach(b=>b.addEventListener('click',async()=>{try{await Demo.api(`/api/orders/${b.dataset.cancel}/cancel`,'POST')}catch(e){show(e.message,'error')}}));
  }

  document.getElementById('orderForm').addEventListener('submit',async ev=>{
    ev.preventDefault();
    if(submitting||budget<=0)return;
    submitting=true;document.getElementById('submitOrder').disabled=true;
    const flow=document.getElementById('flowType').value;
    const pId=document.getElementById('pickupSample').value,dId=document.getElementById('deliverySample').value;
    const pText=document.getElementById('pickupAddress').value.trim(),dText=document.getElementById('deliveryAddress').value.trim();
    const payload={
      flow_type:flow,
      weight:Number(document.getElementById('weight').value||80),
      window_minutes:Number(document.getElementById('windowMinutes').value||180),
      priority:Number(document.getElementById('priority').value||2),
      note:document.getElementById('note').value,
      pickup:flow==='INBOUND'?null:(pId?{sample_id:pId}:{address:pText}),
      delivery:flow==='OUTBOUND'?null:(dId?{sample_id:dId}:{address:dText})
    };
    try{
      const r=await Demo.api('/api/orders','POST',payload);
      show(`${r.order_id} đã được đưa vào hệ thống dưới dạng đơn phát sinh.`);
      document.getElementById('note').value='';
    }catch(e){show(e.message,'error')}
    finally{submitting=false;document.getElementById('submitOrder').disabled=budget<=0;}
  });

  for(const role of ['pickup','delivery'])document.getElementById(role+'Address').addEventListener('input',()=>{
    selected[role]=null;document.getElementById(role+'Sample').value='';refreshMarkers();
  });
  Demo.connect(data=>{
    budget=100-(data.run?.total??data.orders.length);
    document.getElementById('orderBudget').textContent=`Còn có thể thêm ${budget} đơn trong ca này (tối đa 100, gồm cả đơn chưa phát sinh).`;
    document.getElementById('submitOrder').disabled=submitting||budget<=0;
    renderSamples(data);syncMap(data);renderOrders(data);
  });
})();
