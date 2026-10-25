/* Bộ hiển thị bản đồ nhẹ cho prototype nghiên cứu.
   Không phụ thuộc Leaflet/Mapbox. V7 có tile fallback: OSM chuẩn -> OSM HOT. */
window.MapLite = (() => {
  const TILE = 256;
  const clamp = (v,a,b)=>Math.max(a,Math.min(b,v));
  const lonToX = (lon,z)=>(lon+180)/360 * TILE * 2**z;
  const latToY = (lat,z)=>{
    const s = Math.sin(lat*Math.PI/180);
    return (0.5 - Math.log((1+s)/(1-s))/(4*Math.PI)) * TILE * 2**z;
  };
  const xToLon = (x,z)=>x/(TILE*2**z)*360-180;
  const yToLat = (y,z)=>{
    const n=Math.PI-2*Math.PI*y/(TILE*2**z);
    return 180/Math.PI*Math.atan(0.5*(Math.exp(n)-Math.exp(-n)));
  };

  class SlippyMap {
    constructor(el, opts={}){
      this.el = typeof el==='string' ? document.getElementById(el) : el;
      this.center = opts.center || [10.7769,106.7009];
      this.zoom = opts.zoom || 12;
      this.minZoom = opts.minZoom ?? 11;
      this.maxZoom = Math.min(opts.maxZoom ?? 16, 16); // nền Esri Light Gray chỉ có tới zoom 16
      this.maxBounds = opts.maxBounds || [[10.58,106.45],[11.05,107.02]];
      this.markers = new globalThis.Map();
      this.lines = new globalThis.Map();
      this.tileCache = new globalThis.Map();
      this.theme = opts.theme || 'light'; this.el.dataset.theme = this.theme;
      this.drag = null;
      this._build(); this._bind(); this._measure(); this.render();
    }
    // Kích thước khung được lưu sẵn (chỉ đo khi đổi kích thước) để project() không đọc DOM ở từng điểm —
    // đọc clientWidth sau khi vừa sửa DOM buộc trình duyệt tính lại bố cục ngay lập tức.
    _measure(){this.w=this.el.clientWidth;this.h=this.el.clientHeight;}
    _view(){const k=`${this.zoom}|${this.center[0]}|${this.center[1]}|${this.w}|${this.h}`;
      if(this._v?.k!==k){const z=this.zoom;this._v={k,z,ox:lonToX(this.center[1],z)-this.w/2,oy:latToY(this.center[0],z)-this.h/2};}return this._v;}
    _build(){
      this.el.classList.add('ml-map');
      this.el.innerHTML = '';
      this.tiles = document.createElement('div'); this.tiles.className='ml-tiles';
      this.svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); this.svg.classList.add('ml-overlay');
      this.markerLayer = document.createElement('div'); this.markerLayer.className='ml-markers';
      this.controls = document.createElement('div'); this.controls.className='ml-controls';
      this.controls.innerHTML='<button type="button" data-z="in" aria-label="Phóng to">+</button><button type="button" data-z="out" aria-label="Thu nhỏ">−</button>';
      this.attr = document.createElement('div'); this.attr.className='ml-attrib'; this.attr.innerHTML='© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> · Esri, HERE, Garmin';
      this.el.append(this.tiles,this.svg,this.markerLayer,this.controls,this.attr);
      this.controls.querySelector('[data-z=in]').addEventListener('click',()=>this.setView(this.center,this.zoom+1));
      this.controls.querySelector('[data-z=out]').addEventListener('click',()=>this.setView(this.center,this.zoom-1));
      this.ro = new ResizeObserver(()=>{this._measure();this.render();}); this.ro.observe(this.el);
    }
    _bind(){
      this.el.addEventListener('wheel',e=>{e.preventDefault();this.setView(this.center,this.zoom+(e.deltaY<0?1:-1));},{passive:false});
      this.el.addEventListener('pointerdown',e=>{if(e.target.closest('.ml-controls,.ml-marker'))return;this.el.setPointerCapture?.(e.pointerId);this.drag={x:e.clientX,y:e.clientY,c:[...this.center]};this.el.classList.add('dragging');});
      this.el.addEventListener('pointermove',e=>{if(!this.drag)return;const z=this.zoom,cx=lonToX(this.drag.c[1],z),cy=latToY(this.drag.c[0],z);const nx=cx-(e.clientX-this.drag.x),ny=cy-(e.clientY-this.drag.y);this.center=this._clampCenter([yToLat(ny,z),xToLon(nx,z)]);this.render();});
      const end=()=>{this.drag=null;this.el.classList.remove('dragging')};
      this.el.addEventListener('pointerup',end);this.el.addEventListener('pointercancel',end);
    }
    _clampCenter(c){
      const [[s,w],[n,e]]=this.maxBounds;
      return [clamp(c[0],s,n),clamp(c[1],w,e)];
    }
    setView(center,zoom=this.zoom){this.center=this._clampCenter(center);this.zoom=clamp(Math.round(zoom),this.minZoom,this.maxZoom);this.render();}
    fitBounds(points,padding=36){
      if(!points||!points.length)return;
      const lats=points.map(p=>p[0]),lons=points.map(p=>p[1]);
      const center=[(Math.min(...lats)+Math.max(...lats))/2,(Math.min(...lons)+Math.max(...lons))/2];
      const w=Math.max(100,this.w-2*padding),h=Math.max(100,this.h-2*padding);
      let z=this.maxZoom;
      for(;z>this.minZoom;z--){
        const xs=lons.map(l=>lonToX(l,z)),ys=lats.map(l=>latToY(l,z));
        if(Math.max(...xs)-Math.min(...xs)<=w && Math.max(...ys)-Math.min(...ys)<=h)break;
      }
      this.center=this._clampCenter(center);this.zoom=z;this.render();
    }
    project(lat,lon){const v=this._view();return [lonToX(lon,v.z)-v.ox,latToY(lat,v.z)-v.oy];}
    addMarker(id,lat,lon,html,title=''){
      let m=this.markers.get(id);
      if(!m){m=document.createElement('button');m.type='button';m.className='ml-marker';m.dataset.id=id;this.markerLayer.appendChild(m);this.markers.set(id,m);}
      // Chỉ dựng lại DOM của icon khi HTML thực sự thay đổi (tránh dựng lại hàng trăm SVG mỗi giây).
      m.dataset.lat=lat;m.dataset.lon=lon;if(m._html!==html){m.innerHTML=html;m._html=html;}if(m.title!==title)m.title=title;this._placeMarker(m);return m;
    }
    moveMarker(id,lat,lon){const m=this.markers.get(id);if(!m)return null;m.dataset.lat=lat;m.dataset.lon=lon;this._placeMarker(m);return m;}
    removeMarker(id){const m=this.markers.get(id);if(m)m.remove();this.markers.delete(id);}
    clearMarkers(prefix=''){for(const id of [...this.markers.keys()])if(!prefix||id.startsWith(prefix))this.removeMarker(id);}
    setTheme(theme){if(theme===this.theme)return;this.theme=theme;this.el.dataset.theme=theme;for(const img of this.tileCache.values())img.remove();this.tileCache.clear();this.render();}
    addPolyline(id,points,opts={}){this.lines.set(id,{points,opts});this._renderOverlay();}
    // Thay toàn bộ polyline có cùng tiền tố trong một lần vẽ (dùng cho cập nhật theo khung hình).
    setPolylines(prefix,entries){for(const id of [...this.lines.keys()])if(id.startsWith(prefix))this.lines.delete(id);for(const [id,points,opts] of entries)this.lines.set(id,{points,opts:opts||{}});this._renderOverlay();}
    removePolyline(id){this.lines.delete(id);this._renderOverlay();}
    clearPolylines(prefix=''){for(const id of [...this.lines.keys()])if(!prefix||id.startsWith(prefix))this.lines.delete(id);this._renderOverlay();}
    _placeMarker(m){const [x,y]=this.project(Number(m.dataset.lat),Number(m.dataset.lon));m.style.transform=`translate(${x}px,${y}px) translate(-50%,-50%)`;}
    render(){
      if(!this.w||!this.h)this._measure();
      if(!this.w||!this.h)return;
      const z=this.zoom,n=2**z,{ox:left,oy:top}=this._view();
      const x0=Math.floor(left/TILE)-1,x1=Math.floor((left+this.w)/TILE)+1;
      const y0=Math.max(0,Math.floor(top/TILE)-1),y1=Math.min(n-1,Math.floor((top+this.h)/TILE)+1);
      const visible=new Set();
      for(let tx=x0;tx<=x1;tx++)for(let ty=y0;ty<=y1;ty++){
        const wx=((tx%n)+n)%n, key=`${z}/${tx}/${ty}`;
        visible.add(key);
        let img=this.tileCache.get(key);
        if(!img){
        img=new Image();img.className='ml-tile';img.alt='';img.draggable=false;img.referrerPolicy='strict-origin-when-cross-origin';
        // Nền tối giản, không nhãn (Esri Canvas sáng/tối, không cần key); OSM chỉ dùng khi Esri lỗi.
        const providers=[
          `https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_${this.theme==='light'?'Light':'Dark'}_Gray_Base/MapServer/tile/${z}/${ty}/${wx}`,
          `https://tile.openstreetmap.org/${z}/${wx}/${ty}.png`,
          `https://a.tile.openstreetmap.fr/hot/${z}/${wx}/${ty}.png`
        ];
        let providerIndex=0;img.src=providers[providerIndex];
        img.addEventListener('error',()=>{providerIndex++;if(providerIndex<providers.length){img.src=providers[providerIndex];}else{img.classList.add('tile-error');this.el.dataset.tileError='true';}});
        this.tileCache.set(key,img);
        }
        img.style.left=`${tx*TILE-left}px`;img.style.top=`${ty*TILE-top}px`;
        if(img.parentNode!==this.tiles)this.tiles.appendChild(img);
      }
      for(const [key,img] of this.tileCache){if(!visible.has(key))img.remove();}
      if(this.tileCache.size>200){for(const [key] of this.tileCache){if(!visible.has(key))this.tileCache.delete(key);if(this.tileCache.size<=150)break;}}
      this.markers.forEach(m=>this._placeMarker(m));this._renderOverlay();
    }
    _renderOverlay(){
      const view=this._view();
      if(this.svg._vb!==view.k){this.svg._vb=view.k;this.svg.setAttribute('viewBox',`0 0 ${this.w||1} ${this.h||1}`);}
      // Cập nhật theo id: chỉ sửa thuộc tính của polyline đã có, tạo mới/xóa khi cần (không xóa sạch rồi vẽ lại).
      // Tọa độ màn hình chỉ tính lại khi mảng điểm hoặc góc nhìn đổi.
      // opts.glow: quầng sáng cùng màu bằng 2 nét rộng mờ dần — rẻ hơn nhiều so với bộ lọc blur khi cập nhật liên tục.
      // opts.progress (0..1): chỉ tô phần đầu của đường bằng stroke-dasharray trên pathLength chuẩn hóa,
      // nên đoạn "đã chạy" đổi mỗi khung hình mà không phải cắt/chiếu lại hình học.
      const NS='http://www.w3.org/2000/svg';this._els=this._els||new globalThis.Map();
      const set=(el,k,v)=>{v=String(v);if(el.getAttribute(k)!==v)el.setAttribute(k,v);};
      const unset=(el,k)=>{if(el.hasAttribute(k))el.removeAttribute(k);};
      const mk=()=>{const el=document.createElementNS(NS,'polyline');el.setAttribute('fill','none');el.setAttribute('stroke-linecap','round');el.setAttribute('stroke-linejoin','round');this.svg.appendChild(el);return el;};
      const live=new Set();
      for(const [id,{points,opts}] of this.lines){
        if(!points||points.length<2)continue;live.add(id);
        let rec=this._els.get(id);
        const nHalo=opts.glow?2:0;
        if(!rec||rec.halo.length!==nHalo){if(rec){rec.halo.forEach(e=>e.remove());rec.line.remove();}rec={halo:Array.from({length:nHalo},mk),line:mk()};this._els.set(id,rec);}
        const all=[...rec.halo,rec.line];
        if(rec.pts!==points||rec.vk!==view.k){rec.pts=points;rec.vk=view.k;
          let s='';for(let i=0;i<points.length;i++){const p=points[i];s+=(i?' ':'')+(lonToX(p[1],view.z)-view.ox).toFixed(1)+','+(latToY(p[0],view.z)-view.oy).toFixed(1);}
          for(const el of all)el.setAttribute('points',s);}
        const c=opts.color||'#667085',w=opts.weight||3,g=opts.glow||0,go=opts.glowOpacity??.55;
        rec.halo.forEach((el,i)=>{set(el,'stroke',c);set(el,'stroke-width',(w+g*(i?1.2:2.6)).toFixed(1));set(el,'stroke-opacity',(go*(i?.45:.18)).toFixed(3));});
        const pl=rec.line;set(pl,'stroke',c);set(pl,'stroke-width',w);set(pl,'stroke-opacity',opts.opacity??0.75);
        if(opts.progress!=null){const d=`${(Math.max(0,Math.min(1,opts.progress))*1000).toFixed(2)} 100000`;for(const el of all){set(el,'pathLength',1000);set(el,'stroke-dasharray',d);}}
        else{for(const el of all){unset(el,'pathLength');if(el!==pl||!opts.dash)unset(el,'stroke-dasharray');}if(opts.dash)set(pl,'stroke-dasharray',opts.dash);}
      }
      for(const [id,rec] of this._els)if(!live.has(id)){rec.halo.forEach(e=>e.remove());rec.line.remove();this._els.delete(id);}
    }
    destroy(){this.ro?.disconnect();this.el.innerHTML='';}
  }
  return {Map:SlippyMap};
})();
