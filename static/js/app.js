// ==========================================================================
// PhoneCDP — Single-file SPA (Production)
// ==========================================================================

const API = {
    async get(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${r.status}`); return r.json(); },
    async post(u, b) { const r = await fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }); if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.status); } return r.json(); },
    async postForm(u, fd) { const r = await fetch(u, { method: 'POST', body: fd }); if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.status); } return r.json(); },
    async del(u) { const r = await fetch(u, { method: 'DELETE' }); if (!r.ok) throw new Error(r.status); return r.json(); },
    async patch(u, b) { const r = await fetch(u, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }); if (!r.ok) throw new Error(r.status); return r.json(); },
};

// -- Helpers --
function showToast(msg, type = 'info') {
    const t = document.getElementById('toast'), ic = document.getElementById('toast-icon'), m = document.getElementById('toast-message');
    const icons = { success: '<svg class="w-5 h-5 text-green-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>', error: '<svg class="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/></svg>', info: '<svg class="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/></svg>' };
    ic.innerHTML = icons[type] || icons.info; m.textContent = msg;
    t.classList.remove('hidden'); t.classList.add('flex');
    clearTimeout(t._t); t._t = setTimeout(() => { t.classList.add('hidden'); t.classList.remove('flex'); }, 4000);
}
function openImageModal(s) { const m = document.getElementById('image-modal'); document.getElementById('modal-image').src = s; m.classList.remove('hidden'); m.classList.add('flex'); }
function closeImageModal() { const m = document.getElementById('image-modal'); m.classList.add('hidden'); m.classList.remove('flex'); }

function vBadge(v) {
    const c = { AUTHENTIC: 'bg-green-100 text-green-800 border-green-200', SUSPICIOUS: 'bg-yellow-100 text-yellow-800 border-yellow-200', COUNTERFEIT: 'bg-red-100 text-red-800 border-red-200' };
    return `<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold border ${c[v] || 'bg-gray-100 text-gray-800 border-gray-200'}">${v}</span>`;
}
function confGauge(val) {
    const p = Math.round(val * 100), col = val >= .7 ? 'bg-green-500' : val >= .5 ? 'bg-yellow-500' : 'bg-red-500';
    return `<div class="w-full"><div class="flex justify-between text-xs mb-1"><span class="font-medium">Confidence</span><span class="font-bold">${p}%</span></div><div class="w-full bg-gray-200 rounded-full h-2.5 relative"><div class="${col} h-2.5 rounded-full transition-all duration-500" style="width:${p}%"></div><div class="absolute top-0 h-full border-l-2 border-gray-400" style="left:50%" title="Suspicious"></div><div class="absolute top-0 h-full border-l-2 border-gray-600" style="left:70%" title="Authentic"></div></div><div class="flex justify-between text-[10px] text-gray-400 mt-0.5"><span>Counterfeit</span><span style="margin-left:20%">Suspicious</span><span>Authentic</span></div></div>`;
}
function sBar(label, val, wt) {
    const p = Math.round(val * 100), col = val >= .7 ? 'bg-green-400' : val >= .5 ? 'bg-yellow-400' : 'bg-red-400';
    return `<div class="flex items-center gap-3 text-sm"><span class="w-28 text-gray-600">${label} <span class="text-gray-400 text-xs">(${Math.round(wt*100)}%)</span></span><div class="flex-1 bg-gray-200 rounded-full h-2"><div class="${col} h-2 rounded-full" style="width:${p}%"></div></div><span class="w-12 text-right font-mono text-xs">${val.toFixed(3)}</span></div>`;
}
function fmtDate(iso) { const d = new Date(iso); return d.toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'})+' '+d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}); }
function spin(t='Loading...') { return `<div class="flex items-center justify-center py-16"><div class="flex flex-col items-center gap-3"><div class="w-8 h-8 border-4 border-brand-200 border-t-brand-600 rounded-full animate-spin"></div><span class="text-sm text-gray-500">${t}</span></div></div>`; }

// ==========================================================================
// Router
// ==========================================================================
function getRoute() {
    const h = window.location.hash.slice(1) || '/';
    if (h.startsWith('/results/')) return { fn: pgDetail, p: { id: h.split('/')[2] } };
    return { fn: { '/': pgDash, '/generate': pgGen, '/verify': pgVerify, '/results': pgResults }[h] || pgDash, p: {} };
}
function navigate() {
    const { fn, p } = getRoute();
    document.querySelectorAll('.nav-link').forEach(l => {
        const pg = l.dataset.page, h = window.location.hash.slice(1) || '/';
        const on = (pg === 'dashboard' && h === '/') || (pg && h.startsWith('/' + pg));
        l.classList.toggle('bg-brand-700', on);
    });
    fn(document.getElementById('app'), p);
}
window.addEventListener('hashchange', navigate);
window.addEventListener('load', navigate);

// ==========================================================================
// Dashboard
// ==========================================================================
async function pgDash(el) {
    el.innerHTML = spin('Loading dashboard...');
    try {
        const [stats, recent] = await Promise.all([API.get('/api/results/stats'), API.get('/api/results?limit=8')]);
        el.innerHTML = `
        <div class="mb-8"><h1 class="text-2xl font-bold">Dashboard</h1><p class="text-gray-500 mt-1">Copy Detection Pattern verification overview</p></div>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
            <div class="bg-white rounded-xl shadow-sm border p-5"><div class="text-sm text-gray-500 mb-1">Total Patterns</div><div class="text-3xl font-bold text-brand-800">${stats.total_patterns}</div></div>
            <div class="bg-white rounded-xl shadow-sm border p-5"><div class="text-sm text-gray-500 mb-1">Total Verifications</div><div class="text-3xl font-bold text-brand-800">${stats.total_verifications}</div></div>
            <div class="bg-white rounded-xl shadow-sm border p-5"><div class="text-sm text-gray-500 mb-1">Pass Rate</div><div class="text-3xl font-bold ${stats.pass_rate>=70?'text-green-600':stats.pass_rate>=50?'text-yellow-600':'text-red-600'}">${stats.pass_rate}%</div></div>
            <div class="bg-white rounded-xl shadow-sm border p-5"><div class="text-sm text-gray-500 mb-1">Avg Confidence</div><div class="text-3xl font-bold text-brand-800">${(stats.avg_confidence*100).toFixed(0)}%</div></div>
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-8">
            <div class="bg-green-50 rounded-xl border border-green-200 p-5 flex items-center gap-4"><div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center"><svg class="w-6 h-6 text-green-600" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg></div><div><div class="text-2xl font-bold text-green-800">${stats.verdicts.authentic}</div><div class="text-sm text-green-600">Authentic</div></div></div>
            <div class="bg-yellow-50 rounded-xl border border-yellow-200 p-5 flex items-center gap-4"><div class="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center"><svg class="w-6 h-6 text-yellow-600" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg></div><div><div class="text-2xl font-bold text-yellow-800">${stats.verdicts.suspicious}</div><div class="text-sm text-yellow-600">Suspicious</div></div></div>
            <div class="bg-red-50 rounded-xl border border-red-200 p-5 flex items-center gap-4"><div class="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center"><svg class="w-6 h-6 text-red-600" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/></svg></div><div><div class="text-2xl font-bold text-red-800">${stats.verdicts.counterfeit}</div><div class="text-sm text-red-600">Counterfeit</div></div></div>
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="bg-white rounded-xl shadow-sm border p-6"><h2 class="text-lg font-semibold mb-4">Quick Actions</h2><div class="space-y-3"><a href="#/generate" class="block w-full bg-brand-600 text-white text-center py-3 rounded-lg font-medium hover:bg-brand-700 transition">Generate New Pattern</a><a href="#/verify" class="block w-full bg-white text-brand-700 text-center py-3 rounded-lg font-medium border-2 border-brand-200 hover:bg-brand-50 transition">Verify a Photo</a><a href="/api/results/export" class="block w-full bg-white text-gray-700 text-center py-3 rounded-lg font-medium border border-gray-200 hover:bg-gray-50 transition" download>Export CSV</a></div></div>
            <div class="lg:col-span-2 bg-white rounded-xl shadow-sm border p-6"><div class="flex justify-between items-center mb-4"><h2 class="text-lg font-semibold">Recent Verifications</h2><a href="#/results" class="text-sm text-brand-600 hover:underline">View all</a></div>${recent.results.length===0?'<p class="text-gray-400 text-sm py-8 text-center">No verifications yet.</p>':`<div class="overflow-x-auto"><table class="w-full text-sm"><thead><tr class="text-left text-gray-500 border-b"><th class="pb-2 font-medium">Date</th><th class="pb-2 font-medium">Pattern</th><th class="pb-2 font-medium">Verdict</th><th class="pb-2 font-medium text-right">Confidence</th></tr></thead><tbody>${recent.results.map(r=>`<tr class="border-b border-gray-50 hover:bg-gray-50 cursor-pointer" onclick="location.hash='#/results/${r.id}'"><td class="py-2.5 text-gray-600">${fmtDate(r.created_at)}</td><td class="py-2.5">${r.pattern_label||r.pattern_serial}</td><td class="py-2.5">${vBadge(r.verdict)}</td><td class="py-2.5 text-right font-mono">${(r.confidence*100).toFixed(1)}%</td></tr>`).join('')}</tbody></table></div>`}</div>
        </div>`;
    } catch(e) { el.innerHTML = `<div class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">${e.message}</div>`; }
}

// ==========================================================================
// Generate
// ==========================================================================
async function pgGen(el) {
    let patterns = []; try { patterns = await API.get('/api/patterns'); } catch(e) {}
    el.innerHTML = `
    <div class="mb-8"><h1 class="text-2xl font-bold">Generate Pattern</h1><p class="text-gray-500 mt-1">Create a new Copy Detection Pattern for printing and verification</p></div>
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="bg-white rounded-xl shadow-sm border p-6"><h2 class="text-lg font-semibold mb-4">Pattern Settings</h2>
            <form id="gf" class="space-y-4">
                <div><label class="block text-sm font-medium text-gray-700 mb-1">Serial Number</label><input type="text" id="gs" value="SN-${new Date().getFullYear()}-${String(patterns.length+1).padStart(5,'0')}" class="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none"/></div>
                <div><label class="block text-sm font-medium text-gray-700 mb-1">Label / Name</label><input type="text" id="gl" placeholder="e.g. Product Batch A" class="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none"/></div>
                <div><label class="block text-sm font-medium text-gray-700 mb-1">Seed <span class="text-gray-400 font-normal">(empty = random)</span></label><input type="number" id="gd" placeholder="Auto" class="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none"/></div>
                <div><label class="block text-sm font-medium text-gray-700 mb-1">Notes</label><textarea id="gn" rows="2" placeholder="Optional..." class="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-500 focus:border-brand-500 outline-none resize-none"></textarea></div>
                <button type="submit" id="gb" class="w-full bg-brand-600 text-white py-3 rounded-lg font-medium hover:bg-brand-700 transition flex items-center justify-center gap-2"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>Generate Pattern</button>
            </form>
        </div>
        <div class="lg:col-span-2"><div id="gp" class="bg-white rounded-xl shadow-sm border p-6"><h2 class="text-lg font-semibold mb-4">Preview</h2><div class="flex items-center justify-center h-80 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200"><p class="text-gray-400 text-sm">Generate a pattern to see the preview</p></div></div>
        ${patterns.length?`<div class="bg-white rounded-xl shadow-sm border p-6 mt-6"><h2 class="text-lg font-semibold mb-4">Gallery <span class="text-sm font-normal text-gray-400">(${patterns.length})</span></h2><div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">${patterns.map(p=>`<div class="group relative bg-gray-50 rounded-lg border overflow-hidden hover:shadow-md transition cursor-pointer" onclick="openImageModal('/api/patterns/${p.id}/preview')"><img src="/api/patterns/${p.id}/preview" class="w-full aspect-square object-cover" loading="lazy"/><div class="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-2"><div class="text-white text-xs font-medium truncate">${p.label||p.serial_number}</div><div class="text-white/70 text-[10px]">${fmtDate(p.created_at)}</div></div></div>`).join('')}</div></div>`:''}</div>
    </div>`;
    document.getElementById('gf').addEventListener('submit', async e => {
        e.preventDefault(); const b = document.getElementById('gb');
        b.disabled = true; b.innerHTML = '<div class="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> Generating...';
        try {
            const sv = document.getElementById('gd').value;
            const r = await API.post('/api/patterns/generate', { serial_number: document.getElementById('gs').value, label: document.getElementById('gl').value, seed: sv ? parseInt(sv) : null, notes: document.getElementById('gn').value });
            document.getElementById('gp').innerHTML = `<div class="flex justify-between items-center mb-4"><h2 class="text-lg font-semibold">Generated Pattern</h2><div class="flex gap-2"><div class="relative" id="dl-dd"><button onclick="document.getElementById('dl-menu').classList.toggle('hidden')" class="inline-flex items-center gap-1.5 bg-brand-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-brand-700 transition"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>Download<svg class="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg></button><div id="dl-menu" class="hidden absolute right-0 mt-1 w-56 bg-white border border-gray-200 rounded-lg shadow-lg z-10 py-1"><a href="/api/patterns/${r.id}/download" download class="block px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50"><div class="font-medium">PNG Image</div><div class="text-xs text-gray-400">Raw ${r.pattern_size}x${r.pattern_size}px pattern</div></a><div class="border-t border-gray-100 my-1"></div><a href="/api/patterns/${r.id}/pdf?size_mm=15" download class="block px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50"><div class="font-medium">PDF - 15mm x 15mm</div><div class="text-xs text-gray-400">A4 page, pattern centered</div></a><a href="/api/patterns/${r.id}/pdf?size_mm=7.5" download class="block px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50"><div class="font-medium">PDF - 7.5mm x 7.5mm</div><div class="text-xs text-gray-400">A4 page, pattern centered</div></a></div></div><a href="#/verify" class="inline-flex items-center gap-1.5 bg-white text-brand-700 px-4 py-2 rounded-lg text-sm font-medium border border-brand-200 hover:bg-brand-50 transition">Verify This</a></div></div><div class="flex justify-center"><img src="/api/patterns/${r.id}/preview" class="max-w-md w-full rounded-lg shadow-md cursor-pointer border" onclick="openImageModal(this.src)"/></div><div class="mt-4 grid grid-cols-2 gap-4 text-sm"><div class="bg-gray-50 rounded-lg p-3"><span class="text-gray-500">Serial:</span> <span class="font-medium">${r.serial_number}</span></div><div class="bg-gray-50 rounded-lg p-3"><span class="text-gray-500">Seed:</span> <span class="font-mono font-medium">${r.seed}</span></div><div class="bg-gray-50 rounded-lg p-3"><span class="text-gray-500">Size:</span> <span class="font-medium">${r.pattern_size}x${r.pattern_size}px</span></div><div class="bg-gray-50 rounded-lg p-3"><span class="text-gray-500">Created:</span> <span class="font-medium">${fmtDate(r.created_at)}</span></div></div>`;
            document.addEventListener('click', function _clDd(e) { const dd = document.getElementById('dl-dd'), m = document.getElementById('dl-menu'); if (dd && m && !dd.contains(e.target)) m.classList.add('hidden'); if (!document.getElementById('dl-dd')) document.removeEventListener('click', _clDd); });
            showToast('Pattern generated!', 'success');
        } catch(err) { showToast(err.message, 'error'); }
        finally { b.disabled = false; b.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>Generate Pattern'; }
    });
}

// ==========================================================================
// Verify
// ==========================================================================
async function pgVerify(el) {
    let patterns = []; try { patterns = await API.get('/api/patterns'); } catch(e) {}
    if (!patterns.length) { el.innerHTML = `<div class="mb-8"><h1 class="text-2xl font-bold">Verify Pattern</h1></div><div class="bg-yellow-50 border border-yellow-200 rounded-xl p-8 text-center"><h3 class="text-lg font-semibold text-yellow-800 mb-2">No Patterns Found</h3><p class="text-yellow-700 mb-4">Generate a pattern first.</p><a href="#/generate" class="inline-block bg-brand-600 text-white px-6 py-2.5 rounded-lg font-medium hover:bg-brand-700">Generate</a></div>`; return; }
    el.innerHTML = `
    <div class="mb-8"><h1 class="text-2xl font-bold">Verify Pattern</h1><p class="text-gray-500 mt-1">Upload a phone photo to verify against the original</p></div>
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="space-y-5">
            <div class="bg-white rounded-xl shadow-sm border p-5"><div class="flex items-center gap-2 mb-3"><span class="w-6 h-6 bg-brand-600 text-white rounded-full flex items-center justify-center text-xs font-bold">1</span><h3 class="font-semibold">Select Pattern</h3></div><select id="vp" class="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:ring-2 focus:ring-brand-500 outline-none">${patterns.map(p=>`<option value="${p.id}">${p.label||p.serial_number} (Seed: ${p.seed})</option>`).join('')}</select><div id="vt" class="mt-3"><img src="/api/patterns/${patterns[0].id}/preview" class="w-full rounded-lg border cursor-pointer" onclick="openImageModal(this.src)"/></div></div>
            <div class="bg-white rounded-xl shadow-sm border p-5"><div class="flex items-center gap-2 mb-3"><span class="w-6 h-6 bg-brand-600 text-white rounded-full flex items-center justify-center text-xs font-bold">2</span><h3 class="font-semibold">Upload Photo</h3></div><div id="dz" class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-brand-400 hover:bg-brand-50/30 transition cursor-pointer"><svg class="w-10 h-10 mx-auto mb-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg><p class="text-sm text-gray-500 mb-1">Drag & drop photo</p><p class="text-xs text-gray-400">or click to browse</p><input type="file" id="vf" accept="image/*" multiple class="hidden"/></div><div id="fl" class="mt-3 hidden space-y-2"></div></div>
            <div class="bg-white rounded-xl shadow-sm border p-5"><div class="flex items-center gap-2 mb-3"><span class="w-6 h-6 bg-brand-600 text-white rounded-full flex items-center justify-center text-xs font-bold">3</span><h3 class="font-semibold">Options</h3></div><div class="space-y-3"><div><label class="block text-sm text-gray-600 mb-1">Print Size (mm)</label><input type="number" id="vs" placeholder="e.g. 65" class="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-500 outline-none"/></div><div><label class="block text-sm text-gray-600 mb-1">Notes</label><input type="text" id="vn" placeholder="Optional" class="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-500 outline-none"/></div></div></div>
            <button id="vb" disabled class="w-full bg-brand-600 text-white py-3 rounded-lg font-medium hover:bg-brand-700 transition disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>Verify</button>
        </div>
        <div class="lg:col-span-2"><div id="vr" class="bg-white rounded-xl shadow-sm border p-6"><h2 class="text-lg font-semibold mb-4">Results</h2><div class="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200"><p class="text-gray-400 text-sm">Upload a photo to see results</p></div></div></div>
    </div>`;
    const ps = document.getElementById('vp'), dz = document.getElementById('dz'), fi = document.getElementById('vf'), vb = document.getElementById('vb');
    let files = [];
    ps.addEventListener('change', () => { document.getElementById('vt').innerHTML = `<img src="/api/patterns/${ps.value}/preview" class="w-full rounded-lg border cursor-pointer" onclick="openImageModal(this.src)"/>`; });
    dz.addEventListener('click', () => fi.click());
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('border-brand-500','bg-brand-50'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('border-brand-500','bg-brand-50'));
    dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('border-brand-500','bg-brand-50'); setFiles(e.dataTransfer.files); });
    fi.addEventListener('change', () => setFiles(fi.files));
    function setFiles(f) {
        files = Array.from(f); const fl = document.getElementById('fl');
        if (files.length) { fl.classList.remove('hidden'); fl.innerHTML = files.map(f => `<div class="flex items-center gap-2 bg-gray-50 rounded-lg p-2"><svg class="w-5 h-5 text-gray-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg><span class="text-sm truncate flex-1">${f.name}</span><span class="text-xs text-gray-400">${(f.size/1024).toFixed(0)}KB</span></div>`).join(''); vb.disabled = false; }
        else { fl.classList.add('hidden'); vb.disabled = true; }
    }
    vb.addEventListener('click', async () => {
        if (!files.length) return;
        vb.disabled = true; vb.innerHTML = '<div class="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> Verifying...';
        const vr = document.getElementById('vr'); vr.innerHTML = spin('Running verification...');
        try {
            const pid = ps.value, sz = document.getElementById('vs').value, nt = document.getElementById('vn').value;
            if (files.length === 1) {
                const fd = new FormData(); fd.append('captured', files[0]); fd.append('pattern_id', pid);
                if (sz) fd.append('print_size_mm', sz); if (nt) fd.append('notes', nt);
                const r = await API.postForm('/api/verify', fd);
                vr.innerHTML = `<div class="flex justify-between items-start mb-6"><div><h2 class="text-lg font-semibold">Result</h2><p class="text-sm text-gray-500">${fmtDate(r.created_at)}</p></div><div class="flex items-center gap-3">${vBadge(r.verdict)}<a href="#/results/${r.id}" class="text-sm text-brand-600 hover:underline">Details</a></div></div><div class="mb-6">${confGauge(r.confidence)}</div><div class="mb-6 space-y-2.5"><h3 class="text-sm font-semibold text-gray-700 mb-2">Scores</h3>${sBar('Moire',r.scores.moire,r.weights.moire)}${sBar('Color',r.scores.color,r.weights.color)}${sBar('Correlation',r.scores.correlation,r.weights.correlation)}${sBar('Gradient',r.scores.gradient,r.weights.gradient)}</div><div class="grid grid-cols-2 gap-3 mb-6"><div class="bg-gray-50 rounded-lg p-3"><div class="text-xs text-gray-500">Markers</div><div class="font-bold text-lg">${r.markers_found}/4</div></div><div class="bg-gray-50 rounded-lg p-3"><div class="text-xs text-gray-500">Alignment</div><div class="font-bold text-lg capitalize">${r.alignment_method}</div></div></div><h3 class="text-sm font-semibold text-gray-700 mb-3">Images</h3><div class="grid grid-cols-3 gap-3"><div><p class="text-xs text-gray-500 mb-1 text-center">Original</p><img src="/api/verify/${r.id}/images/original" class="w-full rounded-lg border cursor-pointer aspect-square object-cover" onclick="openImageModal(this.src)"/></div><div><p class="text-xs text-gray-500 mb-1 text-center">Captured</p><img src="/api/verify/${r.id}/images/captured" class="w-full rounded-lg border cursor-pointer aspect-square object-cover" onclick="openImageModal(this.src)"/></div><div><p class="text-xs text-gray-500 mb-1 text-center">Aligned</p><img src="/api/verify/${r.id}/images/aligned" class="w-full rounded-lg border cursor-pointer aspect-square object-cover" onclick="openImageModal(this.src)"/></div></div>`;
            } else {
                const fd = new FormData(); files.forEach(f => fd.append('captured_files', f)); fd.append('pattern_id', pid);
                if (sz) fd.append('print_size_mm', sz); if (nt) fd.append('notes', nt);
                const d = await API.postForm('/api/verify/batch', fd);
                const ok = d.results.filter(r => r.verdict === 'AUTHENTIC').length;
                vr.innerHTML = `<div class="flex justify-between items-start mb-6"><div><h2 class="text-lg font-semibold">Batch Results</h2><p class="text-sm text-gray-500">${d.results.length} photos</p></div><div class="text-right"><div class="text-2xl font-bold ${ok===d.results.length?'text-green-600':'text-yellow-600'}">${ok}/${d.results.length}</div><div class="text-xs text-gray-500">Passed</div></div></div><div class="space-y-3">${d.results.map(r=>`<div class="flex items-center gap-4 bg-gray-50 rounded-lg p-4 ${r.id?'cursor-pointer hover:bg-gray-100':''}" ${r.id?`onclick="location.hash='#/results/${r.id}'"`:''} ><div class="flex-1"><div class="font-medium text-sm">${r.filename}</div><div class="text-xs text-gray-500 mt-0.5">${r.verdict!=='ERROR'?`Confidence: ${(r.confidence*100).toFixed(1)}%`:r.error||'Error'}</div></div>${vBadge(r.verdict)}</div>`).join('')}</div>`;
            }
            showToast('Verification complete!', 'success');
        } catch(err) { vr.innerHTML = `<h2 class="text-lg font-semibold mb-4">Results</h2><div class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">${err.message}</div>`; showToast(err.message,'error'); }
        finally { vb.disabled = false; vb.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>Verify'; }
    });
}

// ==========================================================================
// Results
// ==========================================================================
async function pgResults(el) {
    el.innerHTML = spin('Loading results...');
    try {
        const d = await API.get('/api/results?limit=100');
        el.innerHTML = `
        <div class="flex justify-between items-center mb-6"><div><h1 class="text-2xl font-bold">Results</h1><p class="text-gray-500 mt-1">${d.total} verifications</p></div><a href="/api/results/export" download class="inline-flex items-center gap-1.5 bg-white text-gray-700 px-4 py-2 rounded-lg text-sm font-medium border hover:bg-gray-50 transition"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>Export CSV</a></div>
        <div class="bg-white rounded-xl shadow-sm border p-4 mb-6"><div class="flex flex-wrap gap-3 items-center"><span class="text-sm text-gray-500">Filter:</span><button class="fbtn px-3 py-1.5 rounded-lg text-sm font-medium border bg-brand-50 text-brand-700 border-brand-200" data-f="all">All</button><button class="fbtn px-3 py-1.5 rounded-lg text-sm font-medium border bg-white text-gray-600 border-gray-200" data-f="AUTHENTIC">Authentic</button><button class="fbtn px-3 py-1.5 rounded-lg text-sm font-medium border bg-white text-gray-600 border-gray-200" data-f="SUSPICIOUS">Suspicious</button><button class="fbtn px-3 py-1.5 rounded-lg text-sm font-medium border bg-white text-gray-600 border-gray-200" data-f="COUNTERFEIT">Counterfeit</button></div></div>
        ${!d.results.length?'<div class="bg-white rounded-xl shadow-sm border p-12 text-center"><h3 class="text-lg font-semibold text-gray-600 mb-2">No Results</h3><a href="#/verify" class="inline-block bg-brand-600 text-white px-6 py-2.5 rounded-lg font-medium hover:bg-brand-700">Verify a Photo</a></div>':`<div class="bg-white rounded-xl shadow-sm border overflow-hidden"><table class="w-full text-sm"><thead><tr class="text-left text-gray-500 bg-gray-50 border-b"><th class="px-5 py-3 font-medium">Date</th><th class="px-5 py-3 font-medium">Pattern</th><th class="px-5 py-3 font-medium">Verdict</th><th class="px-5 py-3 font-medium text-right">Confidence</th><th class="px-5 py-3 font-medium text-center">Markers</th><th class="px-5 py-3 font-medium">Alignment</th><th class="px-5 py-3 font-medium text-center">mm</th><th class="px-5 py-3 font-medium text-center">Actions</th></tr></thead><tbody>${d.results.map(r=>`<tr class="rr border-b border-gray-50 hover:bg-gray-50" data-v="${r.verdict}"><td class="px-5 py-3 text-gray-600">${fmtDate(r.created_at)}</td><td class="px-5 py-3"><div class="font-medium">${r.pattern_label||r.pattern_serial}</div>${r.notes?`<div class="text-xs text-gray-400 truncate max-w-[150px]">${r.notes}</div>`:''}</td><td class="px-5 py-3">${vBadge(r.verdict)}</td><td class="px-5 py-3 text-right font-mono">${(r.confidence*100).toFixed(1)}%</td><td class="px-5 py-3 text-center">${r.markers_found}/4</td><td class="px-5 py-3 capitalize">${r.alignment_method}</td><td class="px-5 py-3 text-center text-gray-500">${r.print_size_mm||'-'}</td><td class="px-5 py-3 text-center"><div class="flex items-center justify-center gap-1"><a href="#/results/${r.id}" class="text-brand-600 hover:text-brand-800 p-1" title="View"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg></a><button onclick="delResult(${r.id},event)" class="text-red-400 hover:text-red-600 p-1" title="Delete"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg></button></div></td></tr>`).join('')}</tbody></table></div>`}`;
        document.querySelectorAll('.fbtn').forEach(b => b.addEventListener('click', () => {
            document.querySelectorAll('.fbtn').forEach(x => { x.className = 'fbtn px-3 py-1.5 rounded-lg text-sm font-medium border bg-white text-gray-600 border-gray-200'; });
            b.className = 'fbtn px-3 py-1.5 rounded-lg text-sm font-medium border bg-brand-50 text-brand-700 border-brand-200';
            const f = b.dataset.f; document.querySelectorAll('.rr').forEach(r => r.style.display = (f==='all'||r.dataset.v===f)?'':'none');
        }));
    } catch(e) { el.innerHTML = `<div class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">${e.message}</div>`; }
}
async function delResult(id, ev) { ev.stopPropagation(); if (!confirm('Delete this result?')) return; try { await API.del(`/api/results/${id}`); showToast('Deleted','success'); pgResults(document.getElementById('app')); } catch(e) { showToast(e.message,'error'); } }

// ==========================================================================
// Detail
// ==========================================================================
async function pgDetail(el, p) {
    el.innerHTML = spin();
    try {
        const r = await API.get(`/api/verify/${p.id}`);
        el.innerHTML = `
        <div class="mb-6"><a href="#/results" class="inline-flex items-center gap-1 text-sm text-brand-600 hover:underline mb-3"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>Back</a><div class="flex justify-between items-start"><div><h1 class="text-2xl font-bold">Verification #${r.id}</h1><p class="text-gray-500 mt-1">${fmtDate(r.created_at)}</p></div>${vBadge(r.verdict)}</div></div>
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="space-y-5">
                <div class="bg-white rounded-xl shadow-sm border p-5"><h3 class="font-semibold mb-3">Confidence</h3>${confGauge(r.confidence)}</div>
                <div class="bg-white rounded-xl shadow-sm border p-5"><h3 class="font-semibold mb-4">Scores</h3><div class="space-y-3">${sBar('Moire',r.scores.moire,.4)}${sBar('Color',r.scores.color,.3)}${sBar('Correlation',r.scores.correlation,.2)}${sBar('Gradient',r.scores.gradient,.1)}</div></div>
                <div class="bg-white rounded-xl shadow-sm border p-5"><h3 class="font-semibold mb-3">Details</h3><div class="space-y-3 text-sm"><div class="flex justify-between"><span class="text-gray-600">Markers</span><div class="flex gap-1">${[1,2,3,4].map(i=>`<div class="w-5 h-5 rounded-full border-2 ${i<=r.markers_found?'bg-green-400 border-green-500':'bg-gray-100 border-gray-300'} flex items-center justify-center text-[10px] text-white font-bold">${i<=r.markers_found?'&#10003;':''}</div>`).join('')}</div></div><div class="flex justify-between"><span class="text-gray-600">Alignment</span><span class="font-medium capitalize">${r.alignment_method}</span></div><div class="flex justify-between"><span class="text-gray-600">Pattern</span><span class="font-medium">${r.pattern_label||r.pattern_serial}</span></div>${r.print_size_mm?`<div class="flex justify-between"><span class="text-gray-600">Print Size</span><span class="font-medium">${r.print_size_mm}mm</span></div>`:''}</div></div>
                <div class="bg-white rounded-xl shadow-sm border p-5"><h3 class="font-semibold mb-3">Notes</h3><div class="flex gap-2"><input type="text" id="dn" value="${r.notes||''}" placeholder="Add notes..." class="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-brand-500 outline-none"/><button onclick="saveNotes(${r.id})" class="bg-brand-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-brand-700">Save</button></div></div>
            </div>
            <div class="lg:col-span-2"><div class="bg-white rounded-xl shadow-sm border p-5"><h3 class="font-semibold mb-4">Image Comparison</h3><div class="grid grid-cols-3 gap-4"><div><p class="text-sm text-gray-500 mb-2 text-center font-medium">Original</p><img src="/api/verify/${r.id}/images/original" class="w-full rounded-lg border cursor-pointer shadow-sm hover:shadow-md transition" onclick="openImageModal(this.src)"/></div><div><p class="text-sm text-gray-500 mb-2 text-center font-medium">Captured</p><img src="/api/verify/${r.id}/images/captured" class="w-full rounded-lg border cursor-pointer shadow-sm hover:shadow-md transition" onclick="openImageModal(this.src)"/></div><div><p class="text-sm text-gray-500 mb-2 text-center font-medium">Aligned</p><img src="/api/verify/${r.id}/images/aligned" class="w-full rounded-lg border cursor-pointer shadow-sm hover:shadow-md transition" onclick="openImageModal(this.src)"/></div></div></div><div class="bg-white rounded-xl shadow-sm border p-5 mt-5"><div class="flex justify-between items-center"><div><h3 class="font-semibold">Quick Re-test</h3><p class="text-sm text-gray-500 mt-0.5">Verify another photo against the same pattern</p></div><a href="#/verify" class="bg-brand-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-brand-700">New Verification</a></div></div></div>
        </div>`;
    } catch(e) { el.innerHTML = `<a href="#/results" class="text-sm text-brand-600 hover:underline">Back</a><div class="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800 mt-3">${e.message}</div>`; }
}
async function saveNotes(id) { try { await API.patch(`/api/results/${id}/notes`, { notes: document.getElementById('dn').value }); showToast('Notes saved','success'); } catch(e) { showToast(e.message,'error'); } }
