(function () {
  var token = sessionStorage.getItem('askme_admin_token') || '';
  var gate = document.getElementById('gate');
  var admin = document.getElementById('admin');
  var mode = 'paste';

  function api(path, options) {
    options = options || {};
    options.headers = Object.assign({ 'X-Admin-Token': token }, options.headers || {});
    return fetch(path, options).then(function (res) {
      return res.json().then(function (body) { return { ok: res.ok, body: body }; });
    });
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  // ---------------------------------------------------------------- gate

  document.getElementById('gate-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var value = document.getElementById('token').value.trim();
    fetch('/api/admin/auth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: value })
    }).then(function (res) {
      if (!res.ok) {
        document.getElementById('gate-err').textContent = 'That token was rejected.';
        return;
      }
      token = value;
      sessionStorage.setItem('askme_admin_token', token);
      unlock();
    });
  });

  function unlock() {
    gate.hidden = true;
    admin.hidden = false;
    refresh();
  }

  // ------------------------------------------------------------- sources

  function refresh() {
    api('/api/admin/sources').then(function (result) {
      if (!result.ok) { relock(); return; }
      var data = result.body;
      document.getElementById('stats').textContent =
        data.chunks + ' chunks · ' + data.sources.length + ' sources · ' +
        (data.embeddings_ok ? 'embeddings active' : 'embeddings OFF (keyword only)');
      renderSources(data.sources);
    });
  }

  function relock() {
    sessionStorage.removeItem('askme_admin_token');
    gate.hidden = false;
    admin.hidden = true;
  }

  function renderSources(sources) {
    var box = document.getElementById('sources');
    box.textContent = '';
    if (!sources.length) {
      box.appendChild(el('p', 'empty', 'Nothing indexed yet.'));
      return;
    }
    sources.forEach(function (source) {
      var row = el('div', 'src');
      row.appendChild(el('span', 'tag', source.origin));

      var grow = el('div', 'grow');
      grow.appendChild(el('div', null, source.title));
      grow.appendChild(el('div', 'muted small',
        source.type + ' · ' + source.chunks + ' chunk' + (source.chunks === 1 ? '' : 's') +
        ' · ' + source.chars.toLocaleString() + ' chars'));
      row.appendChild(grow);

      var del = el('button', 'ghost', 'Remove');
      del.addEventListener('click', function () {
        if (!confirm('Remove "' + source.title + '" from the index?')) return;
        api('/api/admin/sources/' + source.source_id, { method: 'DELETE' }).then(refresh);
      });
      row.appendChild(del);
      box.appendChild(row);
    });
  }

  // -------------------------------------------------------------- ingest

  document.getElementById('tabs').addEventListener('click', function (e) {
    if (e.target.tagName !== 'BUTTON') return;
    mode = e.target.dataset.tab;
    [].forEach.call(this.children, function (b) { b.classList.toggle('on', b === e.target); });
    [].forEach.call(document.querySelectorAll('.pane'), function (p) {
      p.hidden = p.dataset.pane !== mode;
    });
  });

  document.getElementById('ingest-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var note = document.getElementById('ingest-note');
    var btn = document.getElementById('ingest-btn');
    var title = document.getElementById('title').value.trim();
    var type = document.getElementById('type').value;
    var request;

    if (mode === 'file') {
      var picked = document.getElementById('file').files[0];
      if (!picked) { note.className = 'note bad'; note.textContent = 'Choose a file first.'; return; }
      var form = new FormData();
      form.append('file', picked);
      form.append('title', title);
      form.append('type', type);
      request = api('/api/admin/ingest', { method: 'POST', body: form });
    } else {
      var text = document.getElementById('text').value;
      if (!text.trim()) { note.className = 'note bad'; note.textContent = 'Paste some text first.'; return; }
      request = api('/api/admin/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title, type: type, text: text })
      });
    }

    btn.disabled = true;
    note.className = 'note';
    note.textContent = 'Embedding…';

    request.then(function (result) {
      btn.disabled = false;
      if (!result.ok) {
        note.className = 'note bad';
        note.textContent = result.body.error || 'Failed.';
        return;
      }
      note.className = 'note ok';
      note.textContent = 'Added "' + result.body.title + '" — ' + result.body.chunks +
        ' chunk(s) from ' + result.body.chars.toLocaleString() + ' characters.';
      document.getElementById('text').value = '';
      document.getElementById('file').value = '';
      document.getElementById('title').value = '';
      refresh();
    });
  });

  // -------------------------------------------------------------- search

  document.getElementById('search-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var query = document.getElementById('query').value.trim();
    if (!query) return;
    var box = document.getElementById('results');
    box.textContent = '';
    box.appendChild(el('p', 'empty', 'Searching…'));

    api('/api/admin/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    }).then(function (result) {
      box.textContent = '';
      if (!result.ok) {
        box.appendChild(el('p', 'empty', result.body.error || 'Search failed.'));
        return;
      }
      var hits = result.body.results;
      if (!hits.length) {
        box.appendChild(el('p', 'empty',
          'No match — the assistant would reply "I don\'t have specific information on that."'));
        return;
      }
      hits.forEach(function (hit) {
        var wrap = el('div', 'hit');
        var head = el('div', 'hit-head');
        head.appendChild(el('strong', null, hit.title));

        var signals = [];
        if (hit.rank_dense) signals.push('semantic #' + hit.rank_dense + ' (' + hit.similarity.toFixed(2) + ')');
        if (hit.rank_keyword) signals.push('keyword #' + hit.rank_keyword);
        head.appendChild(el('span', 'hit-meta', signals.join(' · ')));
        wrap.appendChild(head);

        var body = el('div', 'hit-text', hit.text);
        body.title = 'Click to expand';
        body.addEventListener('click', function () { body.classList.toggle('open'); });
        wrap.appendChild(body);
        box.appendChild(wrap);
      });
    });
  });

  if (token) unlock(); else document.getElementById('token').focus();
})();
