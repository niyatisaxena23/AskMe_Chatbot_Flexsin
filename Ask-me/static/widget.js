(function () {
  var chat = document.getElementById('chat');
  var form = document.getElementById('composer');
  var input = document.getElementById('input');
  var send = document.getElementById('send');
  var chips = document.getElementById('chips');
  var sessionId = null;
  var busy = false;

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  }

  function scroll() {
    chat.scrollTop = chat.scrollHeight;
  }

  function addMessage(role, text) {
    var wrap = el('div', 'msg ' + role);
    render(wrap, text);
    chat.appendChild(wrap);
    scroll();
    return wrap;
  }

  function render(container, text) {
    var lines = String(text || '').split('\n');
    var list = null;

    lines.forEach(function (line) {
      var trimmed = line.trim();

      if (!trimmed) {
        list = null;
        return;
      }

      var bullet = trimmed.match(/^[-*•]\s+(.*)$/);

      if (bullet) {
        if (!list) {
          list = el('ul');
          container.appendChild(list);
        }

        var li = el('li');
        linkify(li, bullet[1]);
        list.appendChild(li);
        return;
      }

      list = null;

      var p = el('p');
      linkify(p, trimmed);
      container.appendChild(p);
    });
  }

  function linkify(node, text) {
    var pattern = /https?:\/\/[^\s<>")\]]+/g;
    var cursor = 0;
    var match;

    while ((match = pattern.exec(text)) !== null) {
      if (match.index > cursor) {
        node.appendChild(
          document.createTextNode(text.slice(cursor, match.index))
        );
      }

      var rawUrl = match[0];
      var url = rawUrl.replace(/[.,;:!?]+$/, '');

      if (url) {
        var a = el('a', null, url);
        a.href = url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        node.appendChild(a);
      }

      cursor = match.index + url.length;
    }

    node.appendChild(
      document.createTextNode(text.slice(cursor))
    );
  }

  function addTyping() {
    var wrap = el('div', 'msg bot typing');

    for (var i = 0; i < 3; i++) {
      wrap.appendChild(el('span'));
    }

    chat.appendChild(wrap);
    scroll();

    return wrap;
  }

  function setBusy(state) {
    busy = state;
    send.disabled = state;
    input.disabled = state;

    if (!state) input.focus();
  }

  function ask(question) {
    if (busy || !question) return;

    addMessage('user', question);
    input.value = '';
    setBusy(true);

    var typing = addTyping();

    fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        message: question,
        session_id: sessionId
      })
    })
      .then(function (response) {
        return response.text().then(function (text) {
          var body;

          try {
            body = text ? JSON.parse(text) : {};
          } catch (error) {
            throw new Error(
              'The server returned an invalid response. Check the Flask terminal.'
            );
          }

          if (!response.ok) {
            throw new Error(
              body.error || 'Request failed with status ' + response.status + '.'
            );
          }

          return body;
        });
      })
      .then(function (body) {
        typing.remove();

        if (body.error) {
          addMessage('error', body.error);
          return;
        }

        if (body.session_id) {
          sessionId = body.session_id;
        }

        if (typeof body.reply !== 'string' || !body.reply.trim()) {
          addMessage(
            'error',
            'The server returned no answer. Please try again.'
          );
          return;
        }

        addMessage('bot', body.reply);
      })
      .catch(function (error) {
        typing.remove();

        addMessage(
          'error',
          error.message || 'Connection problem — please try again.'
        );

        console.error('Ask Me request failed:', error);
      })
      .finally(function () {
        setBusy(false);
      });
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    ask(input.value.trim());
  });

  chips.addEventListener('click', function (event) {
    if (event.target.tagName === 'BUTTON') {
      ask(event.target.textContent.trim());
    }
  });

  input.focus();
})();
