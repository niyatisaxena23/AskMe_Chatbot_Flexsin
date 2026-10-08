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

  // Minimal renderer: paragraphs, "-"/"*" bullets, and bare URLs.
  // Deliberately builds DOM nodes rather than setting innerHTML.
  function render(container, text) {
    var lines = String(text).split('\n');
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
          document.createTextNode(
            text.slice(cursor, match.index)
          )
        );
      }

      var url = match[0].replace(/[.,;:]$/, '');

      var a = el('a', null, url);
      a.href = url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';

      node.appendChild(a);

      cursor = match.index + url.length;
    }

    node.appendChild(
      document.createTextNode(
        text.slice(cursor)
      )
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

    if (!state) {
      input.focus();
    }
  }

  function addStreamingMessage() {
    var wrap = el('div', 'msg bot');

    var content = el('div', 'stream-content');
    wrap.appendChild(content);

    chat.appendChild(wrap);
    scroll();

    return {
      wrap: wrap,
      content: content,
      text: ''
    };
  }

  function renderStreamingText(state) {
    /*
     * Re-render the complete sanitized text.
     *
     * This deliberately uses the existing safe DOM renderer rather than
     * innerHTML, so streamed content cannot become executable HTML.
     */
    state.content.textContent = '';
    render(state.content, state.text);
    scroll();
  }

  function parseSSEBlock(block) {
    var eventName = 'message';
    var dataLines = [];

    block.split('\n').forEach(function (line) {
      if (line.indexOf('event:') === 0) {
        eventName = line.slice(6).trim();
      } else if (line.indexOf('data:') === 0) {
        dataLines.push(line.slice(5).trim());
      }
    });

    if (!dataLines.length) {
      return null;
    }

    var data;

    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch (error) {
      return null;
    }

    return {
      event: eventName,
      data: data
    };
  }

  function processSSEBuffer(buffer, state) {
    var parts = buffer.split('\n\n');

    /*
     * The final element may be incomplete.
     * Keep it for the next network chunk.
     */
    var remainder = parts.pop();

    parts.forEach(function (part) {
      var event = parseSSEBlock(part);

      if (!event) {
        return;
      }

      if (event.event === 'meta') {
        if (event.data.session_id) {
          sessionId = event.data.session_id;
        }

        return;
      }

      if (event.event === 'token') {
        var text = event.data.text || '';

        state.text += text;
        renderStreamingText(state);

        return;
      }

      if (event.event === 'done') {
        if (event.data.session_id) {
          sessionId = event.data.session_id;
        }

        return;
      }

      if (event.event === 'error') {
        state.error = event.data.error || 'Something went wrong.';
      }
    });

    return remainder;
  }

  function ask(question) {
    if (busy || !question) {
      return;
    }

    addMessage('user', question);

    input.value = '';
    setBusy(true);

    var typing = addTyping();

    fetch('/api/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify({
        message: question,
        session_id: sessionId
      })
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json()
            .catch(function () {
              return {};
            })
            .then(function (body) {
              throw new Error(
                body.error ||
                'Something went wrong. Please try again.'
              );
            });
        }

        if (!res.body) {
          throw new Error(
            'Streaming is not supported by this browser.'
          );
        }

        typing.remove();

        var state = addStreamingMessage();
        var reader = res.body.getReader();
        var decoder = new TextDecoder('utf-8');
        var buffer = '';

        function readNext() {
          return reader.read().then(function (result) {
            if (result.done) {
              /*
               * Process a final complete event if one is left.
               */
              if (buffer.trim()) {
                processSSEBuffer(buffer, state);
              }

              if (state.error) {
                state.wrap.remove();
                addMessage('error', state.error);
              }

              return;
            }

            buffer += decoder.decode(
              result.value,
              { stream: true }
            );

            buffer = processSSEBuffer(
              buffer,
              state
            );

            if (state.error) {
              throw new Error(state.error);
            }

            return readNext();
          });
        }

        return readNext();
      })
      .catch(function (error) {
        typing.remove();

        var existingStreaming =
          chat.querySelector(
            '.msg.bot .stream-content'
          );

        if (
          existingStreaming &&
          existingStreaming.textContent
        ) {
          return;
        }

        addMessage(
          'error',
          error.message ||
          'Connection problem — please try again.'
        );
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
      ask(event.target.textContent);
    }
  });

  input.focus();
})();