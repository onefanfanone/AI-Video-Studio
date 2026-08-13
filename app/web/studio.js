(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function setAction(action) {
    const field = $("#form-action");
    if (field) field.value = action;
  }

  $$('[data-submit]').forEach((button) => {
    button.addEventListener('click', () => {
      setAction(button.dataset.submit);
      const form = button.closest('form') || $('#wizard-form');
      if (form) form.requestSubmit();
    });
  });

  const wizard = $('#wizard-form');
  if (wizard) {
    let timer;
    $$('input, textarea, select', wizard).forEach((control) => {
      control.addEventListener('input', () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
          const body = new URLSearchParams(new FormData(wizard));
          body.set('action', 'autosave');
          fetch('/wizard/action', { method: 'POST', body }).catch(() => {});
        }, 800);
      });
    });
  }

  const modeInputs = $$('input[name="mode"]');
  function updateMode() {
    const mode = $('input[name="mode"]:checked')?.value || 'direct';
    $$('[data-mode]').forEach((element) => {
      element.hidden = !element.dataset.mode.split(' ').includes(mode);
    });
  }
  modeInputs.forEach((input) => input.addEventListener('change', updateMode));
  updateMode();

  const visualStrategy = $('select[name="visual_strategy"]');
  function updateVisualStrategy() {
    const value = visualStrategy?.value || '';
    $$('[data-visual]').forEach((element) => {
      element.hidden = !element.dataset.visual.split(' ').includes(value);
    });
  }
  visualStrategy?.addEventListener('change', updateVisualStrategy);
  updateVisualStrategy();

  const voiceSelect = $('#voice-profile');
  const audio = $('#voice-preview');
  voiceSelect?.addEventListener('change', () => {
    if (!audio) return;
    const option = voiceSelect.options[voiceSelect.selectedIndex];
    const src = option.dataset.preview || '';
    audio.pause();
    audio.removeAttribute('src');
    if (src) audio.src = src;
    audio.load();
  });

  const previewText = $('#preview-subtitle');
  const previewSamples = { eight: '风起长安梦未央', short: '金币', keyword: '尿液征税' };
  $$('.preview-tabs button').forEach((button) => {
    button.addEventListener('click', () => {
      $$('.preview-tabs button').forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      if (previewText) previewText.textContent = previewSamples[button.dataset.preview] || '';
    });
  });
  function updatePreview() {
    if (!previewText) return;
    const value = (name, fallback) => $(`[name="${name}"]`)?.value || fallback;
    previewText.style.fontFamily = `"${value('subtitle_font_name', 'Microsoft YaHei')}", sans-serif`;
    previewText.style.fontSize = `${Math.max(24, Number(value('subtitle_font_size', 118)) * .36)}px`;
    previewText.style.color = value('subtitle_base_color', '#FFE3EC');
    previewText.style.webkitTextStroke = `${Math.max(1, Number(value('subtitle_outline', 8)) * .25)}px ${value('subtitle_outline_color', '#FF5C91')}`;
    previewText.style.bottom = `${Math.max(8, Number(value('subtitle_margin_bottom', 460)) / 19.2)}%`;
    const shadow = Math.max(0, Number(value('subtitle_shadow', 5)));
    previewText.style.textShadow = `0 ${shadow}px ${shadow}px rgba(80,20,45,.8)`;
    const fadeIn = Number(value('subtitle_fade_in_ms', 150));
    const fadeOut = Number(value('subtitle_fade_out_ms', 150));
    previewText.classList.toggle('fade', fadeIn > 0 || fadeOut > 0);
  }
  $$('[name^="subtitle_"]').forEach((input) => input.addEventListener('input', updatePreview));
  updatePreview();

  const profileSelect = $('select[name="subtitle_profile"]');
  profileSelect?.addEventListener('change', () => {
    const option = profileSelect.options[profileSelect.selectedIndex];
    let settings = {};
    try { settings = JSON.parse(option.dataset.settings || '{}'); } catch (_) {}
    const mapping = {
      font_name: 'subtitle_font_name', font_size: 'subtitle_font_size',
      base_color: 'subtitle_base_color', highlight_color: 'subtitle_highlight_color',
      outline_color: 'subtitle_outline_color', outline: 'subtitle_outline',
      shadow: 'subtitle_shadow', margin_bottom: 'subtitle_margin_bottom',
      fade_in_ms: 'subtitle_fade_in_ms', fade_out_ms: 'subtitle_fade_out_ms',
      max_chars_per_line: 'subtitle_max_chars', max_lines: 'subtitle_max_lines'
    };
    Object.entries(mapping).forEach(([key, name]) => {
      const input = $(`[name="${name}"]`);
      if (input && settings[key] !== undefined) input.value = settings[key];
    });
    updatePreview();
  });

  const chars = $('#script-stats');
  const scriptBox = $('textarea[name="final_script"]') || $('textarea[name="original_script"]');
  function updateStats() {
    if (!chars || !scriptBox) return;
    const effective = (scriptBox.value.match(/[\u3400-\u9fffA-Za-z0-9]/g) || []).length;
    chars.textContent = `${effective} 个有效字 · 预计 ${(effective / 4.2).toFixed(1)} 秒`;
  }
  scriptBox?.addEventListener('input', updateStats);
  updateStats();

  $$('[data-confirm]').forEach((button) => {
    button.addEventListener('click', (event) => {
      if (!window.confirm(button.dataset.confirm)) event.preventDefault();
    });
  });
})();
