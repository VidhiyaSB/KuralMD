// KuralMD voice interview: push-to-talk recorder, TTS player, chat state.
window.interview = function (encounterId, initialLanguage, sttMode) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  return {
    encounterId,
    sttMode: sttMode || 'elevenlabs', // elevenlabs | local | browser | none
    _sr: null, _srText: '', _srConf: null,
    language: initialLanguage,
    turns: [],
    progress: { percent: 0, fields: [] },
    status: 'interview_in_progress',
    questionsAsked: 0,
    urgent: false,
    redFlagReason: '',
    done: false,
    fallback: false,
    prior: null,
    phase: 'idle', // idle | recording | transcribing | thinking | speaking
    speaking: false,
    error: '',
    typed: '',
    showType: false,
    pendingText: '',
    needsStart: true,
    loading: false,
    level: 0,
    recSeconds: 0,
    // recorder internals
    _stream: null, _recorder: null, _chunks: [], _downAt: 0, _toggleMode: false, _timer: null, _analyser: null, _raf: null,
    _pollTimer: null,

    get busy() { return this.phase !== 'idle' && this.phase !== 'speaking'; },
    get currentQuestion() {
      for (let i = this.turns.length - 1; i >= 0; i--) if (this.turns[i].role === 'agent') return this.turns[i];
      return {};
    },

    async init() {
      try {
        const s = await this.api('GET', `/api/interview/${encounterId}/state`);
        this.apply(s);
        if (s.turns.length && (s.done || s.status !== 'interview_in_progress')) { this.needsStart = false; this.pollStatus(); }
      } catch (e) { this.error = e.message; }
    },

    async begin() {
      this.loading = true;
      try {
        // Unlock audio + mic permission in the user gesture.
        this.$refs.player.src = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=';
        this.$refs.player.play().catch(() => {});
        const s = await this.api('POST', `/api/interview/${encounterId}/start`);
        this.apply(s);
        this.needsStart = false;
        this.speakCurrent();
      } catch (e) { this.error = e.message; this.needsStart = false; }
      this.loading = false;
    },

    apply(s) {
      this.turns = s.turns;
      this.progress = s.progress;
      this.status = s.status;
      this.language = s.language;
      this.questionsAsked = s.questions_asked;
      this.urgent = s.urgent;
      this.redFlagReason = s.red_flag_reason;
      this.done = s.done;
      this.fallback = s.fallback;
      this.prior = s.prior;
      this.$nextTick(() => { const c = this.$refs.chat; if (c) c.scrollTop = c.scrollHeight; });
    },

    async api(method, url, body, isForm) {
      const opts = { method, headers: {} };
      if (body && !isForm) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
      if (isForm) opts.body = body;
      const r = await fetch(url, opts);
      if (!r.ok) {
        let msg = `${r.status}`;
        try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch (_) {}
        throw new Error(`Server error: ${msg}`);
      }
      return r.json();
    },

    // ---------------------------------------------------------- speaking
    speakCurrent() {
      const q = this.currentQuestion;
      if (!q.text_original) return;
      this.playTurn(q);
    },
    replay() { this.speakCurrent(); },
    playTurn(t) {
      this.stopSpeaking();
      if (t.audio_url) {
        const p = this.$refs.player;
        p.src = t.audio_url;
        this.speaking = true;
        if (this.phase === 'idle') this.phase = 'speaking';
        p.play().catch(() => { this.speaking = false; if (this.phase === 'speaking') this.phase = 'idle'; this.browserSpeak(t); });
      } else {
        this.browserSpeak(t);
      }
    },
    browserSpeak(t) {
      if (!('speechSynthesis' in window)) return;
      const u = new SpeechSynthesisUtterance(t.text_original);
      u.lang = t.language === 'ta' ? 'ta-IN' : 'en-IN';
      u.rate = 0.95;
      u.onend = () => this.onAudioEnded();
      this.speaking = true;
      if (this.phase === 'idle') this.phase = 'speaking';
      speechSynthesis.speak(u);
    },
    stopSpeaking() {
      try { this.$refs.player.pause(); } catch (_) {}
      if ('speechSynthesis' in window) speechSynthesis.cancel();
      this.speaking = false;
      if (this.phase === 'speaking') this.phase = 'idle';
    },
    onAudioEnded() { this.speaking = false; if (this.phase === 'speaking') this.phase = 'idle'; },

    // ---------------------------------------------------------- recording
    async micDown() {
      if (this.done) return;
      if (this.phase === 'recording') { // second tap in toggle mode
        if (this._toggleMode) this.stopRecording();
        return;
      }
      if (this.busy) return;
      this.stopSpeaking();
      this._downAt = Date.now();
      this._toggleMode = false;
      await this.startRecording();
    },
    micUp() {
      if (this.phase !== 'recording') return;
      if (Date.now() - this._downAt < 400) { this._toggleMode = true; return; } // quick tap -> toggle mode
      if (!this._toggleMode) this.stopRecording();
    },
    micLeave() { if (this.phase === 'recording' && !this._toggleMode && Date.now() - this._downAt > 400) this.stopRecording(); },

    pickMime() {
      const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
      for (const t of types) if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
      return '';
    },

    async startRecording() {
      this.error = '';
      try {
        if (!this._stream) this._stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      } catch (e) {
        this.error = 'Microphone not available. Please allow mic access or type the answer.';
        this.showType = true;
        return;
      }
      const mime = this.pickMime();
      this._chunks = [];
      this._recorder = new MediaRecorder(this._stream, mime ? { mimeType: mime } : undefined);
      this._recorder.ondataavailable = (ev) => { if (ev.data.size) this._chunks.push(ev.data); };
      this._recorder.onstop = () => this.upload(mime);
      this._recorder.start();
      this.startBrowserSTT();
      this.phase = 'recording';
      this.recSeconds = 0;
      this._timer = setInterval(() => { this.recSeconds++; if (this.recSeconds >= 60) this.stopRecording(); }, 1000);
      this.meter();
    },

    meter() {
      try {
        const ctx = this._ctx || (this._ctx = new (window.AudioContext || window.webkitAudioContext)());
        const src = ctx.createMediaStreamSource(this._stream);
        const an = ctx.createAnalyser();
        an.fftSize = 512;
        src.connect(an);
        const data = new Uint8Array(an.frequencyBinCount);
        const tick = () => {
          if (this.phase !== 'recording') { this.level = 0; src.disconnect(); return; }
          an.getByteTimeDomainData(data);
          let sum = 0;
          for (const v of data) sum += Math.abs(v - 128);
          this.level = Math.min(1, (sum / data.length) / 25);
          this._raf = requestAnimationFrame(tick);
        };
        tick();
      } catch (_) {}
    },

    // Chrome Web Speech API (Google) - runs alongside the recorder as an STT fallback.
    startBrowserSTT() {
      this._srText = ''; this._srConf = null; this._sr = null;
      if (!SR) return;
      try {
        const sr = new SR();
        sr.lang = this.language === 'ta' ? 'ta-IN' : 'en-IN';
        sr.continuous = true;
        sr.interimResults = false;
        sr.onresult = (ev) => {
          let text = '', confs = [];
          for (let i = 0; i < ev.results.length; i++) {
            if (ev.results[i].isFinal) { text += ev.results[i][0].transcript + ' '; confs.push(ev.results[i][0].confidence); }
          }
          this._srText = text.trim();
          const valid = confs.filter((c) => c > 0);
          this._srConf = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
        };
        sr.onerror = () => {};
        this._srDone = new Promise((res) => { sr.onend = res; });
        sr.start();
        this._sr = sr;
      } catch (_) { this._sr = null; }
    },
    async browserTranscript() {
      if (!this._sr) return '';
      try { this._sr.stop(); } catch (_) {}
      await Promise.race([this._srDone, new Promise((r) => setTimeout(r, 2500))]);
      return this._srText;
    },

    stopRecording() {
      clearInterval(this._timer);
      if (this._recorder && this._recorder.state !== 'inactive') this._recorder.stop();
      this.phase = 'transcribing';
    },

    async upload(mime) {
      const blob = new Blob(this._chunks, { type: mime || 'audio/webm' });
      const ext = (mime || 'audio/webm').includes('mp4') ? 'mp4' : (mime || '').includes('ogg') ? 'ogg' : 'webm';
      const fd = new FormData();
      fd.append('audio', blob, `answer.${ext}`);
      fd.append('language', this.language);
      const browserText = this.browserTranscript();
      const useBrowser = async (reason) => {
        const text = await browserText;
        if (!text) return false;
        await this.sendAnswer({ text, language: this.language, stt_confidence: this._srConf, input_mode: 'voice', stt_source: 'browser', stt_fallback_reason: reason });
        return true;
      };
      if (this.sttMode === 'browser' || this.sttMode === 'none') {
        if (!(await useBrowser(''))) {
          this.error = SR ? 'No speech detected — please try again or type.' : 'This browser has no speech recognition — please use Chrome or type the answer.';
          this.phase = 'idle';
        }
        return;
      }
      try {
        const r = await this.api('POST', `/api/interview/${encounterId}/transcribe`, fd, true);
        if (!r.ok) {
          if (await useBrowser(r.error)) return;
          this.error = r.error;
          if (/type/i.test(r.error)) this.showType = true;
          this.phase = 'idle';
          return;
        }
        if (!r.text) {
          this.error = this.language === 'ta' ? 'பேச்சு கேக்கல. மறுபடியும் சொல்லுங்க. (No speech detected — please try again.)' : 'No speech detected — please try again.';
          this.phase = 'idle';
          return;
        }
        await this.sendAnswer({ text: r.text, language: r.language || this.language, audio_path: r.audio_path, stt_confidence: r.confidence, input_mode: 'voice' });
      } catch (e) {
        if (await useBrowser(e.message)) return;
        this.error = e.message + ' — you can type the answer instead.';
        this.showType = true;
        this.phase = 'idle';
      }
    },

    async sendTyped() {
      const text = this.typed.trim();
      if (!text) return;
      this.typed = '';
      this.stopSpeaking();
      await this.sendAnswer({ text, language: this.language, input_mode: 'text' });
    },

    async sendAnswer(body) {
      this.pendingText = body.text;
      this.phase = 'thinking';
      try {
        const s = await this.api('POST', `/api/interview/${encounterId}/answer`, body);
        this.pendingText = '';
        this.apply(s);
        this.phase = 'idle';
        this.speakCurrent();
        if (s.done) this.pollStatus();
      } catch (e) {
        this.pendingText = '';
        this.error = e.message;
        this.phase = 'idle';
      }
    },

    async setLanguage(lang) {
      if (lang === this.language) return;
      this.stopSpeaking();
      this.phase = 'thinking';
      try {
        const s = await this.api('POST', `/api/interview/${encounterId}/language`, { language: lang });
        this.apply(s);
        this.phase = 'idle';
        this.speakCurrent();
      } catch (e) { this.error = e.message; this.phase = 'idle'; }
    },

    async finish() {
      if (!confirm('Finish the interview and extract the OP record?')) return;
      this.stopSpeaking();
      try {
        await this.api('POST', `/api/interview/${encounterId}/finish`);
        this.done = true;
        this.status = 'interview_complete';
        this.pollStatus();
      } catch (e) { this.error = e.message; }
    },

    pollStatus() {
      clearInterval(this._pollTimer);
      const check = async () => {
        try {
          const s = await this.api('GET', `/api/encounters/${encounterId}/status`);
          this.status = s.status;
          if (s.status !== 'interview_complete' && s.status !== 'interview_in_progress') clearInterval(this._pollTimer);
        } catch (_) {}
      };
      check();
      this._pollTimer = setInterval(check, 2500);
    },
  };
};
