'use strict';
(() => {
  const data = window.WORKBENCH_TOUR;
  const player = document.getElementById('tourPlayer');
  const video = document.getElementById('tourVideo');
  const toggle = document.getElementById('tourToggle');
  const clock = document.getElementById('tourClock');
  const caption = document.getElementById('captureCaption');
  const buttons = document.getElementById('tourChapters');
  const capture = new URLSearchParams(location.search).has('capture');
  let active = false, lastChapter = null, frame;
  const stamp = t => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  function sync(time, recording = false) {
    time = Math.max(0, Math.min(data.duration, time));
    const chapter = data.chapters.filter(c => c.start <= time).at(-1);
    if (chapter !== lastChapter) {
      window.workbenchSetPalette(chapter.palette);
      window.workbenchSelect(chapter.selection);
      document.body.dataset.tourFocus = chapter.focus;
      [...buttons.children].forEach((b, i) => b.setAttribute('aria-pressed', String(data.chapters[i] === chapter)));
      lastChapter = chapter;
    }
    clock.value = `${stamp(time)} / 01:45`;
    if (recording) {
      caption.hidden = false;
      caption.textContent = data.captions.find(c => c.start <= time && time < c.end)?.text || '';
    }
    window.WORKBENCH_TOUR_STATE = {active, time, selection: chapter.selection, chapter: chapter.label, palette: chapter.palette, focus: chapter.focus};
  }
  function tick() {
    if (!active) return;
    sync(video.currentTime);
    frame = requestAnimationFrame(tick);
  }
  function close() {
    active = false; video.pause(); cancelAnimationFrame(frame); player.hidden = true;
    lastChapter = null; delete document.body.dataset.tourFocus; window.workbenchSelect(null);
    window.WORKBENCH_TOUR_STATE = {active: false};
    if (location.hash === '#tour') history.replaceState(null, '', location.pathname + location.search);
    document.getElementById('demo').focus({preventScroll: true});
  }
  async function play() {
    try { await video.play(); toggle.textContent = 'Pause'; }
    catch { toggle.textContent = 'Play'; }
  }
  function open(audible = false) {
    if (active) return;
    active = true; player.hidden = false; lastChapter = null;
    video.muted = !audible;
    video.currentTime = 0; sync(0); tick(); play();
    if (location.hash !== '#tour') location.hash = 'tour';
    scrollTo(0, 0);
  }
  data.chapters.forEach(chapter => {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = chapter.label; b.setAttribute('aria-pressed', 'false');
    b.onclick = () => { video.currentTime = chapter.start; sync(chapter.start); };
    buttons.append(b);
  });
  toggle.onclick = () => video.paused ? play() : video.pause();
  video.addEventListener('pause', () => { toggle.textContent = video.ended ? 'Replay' : 'Play'; });
  video.addEventListener('play', () => { toggle.textContent = 'Pause'; });
  video.addEventListener('seeked', () => active && sync(video.currentTime));
  video.addEventListener('timeupdate', () => active && sync(video.currentTime));
  video.addEventListener('ended', () => { sync(data.duration); toggle.textContent = 'Replay'; });
  video.addEventListener('error', () => { toggle.textContent = 'Retry'; document.getElementById('status').textContent = 'Video unavailable. Select a chapter to explore the receipts.'; });
  document.getElementById('tourClose').onclick = close;
  document.getElementById('demo').onclick = () => open(true);
  addEventListener('hashchange', () => location.hash === '#tour' ? open() : active && close());
  addEventListener('keydown', e => { if (e.key === 'Escape' && active) close(); });
  window.workbenchTourCapture = time => {
    document.body.dataset.capture = 'true';
    sync(time, true);
  };
  if (capture) document.body.dataset.capture = 'true';
  else if (location.hash === '#tour') open();
})();
