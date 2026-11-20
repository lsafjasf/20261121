(() => {
  const D = window.REVIEW_DATA;
  if (!D) return;
  const duration = Number(D.meta.duration) || 1;
  const plain = document.getElementById("video-plain");
  const outlined = document.getElementById("video-outlined");
  const videos = { plain, outlined };
  const playBtn = document.getElementById("play-btn");
  const scrub = document.getElementById("scrub");
  const timecode = document.getElementById("timecode");
  const speed = document.getElementById("speed");
  const firstRisk = document.getElementById("first-risk");
  const timeline = document.getElementById("timeline");
  const panes = {
    plain: document.getElementById("pane-plain"),
    outlined: document.getElementById("pane-outlined"),
  };
  let current = 0;
  let activeVersion = "plain";

  const fmt = (value) => {
    const safe = Math.max(0, Number(value) || 0);
    const minutes = Math.floor(safe / 60);
    const seconds = safe - minutes * 60;
    return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
  };
  const clamp = (value) => Math.max(0, Math.min(duration, Number(value) || 0));

  function updateClock(time) {
    current = clamp(time);
    scrub.value = String(current);
    timecode.textContent = `${fmt(current)} / ${fmt(duration)}`;
    const line = document.getElementById("playhead");
    if (line) {
      const x = 52 + (current / duration) * 930;
      line.setAttribute("x1", x.toFixed(1));
      line.setAttribute("x2", x.toFixed(1));
    }
  }

  function setActivePane(version) {
    activeVersion = version;
    Object.entries(panes).forEach(([key, pane]) => {
      pane.classList.toggle("active", key === version);
    });
  }

  function setActiveCard(cueId) {
    document.querySelectorAll(".segment-card").forEach((card) => {
      card.classList.toggle("active", card.dataset.cue === cueId);
    });
  }

  function seekBoth(time, version = activeVersion, cueId = null, scroll = false) {
    const target = clamp(time);
    Object.values(videos).forEach((video) => {
      video.pause();
      if (video.readyState >= 1) {
        video.currentTime = target;
      } else {
        video.addEventListener("loadedmetadata", () => { video.currentTime = target; }, { once: true });
      }
    });
    updateClock(target);
    setActivePane(version);
    if (cueId) setActiveCard(cueId);
    try {
      const url = new URL(location.href);
      url.searchParams.set("t", target.toFixed(3));
      url.searchParams.set("v", version);
      if (cueId) url.searchParams.set("cue", cueId);
      history.replaceState(null, "", url);
    } catch (_) { /* file:// fallback */ }
    if (scroll) {
      document.querySelector(".player-shell").scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  function playBoth() {
    const target = current >= duration - 0.04 ? 0 : current;
    Object.values(videos).forEach((video) => { video.currentTime = target; });
    Promise.all(Object.values(videos).map((video) => video.play())).catch(() => {});
    playBtn.textContent = "Ⅱ";
    playBtn.setAttribute("aria-label", "暂停");
  }

  function pauseBoth() {
    Object.values(videos).forEach((video) => video.pause());
    playBtn.textContent = "▶";
    playBtn.setAttribute("aria-label", "播放");
  }

  playBtn.addEventListener("click", () => {
    if (Object.values(videos).some((video) => !video.paused)) pauseBoth();
    else playBoth();
  });
  Object.values(videos).forEach((video) => {
    video.addEventListener("click", () => {
      if (video.paused) playBoth(); else pauseBoth();
    });
  });

  scrub.addEventListener("input", () => seekBoth(Number(scrub.value)));
  speed.addEventListener("change", () => {
    Object.values(videos).forEach((video) => { video.playbackRate = Number(speed.value); });
  });
  plain.addEventListener("timeupdate", () => {
    if (Math.abs(outlined.currentTime - plain.currentTime) > 0.12) {
      outlined.currentTime = plain.currentTime;
    }
    if (!plain.paused) updateClock(plain.currentTime);
  });
  outlined.addEventListener("timeupdate", () => {
    if (Math.abs(plain.currentTime - outlined.currentTime) > 0.12) {
      plain.currentTime = outlined.currentTime;
    }
  });
  plain.addEventListener("ended", pauseBoth);

  document.addEventListener("click", (event) => {
    const button = event.target.closest(".jump");
    if (!button) return;
    seekBoth(
      Number(button.dataset.time),
      button.dataset.version || "plain",
      button.dataset.cue || null,
      button.dataset.scroll === "true",
    );
  });

  firstRisk.addEventListener("click", () => {
    const segment = D.segments.find((item) => item.risk) || D.segments[0];
    if (segment) seekBoth(segment.worst.time, segment.worst.version, segment.id, true);
  });

  function svgEl(name, attrs = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  }

  function xOf(time) {
    return 52 + (clamp(time) / duration) * 930;
  }
  function yOf(value) {
    const low = Math.log10(1);
    const high = Math.log10(21);
    const v = Math.max(1, Math.min(21, Number(value) || 1));
    return 154 - ((Math.log10(v) - low) / (high - low)) * 132;
  }

  function drawTimeline() {
    timeline.setAttribute("viewBox", "0 0 1000 190");
    const bg = svgEl("rect", { x: 0, y: 0, width: 1000, height: 190, fill: "transparent" });
    timeline.appendChild(bg);
    [1, 2, 3, 5, 10, 20].forEach((value) => {
      const y = yOf(value);
      timeline.appendChild(svgEl("line", {
        x1: 52, x2: 982, y1: y, y2: y,
        stroke: value === 3 ? "rgba(242,189,77,.55)" : "rgba(255,255,255,.075)",
        "stroke-width": 1, "stroke-dasharray": value === 3 ? "5 5" : "2 5",
      }));
      const label = svgEl("text", { x: 43, y: y + 3, "text-anchor": "end", class: "axis" });
      label.textContent = `${value}:1`;
      timeline.appendChild(label);
    });
    D.segments.forEach((segment, index) => {
      const x0 = xOf(segment.start);
      const x1 = xOf(segment.end);
      timeline.appendChild(svgEl("rect", {
        x: x0, y: 16, width: Math.max(1, x1 - x0), height: 142,
        fill: index % 2 ? "rgba(255,255,255,.018)" : "rgba(90,127,160,.035)",
      }));
      const tag = svgEl("text", { x: x0 + 4, y: 13, class: "axis", fill: "#667785" });
      tag.textContent = segment.id;
      timeline.appendChild(tag);
      segment.plain_episodes.forEach((episode) => {
        timeline.appendChild(svgEl("rect", {
          x: xOf(episode.start), y: 159, width: Math.max(1, xOf(episode.end) - xOf(episode.start)),
          height: 4, fill: episode.kind === "sustained" ? "#ff6b62" : "#ff9d57",
        }));
      });
      if (segment.outlined_episodes.length) {
        segment.outlined_episodes.forEach((episode) => {
          timeline.appendChild(svgEl("rect", {
            x: xOf(episode.start), y: 165, width: Math.max(1, xOf(episode.end) - xOf(episode.start)),
            height: 3, fill: "#55d6ae",
          }));
        });
      }
      const makePath = (values, color, dash) => {
        const commands = segment.series.times.map((time, point) => {
          const value = values[point];
          return `${point ? "L" : "M"}${xOf(time).toFixed(1)},${yOf(value).toFixed(1)}`;
        }).join(" ");
        timeline.appendChild(svgEl("path", {
          d: commands, fill: "none", stroke: color, "stroke-width": 2,
          "stroke-linejoin": "round", "stroke-dasharray": dash || "none",
        }));
      };
      makePath(segment.series.plain, "#ff6b62");
      makePath(segment.series.outlined, "#55d6ae", "5 4");
      const markerX = xOf(segment.worst.time);
      const markerY = yOf(segment.at_worst.plain.contrast_p20);
      timeline.appendChild(svgEl("circle", {
        cx: markerX, cy: markerY, r: 4, fill: "#080b0f", stroke: "#f2bd4d", "stroke-width": 2,
      }));
      const label = svgEl("text", {
        x: markerX, y: Math.max(26, markerY - 9), "text-anchor": "middle", class: "axis",
        fill: "#f2bd4d", "font-weight": 600,
      });
      label.textContent = segment.worst.contrast.toFixed(1);
      timeline.appendChild(label);
    });
    for (let tick = 0; tick <= duration; tick += duration <= 20 ? 2 : 5) {
      const x = xOf(tick);
      const text = svgEl("text", { x, y: 185, "text-anchor": "middle", class: "axis" });
      text.textContent = `${tick}s`;
      timeline.appendChild(text);
    }
    timeline.appendChild(svgEl("line", {
      id: "playhead", x1: 52, x2: 52, y1: 16, y2: 158,
      stroke: "#f2bd4d", "stroke-width": 1, opacity: ".8",
    }));
    timeline.addEventListener("click", (event) => {
      const rect = timeline.getBoundingClientRect();
      const viewX = ((event.clientX - rect.left) / rect.width) * 1000;
      const time = ((viewX - 52) / 930) * duration;
      const nearest = D.segments.reduce((best, segment) => {
        const distance = Math.min(Math.abs(time - segment.start), Math.abs(time - segment.end));
        return distance < best.distance ? { segment, distance } : best;
      }, { segment: null, distance: Infinity }).segment;
      seekBoth(time, "plain", nearest && time >= nearest.start && time <= nearest.end ? nearest.id : null);
    });
    updateClock(current);
  }

  function finishInitialSeek() {
    const params = new URLSearchParams(location.search);
    const explicit = params.get("t") !== null;
    const cue = params.get("cue");
    const version = params.get("v") === "outlined" ? "outlined" : "plain";
    if (explicit) {
      seekBoth(Number(params.get("t")), version, cue);
      return;
    }
    const first = D.segments.find((segment) => segment.risk) || D.segments[0];
    if (first) seekBoth(first.worst.time, first.worst.version, first.id);
  }

  drawTimeline();
  if (plain.readyState >= 1) finishInitialSeek();
  else plain.addEventListener("loadedmetadata", finishInitialSeek, { once: true });
})();
