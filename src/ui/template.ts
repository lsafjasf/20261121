export const appTemplate = `
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">∷</div>
        <div>
          <h1>发言区间审稿台</h1>
          <p>长视频 · 实际呈现锚定 · <span id="session-mode-label">修复版</span></p>
        </div>
      </div>
      <div class="top-actions">
        <label class="mode-control" title="切换模式会重建当前审稿会话">
          <span>逻辑</span>
          <select id="mode-select">
            <option value="fixed">修复版</option>
            <option value="legacy">旧逻辑</option>
          </select>
        </label>
        <button id="race-button" class="button button-ghost" type="button">回放竞态</button>
        <button id="add-source-button" class="button button-solid" type="button">打开视频</button>
        <input id="file-input" type="file" accept="video/*,.mkv,.mov,.mp4,.webm" multiple hidden />
        <button id="export-button" class="button button-accent" type="button">导出区间</button>
      </div>
    </header>

    <div id="legacy-banner" class="legacy-banner" hidden>
      旧逻辑模式只用于复现：迟到帧没有源/跳转票据校验，导出可能被污染。
    </div>

    <main class="layout">
      <section class="stage-column" aria-label="媒体审阅区">
        <div class="stage-card">
          <div class="stage-head">
            <div class="now-playing">
              <span class="eyebrow">当前来源</span>
              <strong id="active-source-name">尚未载入视频</strong>
            </div>
            <div class="media-identity" id="active-source-id">media —</div>
          </div>

          <div class="video-shell" id="video-shell">
            <video id="video" playsinline preload="metadata"></video>
            <div class="empty-state" id="empty-state">
              <div class="drop-target" id="drop-target">
                <span class="drop-icon">↳</span>
                <strong>拖入一段长视频</strong>
                <span>或使用右上角“打开视频”</span>
                <small>文件指纹只读取首尾各 512 KB</small>
              </div>
            </div>
            <div class="presentation-badge" id="presentation-badge">等待画面</div>
            <div class="recording-hud" id="recording-hud" hidden>
              <span class="recording-dot"></span>
              <span id="recording-hud-label">记录中</span>
            </div>
          </div>

          <div class="transport">
            <button id="play-button" class="transport-button" type="button" aria-label="播放">▶</button>
            <button id="back-button" class="transport-button" type="button" aria-label="后退 5 秒">−5s</button>
            <button id="forward-button" class="transport-button" type="button" aria-label="前进 5 秒">+5s</button>
            <div class="timeline-wrap">
              <input id="timeline" type="range" min="0" max="1000" value="0" step="1" aria-label="播放头" />
              <div class="timeline-labels">
                <span id="current-time">00:00.000</span>
                <span id="duration-time">00:00.000</span>
              </div>
            </div>
            <button id="mute-button" class="transport-button" type="button" aria-label="静音">声</button>
          </div>

          <div class="hold-bar">
            <div class="hold-copy">
              <span class="hold-key">R</span>
              <span><strong>按住记录</strong>，松开结束；重复按键不会新建区间</span>
            </div>
            <div class="status-strip">
              <span id="record-status" class="status-pill">待机</span>
              <span class="status-meta">迟到帧拦截 <b id="stale-count">0</b></span>
              <span class="status-meta">区间 <b id="interval-count">0</b></span>
            </div>
          </div>
        </div>

        <section class="explain-card">
          <div>
            <span class="eyebrow">锚定规则</span>
            <h2>以画面真正呈现的 <code>mediaTime</code> 为准</h2>
          </div>
          <p>
            每个帧票据冻结“来源代数 + 跳转代数”。换源、拖动播放头或反向呈现后，旧票据不再有资格修改当前区间；记录中的区间会在最后有效前向位置关闭，并闩锁到松键。
          </p>
        </section>
      </section>

      <aside class="sidebar">
        <section class="panel">
          <div class="panel-head">
            <div>
              <span class="eyebrow">媒体</span>
              <h2>来源与草稿</h2>
            </div>
            <button id="side-add-button" class="icon-button" type="button" title="打开视频">＋</button>
          </div>
          <div id="source-list" class="source-list"></div>
        </section>

        <section class="panel interval-panel">
          <div class="panel-head">
            <div>
              <span class="eyebrow">审阅</span>
              <h2>区间明细</h2>
            </div>
            <span id="panel-interval-count" class="count-badge">0</span>
          </div>
          <div id="interval-list" class="interval-list"></div>
        </section>

        <section class="panel journal-panel">
          <div class="panel-head compact">
            <div>
              <span class="eyebrow">诊断</span>
              <h2>事件记录</h2>
            </div>
            <button id="clear-session-button" class="text-button" type="button">清空会话</button>
          </div>
          <ol id="journal-list" class="journal-list"></ol>
        </section>
      </aside>
    </main>

    <div id="toast" class="toast" role="status" aria-live="polite"></div>
    <dialog id="race-dialog" class="race-dialog">
      <div class="dialog-head">
        <div>
          <span class="eyebrow">确定性事件回放</span>
          <h2>换源迟到帧对照</h2>
        </div>
        <button id="race-close" class="icon-button" type="button" aria-label="关闭">×</button>
      </div>
      <p class="dialog-copy">A 源 12.000s 的帧被故意延迟 300ms，在切换到 B 源并开始新区间后到达。</p>
      <div id="race-result" class="race-result"></div>
      <button id="race-rerun" class="button button-solid" type="button">重新回放</button>
    </dialog>
  </div>
`;
