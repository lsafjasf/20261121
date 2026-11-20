// 端到端冒烟测试：真实浏览器中验证页面自检、通道响应方向与环绕可复现性。
// 运行: node e2e.mjs
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';

const PORT = 8765;
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 800));

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '  (' + detail + ')' : ''}`);
  if (!cond) failures++;
};
const num = (s) => (s === '-∞' ? -Infinity : parseFloat(s));

try {
  const browser = await chromium.launch({
    args: ['--autoplay-policy=no-user-gesture-required', '--mute-audio'],
  });
  const page = await browser.newPage();
  page.on('pageerror', (err) => { console.log('PAGE ERROR:', err.message); failures++; });
  await page.goto(`http://127.0.0.1:${PORT}/index.html`);

  await page.waitForFunction(() => document.getElementById('selftest').textContent.includes('PASS'));
  const selftest = await page.$eval('#selftest', (el) => el.textContent);
  check('页面数学自检全部通过', !selftest.includes('FAIL'));

  const meters = async () => page.$$eval('.db', (els) => els.map((e) => e.textContent));
  const setYaw = async (v) => page.$eval('#yaw', (el, val) => {
    el.value = String(val);
    el.dispatchEvent(new Event('input'));
  }, v);

  // 前方声源: X 通道应显著高于 Y/Z，L≈R
  await page.click('#srcFront');
  await page.waitForTimeout(700);
  let m = (await meters()).map(num);
  check('前源: X 远高于 Y/Z', m[3] > m[1] + 20 && m[3] > m[2] + 20, `Y=${m[1]} Z=${m[2]} X=${m[3]}`);
  check('前源: L≈R', Math.abs(m[4] - m[5]) < 1.5, `L=${m[4]} R=${m[5]}`);

  // 头左转 90°: 前源应转到右侧 -> Y 通道升高, R>L
  await setYaw(90);
  await page.waitForTimeout(600);
  m = (await meters()).map(num);
  check('头左转90°: Y 通道远高于 X', m[1] > m[3] + 20, `Y=${m[1]} X=${m[3]}`);
  check('头左转90°: R>L', m[5] > m[4] + 3, `L=${m[4]} R=${m[5]}`);
  await setYaw(0);

  // 上方声源: Z 通道应显著高于 X/Y
  await page.click('#srcTop');
  await page.waitForTimeout(700);
  m = (await meters()).map(num);
  check('顶源: Z 远高于 X/Y', m[2] > m[3] + 20 && m[2] > m[1] + 20, `Y=${m[1]} Z=${m[2]} X=${m[3]}`);

  // 侧方(左)声源: Y 通道高, L>R
  await page.click('#srcSide');
  await page.waitForTimeout(700);
  m = (await meters()).map(num);
  check('左源: Y 远高于 X/Z', m[1] > m[3] + 20 && m[1] > m[2] + 20, `Y=${m[1]} Z=${m[2]} X=${m[3]}`);
  check('左源: L>R', m[4] > m[5] + 3, `L=${m[4]} R=${m[5]}`);

  // 环绕一圈: 输出应可复现
  await page.click('#srcFront');
  await page.waitForTimeout(400);
  await page.click('#orbit');
  await page.waitForFunction(
    () => document.getElementById('status').textContent.includes('环绕一圈'),
    { timeout: 15000 },
  );
  const status = await page.$eval('#status', (el) => el.textContent);
  check('环绕360° 输出可复现', status.includes('验证通过'), status);

  await browser.close();
} finally {
  server.kill();
}

console.log(failures === 0 ? '\nE2E 全部通过' : `\nE2E ${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
