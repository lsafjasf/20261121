// 坐标系 / 旋转方向 / 360° 可复现性验证（与页面内自检同一套数学）。
import { encodeFOA, rotationMatrix, rotateFOA, decodeStereo, wrapDeg } from './foa.js';

let failures = 0;
function check(name, cond, detail = '') {
  const ok = !!cond;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  (' + detail + ')' : ''}`);
  if (!ok) failures++;
}
const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

// 1) ACN/SN3D 编码: 前/左/上 三个方向的通道响应
const front = encodeFOA(0, 0);
check('前方声源 -> X=1, Y=Z=0', near(front[3], 1) && near(front[1], 0) && near(front[2], 0) && near(front[0], 1));
const left = encodeFOA(90, 0);
check('左侧声源 -> Y=1, X=Z=0', near(left[1], 1) && near(left[3], 0) && near(left[2], 0));
const top = encodeFOA(0, 90);
check('上方声源 -> Z=1, X=Y=0', near(top[2], 1) && near(top[3], 0) && near(top[1], 0));

// 2) 头部朝前时立体声解码: 左源偏左, 前源居中
const id = rotationMatrix(0, 0, 0);
const [lL, lR] = decodeStereo(rotateFOA(left, id));
check('左源解码 L>R', lL > lR, `L=${lL.toFixed(3)} R=${lR.toFixed(3)}`);
const [fL, fR] = decodeStereo(rotateFOA(front, id));
check('前源解码 L=R', near(fL, fR), `L=${fL.toFixed(3)}`);

// 3) 旋转方向: 头向左转 90°(yaw=+90), 前方固定声源应移到右侧(Y<0, R>L)
const yawed = rotateFOA(front, rotationMatrix(90, 0, 0));
check('头左转90° 前源 -> Y=-1(右侧)', near(yawed[1], -1) && near(yawed[3], 0));
const [yL, yR] = decodeStereo(yawed);
check('头左转90° 解码 R>L', yR > yL, `L=${yL.toFixed(3)} R=${yR.toFixed(3)}`);

// 4) 抬头 90°(pitch=+90): 上方声源应来到正前方(X=1)
const pitched = rotateFOA(top, rotationMatrix(0, 90, 0));
check('抬头90° 顶源 -> X=1(正前)', near(pitched[3], 1) && near(pitched[2], 0));

// 5) 侧方声源 + 头左转 90°(yaw=+90): 左源应来到正前方
const turned = rotateFOA(left, rotationMatrix(90, 0, 0));
check('头左转90° 左源 -> X=1(正前)', near(turned[3], 1) && near(turned[1], 0));

// 5b) 右耳下沉 90°(roll=+90): 上方声源应转到左侧(Y=1)
const rolled = rotateFOA(top, rotationMatrix(0, 0, 90));
check('右倾90° 顶源 -> Y=1(左侧)', near(rolled[1], 1) && near(rolled[2], 0));

// 6) 环绕一圈可复现: yaw 0->360 后输出与初始完全一致
const m0 = rotationMatrix(0, 15, -10);
const m360 = rotationMatrix(360, 15, -10);
for (const [name, src] of [['前', front], ['左', left], ['上', top]]) {
  const a = decodeStereo(rotateFOA(src, m0));
  const b = decodeStereo(rotateFOA(src, m360));
  check(`环绕360° ${name}源输出可复现`, near(a[0], b[0]) && near(a[1], b[1]),
    `dL=${Math.abs(a[0]-b[0]).toExponential(2)} dR=${Math.abs(a[1]-b[1]).toExponential(2)}`);
}

// 7) 分步累加 36x10° 与直接 360° 等价（模拟拖动环绕）
let acc = rotationMatrix(0, 0, 0);
const step = rotationMatrix(10, 0, 0);
const mul = (a, b) => a.map((r, i) => r.map((_, j) => a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j]));
for (let i = 0; i < 36; i++) acc = mul(acc, step);
const o1 = decodeStereo(rotateFOA(front, acc));
const o0 = decodeStereo(rotateFOA(front, rotationMatrix(0, 0, 0)));
check('分步旋转 36x10° 回到原位', near(o1[0], o0[0], 1e-9) && near(o1[1], o0[1], 1e-9));

// 8) wrapDeg
check('wrapDeg(360)=0, wrapDeg(270)=-90', wrapDeg(360) === 0 && wrapDeg(270) === -90);

console.log(failures === 0 ? '\n全部通过' : `\n${failures} 项失败`);
process.exit(failures === 0 ? 0 : 1);
