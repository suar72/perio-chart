// On-device reader for scan.html. Everything runs in the browser; the photo never leaves the device.
//
// 1. Ink: dark pixels of the flattened photo that are not part of the printed blank chart (template.js).
// 2. Per chart row, ink blobs are assigned to the cell their centre falls in, so a "10" that spills
//    over a cell edge still lands in one cell; blobs that overlap horizontally form one character.
// 3. Each character goes through a small CNN trained on MNIST (model/digits.js, tools/train-digits.py).
// 4. Values are checked against what the cell allows (PD 0-15, mobility 0-3, furcation I-III).
// 5. A struck-out tooth is a column whose two diagonals are mostly ink.
// 6. Prognosis letters (G F P Q H) use a second CNN trained on EMNIST (letters.js); HN, Age and Date
//    reuse the digit CNN, with date slashes recognised by shape; checkboxes count ink inside the box.
// 7. Patient, Examiner, Diagnosis and Treatment plan are not read: their ink is cut out as images.
(function () {
  const MM = {
    inkThreshold: 32,      // adaptive threshold offset (higher = only darker strokes count)
    templateGrow: 7,       // px the printed lines are widened by before subtraction (absorbs misalignment)
    healGap: 0,            // subtract mode: px gap across a removed printed line to re-join; 0 = off (no gain in tests)
    despeckle: true,       // remove 1-px slivers left over from printed lines
    inkMode: 'ratio',      // 'ratio': darker than the printed chart (sees ink over gray dividers); 'subtract': erase printed lines
    inkRatio: 0.62,        // ratio mode: photo/printed brightness below this is ink
    bandPad: 1.0,          // mm a row is searched beyond its printed edges, for tall or overhanging digits
    minArea: 0.5,          // mm²: smaller blobs are specks
    minHeight: 0.9,        // mm: shorter blobs are dots and dashes
    strikeCoverage: 0.6,   // share of each diagonal that must be ink to call a tooth struck out
  };

  /* ---------- tiny CNN (mirrors tools/train-digits.py) ---------- */
  const nets = new Map();
  function loadNet(model) {
    if (nets.has(model)) return nets.get(model);
    const w = {};
    for (const [k, v] of Object.entries(model.weights)) {
      const bytes = Uint8Array.from(atob(v.data), c => c.charCodeAt(0));
      w[k] = { shape: v.shape, data: new Float32Array(bytes.buffer) };
    }
    nets.set(model, w);
    return w;
  }

  function conv3x3Relu(input, cin, size, W, B) {
    const cout = W.shape[0], out = new Float32Array(cout * size * size);
    for (let o = 0; o < cout; o++) {
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        let s = B.data[o];
        for (let i = 0; i < cin; i++) {
          const base = (o * cin + i) * 9, plane = i * size * size;
          for (let ky = -1; ky <= 1; ky++) {
            const yy = y + ky;
            if (yy < 0 || yy >= size) continue;
            for (let kx = -1; kx <= 1; kx++) {
              const xx = x + kx;
              if (xx < 0 || xx >= size) continue;
              s += W.data[base + (ky + 1) * 3 + kx + 1] * input[plane + yy * size + xx];
            }
          }
        }
        out[(o * size + y) * size + x] = s > 0 ? s : 0;
      }
    }
    return out;
  }

  function maxPool2(input, ch, size) {
    const h = size / 2, out = new Float32Array(ch * h * h);
    for (let c = 0; c < ch; c++) for (let y = 0; y < h; y++) for (let x = 0; x < h; x++) {
      const p = c * size * size + 2 * y * size + 2 * x;
      out[(c * h + y) * h + x] = Math.max(input[p], input[p + 1], input[p + size], input[p + size + 1]);
    }
    return out;
  }

  function dense(input, W, B, relu) {
    const [nOut, nIn] = W.shape, out = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = B.data[o];
      for (let i = 0; i < nIn; i++) s += W.data[o * nIn + i] * input[i];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  function classify(x28, model) {
    const w = loadNet(model);
    let a = maxPool2(conv3x3Relu(x28, 1, 28, w['c1.weight'], w['c1.bias']), 8, 28);
    a = maxPool2(conv3x3Relu(a, 8, 14, w['c2.weight'], w['c2.bias']), 16, 14);
    a = dense(dense(a, w['f1.weight'], w['f1.bias'], true), w['f2.weight'], w['f2.bias'], false);
    const m = Math.max(...a), e = a.map(v => Math.exp(v - m)), z = e.reduce((p, q) => p + q, 0);
    return e.map(v => v / z);
  }

  // Read the character a few ways (box size, 1 px shifts) and average: steadier, better-calibrated.
  function classifyTTA(cv, mask, model = window.DIGIT_MODEL) {
    const n = model.weights['f2.bias'].shape[0], probs = new Float32Array(n);
    const variants = [[20, 0, 0], [18, 0, 0], [22, 0, 0], [20, 1, 0], [20, -1, 0], [20, 0, 1], [20, 0, -1]];
    for (const [box, dx, dy] of variants) {
      const p = classify(to28(cv, mask, box, dx, dy), model);
      for (let i = 0; i < n; i++) probs[i] += p[i] / variants.length;
    }
    let best = 0;
    for (let i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
    return { digit: best, label: model.classes ? model.classes[best] : String(best), p: probs[best] };
  }

  /* ---------- image helpers ---------- */
  let template = null;
  function loadTemplate() {
    if (template) return Promise.resolve(template);
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => { template = img; res(img); };
      img.onerror = () => rej(new Error('template.js did not load'));
      img.src = window.TEMPLATE.src;
    });
  }

  // Ink where the photo is clearly darker than the printed chart at that spot. Brightness is first
  // evened out (paper = 1.0), and the template is darkened to its darkest neighbour so a 1-2 px
  // misalignment of a printed line isn't mistaken for ink. Pen over a light-gray cell divider still
  // counts (ink multiplies down); over solid black borders it can't be seen and is ignored.
  function inkMaskRatio(cv, flat, tmplImg, px) {
    const gray = new cv.Mat(), g = new cv.Mat(), bg = new cv.Mat(), norm = new cv.Mat();
    cv.cvtColor(flat, gray, cv.COLOR_RGBA2GRAY);
    gray.convertTo(g, cv.CV_32F);
    const kb = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(Math.round(2 * px), Math.round(2 * px)));
    cv.dilate(g, bg, kb);                                         // paper brightness without strokes
    cv.GaussianBlur(bg, bg, new cv.Size(0, 0), 4 * px);
    cv.divide(g, bg, norm);

    const c = document.createElement('canvas');
    c.width = flat.cols; c.height = flat.rows;
    c.getContext('2d').drawImage(tmplImg, 0, 0, c.width, c.height);
    const t = cv.imread(c), tg = new cv.Mat(), tmin = new cv.Mat(), ratio = new cv.Mat();
    cv.cvtColor(t, tg, cv.COLOR_RGBA2GRAY);
    const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(MM.templateGrow, MM.templateGrow));
    cv.erode(tg, tmin, k);
    const tminF = new cv.Mat();
    tmin.convertTo(tminF, cv.CV_32F, 1 / 255);
    cv.max(tminF, new cv.Mat(tminF.rows, tminF.cols, cv.CV_32F, new cv.Scalar(0.05)), tminF);
    cv.divide(norm, tminF, ratio);

    const inkF = new cv.Mat(), ink = new cv.Mat(), visible = new cv.Mat(), printed = new cv.Mat();
    cv.threshold(ratio, inkF, MM.inkRatio, 255, cv.THRESH_BINARY_INV);
    inkF.convertTo(ink, cv.CV_8U);
    cv.threshold(tmin, visible, 90, 255, cv.THRESH_BINARY);       // not on/near a black line
    cv.bitwise_and(ink, visible, ink);
    cv.threshold(tmin, printed, 215, 255, cv.THRESH_BINARY_INV);  // any printed mark nearby
    const sliver = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
    if (MM.despeckle) cv.morphologyEx(ink, ink, cv.MORPH_OPEN, sliver);
    [gray, g, bg, norm, kb, t, tg, tmin, ratio, k, tminF, inkF, visible, sliver].forEach(m => m.delete());
    return { ink, printed };
  }

  // Binary ink mask (255 = handwriting) at the flattened image's size.
  function inkMask(cv, flat, tmplImg, px) {
    if (MM.inkMode === 'ratio') return inkMaskRatio(cv, flat, tmplImg, px);
    const gray = new cv.Mat(), dark = new cv.Mat();
    cv.cvtColor(flat, gray, cv.COLOR_RGBA2GRAY);
    cv.adaptiveThreshold(gray, dark, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 2 * Math.round(3 * px) + 1, MM.inkThreshold);

    const c = document.createElement('canvas');
    c.width = flat.cols; c.height = flat.rows;
    c.getContext('2d').drawImage(tmplImg, 0, 0, c.width, c.height);
    const t = cv.imread(c), tg = new cv.Mat(), printed = new cv.Mat();
    cv.cvtColor(t, tg, cv.COLOR_RGBA2GRAY);
    cv.threshold(tg, printed, 215, 255, cv.THRESH_BINARY_INV);
    const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(MM.templateGrow, MM.templateGrow));
    cv.dilate(printed, printed, k);

    const notPrinted = new cv.Mat(), ink = new cv.Mat();
    cv.bitwise_not(printed, notPrinted);
    cv.bitwise_and(dark, notPrinted, ink);
    // Drop 1-px slivers of printed lines that survived the subtraction (pen strokes are thicker).
    const sliver = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
    if (MM.despeckle) cv.morphologyEx(ink, ink, cv.MORPH_OPEN, sliver);
    // Re-join strokes cut where they crossed a printed line (a "0" spilling over a cell divider),
    // filling only inside the removed printed area and only where ink continues on both sides.
    const heal = new cv.Mat();
    if (MM.healGap) {
      const kh = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(MM.healGap, 1));
      const kv = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, MM.healGap));
      const a = new cv.Mat(), b = new cv.Mat();
      cv.morphologyEx(ink, a, cv.MORPH_CLOSE, kh);
      cv.morphologyEx(ink, b, cv.MORPH_CLOSE, kv);
      cv.bitwise_or(a, b, heal);
      cv.bitwise_and(heal, printed, heal);
      cv.bitwise_or(ink, heal, ink);
      [kh, kv, a, b].forEach(m => m.delete());
    }
    [gray, dark, t, tg, k, sliver, heal, notPrinted].forEach(m => m.delete());
    return { ink, printed };
  }

  // MNIST-style 28x28: fit the character into 20x20, centre by centre of mass.
  function to28(cv, mask, box = 20, dx = 0, dy = 0) {
    const s = box / Math.max(mask.cols, mask.rows);
    const w = Math.max(1, Math.round(mask.cols * s)), h = Math.max(1, Math.round(mask.rows * s));
    const f = new cv.Mat(), r = new cv.Mat();
    mask.convertTo(f, cv.CV_32F, 1 / 255);
    cv.resize(f, r, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
    let sx = 0, sy = 0, sm = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = r.data32F[y * w + x]; sx += v * x; sy += v * y; sm += v;
    }
    const ox = Math.round(14 - (sm ? sx / sm : w / 2)) + dx, oy = Math.round(14 - (sm ? sy / sm : h / 2)) + dy;
    const out = new Float32Array(28 * 28);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const X = x + ox, Y = y + oy;
      if (X >= 0 && X < 28 && Y >= 0 && Y < 28) out[Y * 28 + X] = Math.min(1, r.data32F[y * w + x] * 1.4);
    }
    f.delete(); r.delete();
    return out;
  }

  /* ---------- reading ---------- */
  function interpret(kind, chars) {
    if (!chars.length) return null;
    const p = chars.reduce((a, c) => a * c.p, 1), s = chars.map(c => c.digit).join('');
    if (kind === 'pd') {
      const v = +s;
      return chars.length <= 2 && v <= 15 ? { value: String(v), confidence: p } : { value: s.slice(0, 2), confidence: 0 };
    }
    if (kind === 'mob') return chars.length === 1 && chars[0].digit <= 3 ? { value: s, confidence: p } : { value: '', confidence: 0 };
    if (kind === 'furc') {
      if (chars.length >= 2 && chars.length <= 3 && chars.every(c => c.digit === 1)) return { value: 'I'.repeat(chars.length), confidence: p };
      if (chars.length === 1 && chars[0].digit >= 1 && chars[0].digit <= 3) return { value: s, confidence: p };
      return { value: '', confidence: 0 };
    }
    if (kind === 'prog') return chars.length === 1 && chars[0].label ? { value: chars[0].label, confidence: p } : { value: '', confidence: 0 };
    return { value: '', confidence: 0 };
  }

  function struckTeeth(cv, ink, printed, columns, px) {
    const out = [], r = Math.round(0.5 * px), W = ink.cols, H = ink.rows;
    const at = (m, x, y) => m.data[Math.min(H - 1, Math.max(0, y)) * W + Math.min(W - 1, Math.max(0, x))];
    for (const c of columns) {
      const x0 = c.x * px, y0 = c.y * px, w = c.w * px, h = c.h * px;
      const cover = diag => {
        let hit = 0, n = 0;
        for (let i = 2; i <= 98; i++) {
          const t = i / 100, x = Math.round(x0 + (diag ? t : 1 - t) * w), y = Math.round(y0 + t * h);
          if (at(printed, x, y)) continue;          // printed line here: can't tell, skip the sample
          n++;
          search: for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
            if (at(ink, x + dx, y + dy)) { hit++; break search; }
          }
        }
        return n ? hit / n : 0;
      };
      if (cover(true) >= MM.strikeCoverage && cover(false) >= MM.strikeCoverage) out.push(c.tooth);
    }
    return out;
  }

  async function run({ cv, flat, px, layout, warpArea }) {
    const tmpl = await loadTemplate();
    const { ink, printed } = inkMask(cv, flat, tmpl, px);
    const missing = struckTeeth(cv, ink, printed, layout.columns, px);
    const skip = new Set(missing);
    const values = {};
    const minArea = MM.minArea * px * px, minH = MM.minHeight * px;

    for (const row of layout.rows) {
      const fields = layout.fields.filter(f => f.jaw === row.jaw && f.r === row.r && !skip.has(f.tooth));
      if (!fields.length) continue;
      const pad = MM.bandPad * px;
      const x0 = Math.round(Math.min(...fields.map(f => f.x)) * px), x1 = Math.round(Math.max(...fields.map(f => f.x + f.w)) * px);
      const y0 = Math.max(0, Math.round(row.y * px - pad)), y1 = Math.min(ink.rows, Math.round((row.y + row.h) * px + pad));
      const roi = ink.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0)), band = roi.clone();
      roi.delete();
      const labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
      const n = cv.connectedComponentsWithStats(band, labels, stats, cents, 8, cv.CV_32S);

      // Blobs, assigned to the cell containing their centre.
      const byField = new Map();
      for (let k = 1; k < n; k++) {
        const s = i => stats.data32S[k * 5 + i];
        const [bx, by, bw, bh, area] = [s(0), s(1), s(2), s(3), s(4)];
        if (area < minArea || bh < minH || bh > (y1 - y0) * 1.05) continue;
        const cx = (x0 + cents.doubleAt(k, 0)) / px, cy = (y0 + cents.doubleAt(k, 1)) / px;
        if (cy < row.y || cy > row.y + row.h) continue;   // belongs to the row above or below
        const f = fields.find(f => cx >= f.x && cx < f.x + f.w);
        if (!f) continue;
        if (!byField.has(f.i)) byField.set(f.i, []);
        byField.get(f.i).push({ k, x: bx, y: by, w: bw, h: bh });
      }

      for (const [i, blobs] of byField) {
        // Blobs overlapping horizontally belong to one character (a "4" or "7" drawn in two strokes).
        blobs.sort((a, b) => a.x - b.x);
        const groups = [];
        for (const b of blobs) {
          const g = groups[groups.length - 1];
          const overlap = g ? Math.min(g.x + g.w, b.x + b.w) - Math.max(g.x, b.x) : -1;
          if (g && overlap > 0.4 * Math.min(g.w, b.w)) {
            const nx = Math.min(g.x, b.x), ny = Math.min(g.y, b.y);
            g.w = Math.max(g.x + g.w, b.x + b.w) - nx; g.h = Math.max(g.y + g.h, b.y + b.h) - ny;
            g.x = nx; g.y = ny; g.ks.push(b.k);
          } else groups.push({ ...b, ks: [b.k] });
        }
        const lab = labels.data32S, lw = labels.cols;
        const chars = groups.map(g => {
          const m = cv.Mat.zeros(g.h, g.w, cv.CV_8U);
          for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) {
            if (g.ks.includes(lab[(g.y + y) * lw + g.x + x])) m.data[y * g.w + x] = 255;
          }
          const model = layout.fields[i].kind === 'prog' ? window.LETTER_MODEL : window.DIGIT_MODEL;
          const x28 = to28(cv, m), c = model ? classifyTTA(cv, m, model) : { digit: -1, label: '', p: 0 };
          if (MM.debug) (window.READER_DEBUG ||= {})[i] = [...((window.READER_DEBUG || {})[i] || []), { x28: [...x28], ...c, w: g.w, h: g.h }];
          m.delete();
          return c;
        });
        const kind = layout.fields[i].kind, r = interpret(kind, chars);
        if (r) values[i] = r;
      }
      [band, labels, stats, cents].forEach(m => m.delete());
    }
    const extras = readExtras(cv, flat, ink, layout, px, warpArea);
    ink.delete(); printed.delete();
    return { values, missing, ...extras };
  }

  /* ---------- header, checkboxes, handwriting images ---------- */
  const clampRect = (cv, m, x0, y0, x1, y1) => {
    x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
    x1 = Math.min(m.cols, Math.round(x1)); y1 = Math.min(m.rows, Math.round(y1));
    return new cv.Rect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
  };

  // Characters written along one header line, left to right. Slashes are recognised by shape.
  function readLine(cv, ink, r, px, allowSlash = false) {
    const xmin = r.xmin ?? r.x - 1, xmax = r.xmax ?? r.x + r.w + 1;   // free space around the underline
    const rect = clampRect(cv, ink, (xmin - 3) * px, (r.y - 2.2) * px, (xmax + 3) * px, (r.y + r.h + 0.6) * px);   // ±3 mm to notice spill-over
    const roi = ink.roi(rect), band = roi.clone();
    roi.delete();
    const labels = new cv.Mat(), stats = new cv.Mat(), cents = new cv.Mat();
    const n = cv.connectedComponentsWithStats(band, labels, stats, cents, 8, cv.CV_32S);
    const blobs = [];
    let spill = false;   // writing that runs on past this field's space: what's read may be cut short
    for (let k = 1; k < n; k++) {
      const s = i => stats.data32S[k * 5 + i];
      const [bx, by, bw, bh, area] = [s(0), s(1), s(2), s(3), s(4)];
      if (area < MM.minArea * px * px || bh < MM.minHeight * px) continue;
      const cx = rect.x / px + cents.doubleAt(k, 0) / px;
      if (cx < xmin || cx > xmax) { spill = true; continue; }
      blobs.push({ k, x: bx, y: by, w: bw, h: bh });
    }
    const lab = labels.data32S, lw = labels.cols;
    const medianH = blobs.map(b => b.h).sort((a, b) => a - b)[blobs.length >> 1] || 0;
    // Possible "/" (dates only), judged stroke by stroke before digits are grouped, since a slash leans
    // into its neighbours: one stroke leaning right, as tall as the digits, thin on every row (a 7
    // isn't). A slanted 1 looks the same, so the date parser decides which candidates are slashes.
    for (const b of blobs) {
      if (!allowSlash) break;
      let tx = 0, tn = 0, bx = 0, bn = 0, widest = 0;
      for (let y = 0; y < b.h; y++) {
        let first = -1, last = -1;
        for (let x = 0; x < b.w; x++) {
          if (lab[(b.y + y) * lw + b.x + x] !== b.k) continue;
          if (first < 0) first = x;
          last = x;
          if (y < b.h / 2) { tx += x; tn++; } else { bx += x; bn++; }
        }
        if (first >= 0) widest = Math.max(widest, last - first + 1);
      }
      b.lean = tn && bn ? (tx / tn - bx / bn) / b.h : 0;
      b.slash = b.w < 0.75 * b.h && b.h > 0.6 * medianH && b.lean > 0.06 && widest < 0.36 * b.h;
    }
    blobs.sort((a, b) => a.x - b.x);
    const groups = [];
    for (const b of blobs) {
      if (b.slash) { groups.push({ ...b, ks: [b.k] }); continue; }
      const g = groups.filter(g => !g.slash).pop();
      const overlap = g ? Math.min(g.x + g.w, b.x + b.w) - Math.max(g.x, b.x) : -1;
      if (g && overlap > 0.4 * Math.min(g.w, b.w)) {
        const nx = Math.min(g.x, b.x), ny = Math.min(g.y, b.y);
        g.w = Math.max(g.x + g.w, b.x + b.w) - nx; g.h = Math.max(g.y + g.h, b.y + b.h) - ny;
        g.x = nx; g.y = ny; g.ks.push(b.k);
      } else groups.push({ ...b, ks: [b.k] });
    }
    groups.sort((a, b) => a.x - b.x);
    // Writing that runs into the edge of its space may be cut off: flag it.
    const touchesEdge = spill;
    const chars = groups.map(g => {
      const m = cv.Mat.zeros(g.h, g.w, cv.CV_8U);
      for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) {
        if (g.ks.includes(lab[(g.y + y) * lw + g.x + x])) m.data[y * g.w + x] = 255;
      }
      const c = classifyTTA(cv, m, window.DIGIT_MODEL);
      m.delete();
      if (g.slash) Object.assign(c, { canSlash: true, lean: g.lean });
      return touchesEdge ? { ...c, p: Math.min(c.p, 0.5) } : c;
    });
    if (MM.debug) (window.READER_DEBUG_HDR ||= []).push({ band: [rect.x, rect.y, rect.width, rect.height], blobs: blobs.map(b => [b.x, b.y, b.w, b.h]),
      groups: groups.map((g, k) => ({ x: g.x, w: g.w, h: g.h, n: g.ks.length, got: (chars[k].canSlash ? '/?' : '') + chars[k].label, p: +chars[k].p.toFixed(2) })) });
    [band, labels, stats, cents].forEach(m => m.delete());
    return chars;
  }

  const validDate = ([d, mo, y]) => d.length >= 1 && d.length <= 2 && mo.length >= 1 && mo.length <= 2 &&
    +d >= 1 && +d <= 31 && +mo >= 1 && +mo <= 12 && (y.length === 2 || y.length === 4);

  // Date: try every pair of slash candidates (a slanted 1 and a / look alike) and keep the split that
  // makes a real date, preferring the most slanted strokes; with no separators, 8 or 6 digits split d/m/y.
  function readDate(chars) {
    const conf = Math.min(...chars.map(c => c.p));
    const cand = chars.map((c, k) => c.canSlash || c.digit === 1 ? k : -1).filter(k => k >= 0);   // a slash is often read as 1
    let best = null;
    for (let a = 0; a < cand.length; a++) for (let b = a + 1; b < cand.length; b++) {
      const i = cand[a], j = cand[b], dig = (from, to) => chars.slice(from, to).map(c => c.digit).join('');
      const parts = [dig(0, i), dig(i + 1, j), dig(j + 1)];
      const score = (chars[i].lean || 0) + (chars[j].lean || 0);
      if (validDate(parts) && (!best || score > best.score)) best = { parts, score };
    }
    if (!best) {
      const s = chars.map(c => c.digit).join('');
      if (s.length === 8 || s.length === 6) {
        const parts = [s.slice(0, 2), s.slice(2, 4), s.slice(4)];
        if (validDate(parts)) best = { parts, score: 0 };
      }
    }
    if (!best) return { value: chars.map(c => c.digit).join(''), confidence: 0 };
    const [d, mo, y] = best.parts;
    // An exam date is from this year or last (Gregorian or Thai Buddhist era); anything else is likely misread or cut off.
    const now = new Date().getFullYear(), yr = y.length === 2 ? +y + 100 * Math.floor(now / 100) : +y;
    const recent = [now, now - 1, now + 543, now + 542].some(v => v === yr || (y.length === 2 && v % 100 === +y));
    const sure = best.score > 0.25 && recent;
    return { value: `${d.padStart(2, '0')}/${mo.padStart(2, '0')}/${y}`, confidence: sure ? conf : Math.min(conf, recent ? 0.8 : 0.5) };
  }

  function interpretHeader(name, chars) {
    if (!chars.length) return null;
    const conf = Math.min(...chars.map(c => c.p)), s = chars.map(c => c.digit).join('');
    if (name === 'HN') return { value: s, confidence: conf };
    if (name === 'Age') return { value: s, confidence: chars.length <= 3 && +s <= 120 ? conf : 0 };
    if (name === 'Date') return readDate(chars);
    return null;
  }

  // Ticked if there is clearly ink in or just around the printed box (ticks overshoot).
  function readCheck(cv, ink, c, px) {
    const frac = (grow) => {
      const r = clampRect(cv, ink, (c.x - grow) * px, (c.y - grow) * px, (c.x + c.w + grow) * px, (c.y + c.h + grow) * px);
      const roi = ink.roi(r), n = cv.countNonZero(roi);
      roi.delete();
      return n / (r.width * r.height);
    };
    const inner = frac(-0.35), outer = frac(1.0);
    const checked = inner > 0.06 || outer > 0.05;
    const sure = inner > 0.12 || outer > 0.1 || (inner < 0.025 && outer < 0.02);
    return { checked, confidence: sure ? 0.99 : 0.6 };
  }

  // The ink in an area as a transparent PNG, positioned in layout mm. Cut from the photo at 20 px/mm
  // (small Thai script needs it). Ink strength is how much darker the photo is than the blank chart
  // at that spot, so pen over printed lines survives and stroke edges stay smooth; pen colour is kept.
  const HW_SCALE = 20;
  function inkImage(cv, ink, r, px, warpArea) {
    const rect = clampRect(cv, ink, r.x * px, r.y * px, (r.x + r.w) * px, (r.y + r.h) * px);
    const roi = ink.roi(rect), count = cv.countNonZero(roi);
    roi.delete();
    if (count < 2.5 * px * px) return null;                     // under ~2.5 mm² of ink: nothing written
    const area = { x: rect.x / px, y: rect.y / px, w: rect.width / px, h: rect.height / px };
    const col = warpArea(area, HW_SCALE), W = col.cols, H = col.rows;

    const gray = new cv.Mat(), g = new cv.Mat(), bg = new cv.Mat(), norm = new cv.Mat();
    cv.cvtColor(col, gray, cv.COLOR_RGBA2GRAY);
    gray.convertTo(g, cv.CV_32F);
    const kb = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2 * HW_SCALE, 2 * HW_SCALE));
    cv.dilate(g, bg, kb);
    cv.GaussianBlur(bg, bg, new cv.Size(0, 0), 2 * HW_SCALE);
    cv.divide(g, bg, norm, 255);                                  // paper = 255

    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const tp = window.TEMPLATE.px;
    c.getContext('2d').drawImage(template, area.x * tp, area.y * tp, area.w * tp, area.h * tp, 0, 0, W, H);
    const t = cv.imread(c), tg = new cv.Mat();
    cv.cvtColor(t, tg, cv.COLOR_RGBA2GRAY);
    const kt = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(Math.round(HW_SCALE / 4) * 2 + 1, Math.round(HW_SCALE / 4) * 2 + 1));
    cv.erode(tg, tg, kt);                                         // darkest print nearby: absorbs misalignment

    for (let i = 0; i < W * H; i++) {
      const o = i * 4, d = tg.data[i] - norm.data32F[i];
      const a = Math.max(0, Math.min(255, (d - 40) * 2.4));
      for (let ch = 0; ch < 3; ch++) col.data[o + ch] = Math.round(col.data[o + ch] * 0.55);
      col.data[o + 3] = a;
    }
    const canvas = document.createElement('canvas');
    cv.imshow(canvas, col);
    [col, gray, g, bg, norm, kb, t, tg, kt].forEach(m => m.delete());
    return { src: canvas.toDataURL('image/png'), ...area };
  }

  function readExtras(cv, flat, ink, layout, px, warpArea) {
    const header = {}, checks = {}, images = {};
    for (const name of ['HN', 'Age', 'Date']) {
      const r = layout.header?.[name];
      if (!r) continue;
      const v = interpretHeader(name, readLine(cv, ink, r, px, name === 'Date'));
      if (v) header[name] = v;
    }
    for (const c of layout.checks || []) checks[c.i] = { label: c.label, ...readCheck(cv, ink, c, px) };
    for (const name of ['Patient', 'Examiner']) {
      const r = layout.header?.[name];
      if (!r) continue;
      const xmin = r.xmin ?? r.x - 1, xmax = r.xmax ?? r.x + r.w + 1;
      images[name] = inkImage(cv, ink, { x: xmin, y: r.y - 2.2, w: xmax - xmin, h: r.h + 2.8 }, px, warpArea);
    }
    for (const [key, a] of Object.entries(layout.areas || {})) {
      images[key] = inkImage(cv, ink, { x: a.x - 0.5, y: a.y - 1.5, w: a.w + 1, h: a.h + 2.1 }, px, warpArea);
    }
    for (const k of Object.keys(images)) if (!images[k]) delete images[k];
    return { header, checks, images };
  }

  window.OnDeviceReader = { run, settings: MM };
})();
