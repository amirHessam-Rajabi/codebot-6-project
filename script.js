/* =========================================================
   کدبات ۶ — منطق تعاملی
   شامل: کشیدن روی بوم، پیش‌پردازش شبیه MNIST، forward-pass
   شبکه‌ی عصبی با جاوااسکریپت خالص، و به‌روزرسانی نمودارها.
   ========================================================= */

(function () {
  "use strict";

  /* ---------------- کمکی: تبدیل ارقام به فارسی (باید پیش از هر استفاده تعریف شود) ---------------- */
  const FA_DIGITS = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"];
  function toFarsiDigits(n) {
    return String(n).replace(/[0-9]/g, (d) => FA_DIGITS[+d]);
  }

  /* ---------------- تنظیمات و عناصر ---------------- */
  const drawCanvas = document.getElementById("drawCanvas");
  const drawCtx = drawCanvas.getContext("2d", { willReadFrequently: true });
  const pixelCanvas = document.getElementById("pixelCanvas");
  const pixelCtx = pixelCanvas.getContext("2d");
  const canvasHint = document.getElementById("canvasHint");
  const clearBtn = document.getElementById("clearBtn");
  const barsWrap = document.getElementById("bars");
  const bigDigitEl = document.getElementById("bigDigit");
  const confidenceNote = document.getElementById("confidenceNote");
  const accStat = document.getElementById("accStat");

  pixelCanvas.width = 28;
  pixelCanvas.height = 28;

  // نمایش دقت مدل از فایل وزن‌ها
  // (توجه: MODEL_WEIGHTS با const در assets/model.js تعریف شده، پس روی
  // window قرار نمی‌گیرد؛ باید مستقیماً به‌عنوان متغیر سراسری بررسی شود)
  const modelReady = typeof MODEL_WEIGHTS !== "undefined";
  if (modelReady && accStat) {
    const pct = Math.round(MODEL_WEIGHTS.test_accuracy * 100);
    accStat.textContent = toFarsiDigits(pct) + "٪";
  }

  /* ---------------- رسم روی بوم ---------------- */
  drawCtx.fillStyle = "#000";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = "#fff";
  drawCtx.lineWidth = 18;

  let drawing = false;
  let hasInk = false;
  let predictTimer = null;
  let strokePts = [];

  function canvasPoint(evt) {
    const rect = drawCanvas.getBoundingClientRect();
    const cx = (evt.touches ? evt.touches[0].clientX : evt.clientX) - rect.left;
    const cy = (evt.touches ? evt.touches[0].clientY : evt.clientY) - rect.top;
    return {
      x: (cx / rect.width) * drawCanvas.width,
      y: (cy / rect.height) * drawCanvas.height,
    };
  }

  function midpoint(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  function startDraw(evt) {
    evt.preventDefault();
    drawing = true;
    hasInk = true;
    canvasHint.classList.add("hidden");
    const p = canvasPoint(evt);
    strokePts = [p];
    drawCtx.beginPath();
    drawCtx.arc(p.x, p.y, drawCtx.lineWidth / 2, 0, Math.PI * 2);
    drawCtx.fillStyle = "#fff";
    drawCtx.fill();
    schedulePredict(60);
  }

  // خط‌های نرم و منحنی (نه چندضلعیِ تیز) — چون مدل روی داده‌ای با خطوط
  // طبیعی و نرم آموزش دیده، این کار دقت را روی دست‌خط واقعی بهتر می‌کند.
  function moveDraw(evt) {
    if (!drawing) return;
    evt.preventDefault();
    const p = canvasPoint(evt);
    strokePts.push(p);
    const n = strokePts.length;
    if (n < 3) {
      drawCtx.beginPath();
      drawCtx.moveTo(strokePts[0].x, strokePts[0].y);
      drawCtx.lineTo(p.x, p.y);
      drawCtx.stroke();
    } else {
      const p0 = strokePts[n - 3], p1 = strokePts[n - 2], p2 = strokePts[n - 1];
      const m1 = midpoint(p0, p1), m2 = midpoint(p1, p2);
      drawCtx.beginPath();
      drawCtx.moveTo(m1.x, m1.y);
      drawCtx.quadraticCurveTo(p1.x, p1.y, m2.x, m2.y);
      drawCtx.stroke();
    }
    schedulePredict(90);
  }

  function endDraw() {
    if (!drawing) return;
    drawing = false;
    schedulePredict(0);
  }

  drawCanvas.addEventListener("mousedown", startDraw);
  drawCanvas.addEventListener("mousemove", moveDraw);
  window.addEventListener("mouseup", endDraw);
  drawCanvas.addEventListener("touchstart", startDraw, { passive: false });
  drawCanvas.addEventListener("touchmove", moveDraw, { passive: false });
  drawCanvas.addEventListener("touchend", endDraw);

  clearBtn.addEventListener("click", () => {
    drawCtx.fillStyle = "#000";
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    hasInk = false;
    canvasHint.classList.remove("hidden");
    resetOutputs();
  });

  function schedulePredict(delay) {
    if (predictTimer) clearTimeout(predictTimer);
    predictTimer = setTimeout(runPrediction, delay);
  }

  /* ---------------- پیش‌پردازش شبیه MNIST ---------------- */
  // ۱) پیدا کردن کادر دور رقم رسم‌شده، ۲) برش و تغییر اندازه به کادر ۲۰×۲۰
  // داخل بوم ۲۸×۲۸، ۳) مرکز کردن بر اساس مرکز جرم -- دقیقاً همان خط لوله‌ای
  // که برای ساخت داده‌ی آموزشی این مدل استفاده شد.
  function preprocess() {
    const w = drawCanvas.width, h = drawCanvas.height;
    const img = drawCtx.getImageData(0, 0, w, h).data;

    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        if (img[idx] > 25) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null; // چیزی رسم نشده

    const padX = (maxX - minX) * 0.18 + 6;
    const padY = (maxY - minY) * 0.18 + 6;
    minX = Math.max(0, minX - padX);
    minY = Math.max(0, minY - padY);
    maxX = Math.min(w, maxX + padX);
    maxY = Math.min(h, maxY + padY);
    const bw = maxX - minX, bh = maxY - minY;

    const target = 20; // رقم داخل کادر ۲۰×۲۰ جا می‌شود (روال استاندارد MNIST)
    const scale = target / Math.max(bw, bh);
    const newW = Math.max(1, bw * scale);
    const newH = Math.max(1, bh * scale);

    const tmp = document.createElement("canvas");
    tmp.width = 28; tmp.height = 28;
    const tctx = tmp.getContext("2d");
    tctx.imageSmoothingEnabled = true;
    tctx.imageSmoothingQuality = "high";
    tctx.fillStyle = "#000";
    tctx.fillRect(0, 0, 28, 28);
    const dx = (28 - newW) / 2, dy = (28 - newH) / 2;
    tctx.drawImage(drawCanvas, minX, minY, bw, bh, dx, dy, newW, newH);

    const raw = tctx.getImageData(0, 0, 28, 28).data;
    const gray = new Float64Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) gray[i] = raw[i * 4]; // کانال قرمز کافی است (سیاه‌وسفید)

    return centerByMass(gray);
  }

  // مرکز کردن بر اساس مرکز جرم با نمونه‌برداری دوخطی (مشابه scipy.ndimage.shift)
  function centerByMass(gray) {
    let sum = 0, sx = 0, sy = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = gray[y * 28 + x];
        sum += v; sx += v * x; sy += v * y;
      }
    }
    if (sum <= 0) return gray;
    const cx = sx / sum, cy = sy / sum;
    const shiftX = 14 - cx, shiftY = 14 - cy;

    const out = new Float64Array(28 * 28);
    for (let ty = 0; ty < 28; ty++) {
      for (let tx = 0; tx < 28; tx++) {
        out[ty * 28 + tx] = bilinear(gray, tx - shiftX, ty - shiftY);
      }
    }
    return out;
  }

  function bilinear(gray, x, y) {
    if (x < 0 || y < 0 || x > 27 || y > 27) {
      const x0 = Math.floor(x), y0 = Math.floor(y);
      if (x0 < -1 || y0 < -1 || x0 > 27 || y0 > 27) return 0;
    }
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = x0 + 1, y1 = y0 + 1;
    const fx = x - x0, fy = y - y0;
    const g = (xx, yy) => (xx < 0 || yy < 0 || xx > 27 || yy > 27) ? 0 : gray[yy * 28 + xx];
    const v00 = g(x0, y0), v10 = g(x1, y0), v01 = g(x0, y1), v11 = g(x1, y1);
    const top = v00 * (1 - fx) + v10 * fx;
    const bot = v01 * (1 - fx) + v11 * fx;
    return top * (1 - fy) + bot * fy;
  }

  /* ---------------- forward pass شبکه‌ی عصبی ---------------- */
  function forward(input01) {
    const layers = MODEL_WEIGHTS.layers;
    let a = input01;
    const activations = [a];
    for (let li = 0; li < layers.length; li++) {
      const W = layers[li].W, b = layers[li].b;
      const outSize = b.length;
      const inSize = a.length;
      const z = new Float64Array(outSize);
      for (let j = 0; j < outSize; j++) z[j] = b[j];
      for (let i = 0; i < inSize; i++) {
        const ai = a[i];
        if (ai === 0) continue;
        const row = W[i];
        for (let j = 0; j < outSize; j++) z[j] += ai * row[j];
      }
      const isLast = li === layers.length - 1;
      let out;
      if (!isLast) {
        out = new Float64Array(outSize);
        for (let j = 0; j < outSize; j++) out[j] = z[j] > 0 ? z[j] : 0;
      } else {
        let max = -Infinity;
        for (let j = 0; j < outSize; j++) if (z[j] > max) max = z[j];
        let sum = 0;
        out = new Float64Array(outSize);
        for (let j = 0; j < outSize; j++) { out[j] = Math.exp(z[j] - max); sum += out[j]; }
        for (let j = 0; j < outSize; j++) out[j] /= sum;
      }
      activations.push(out);
      a = out;
    }
    return activations; // [input, hidden1, hidden2, ..., output(softmax)]
  }

  /* ---------------- اجرای کامل یک پیش‌بینی ---------------- */
  function runPrediction() {
    if (!hasInk || !modelReady) return;
    const gray = preprocess();
    if (!gray) { resetOutputs(); return; }

    drawPixelPreview(gray);

    const input = new Float64Array(784);
    for (let i = 0; i < 784; i++) input[i] = gray[i] / 255;

    const activations = forward(input);
    const probs = activations[activations.length - 1];

    updateBars(probs);
    updateNetwork(activations);
  }

  function resetOutputs() {
    bigDigitEl.textContent = "؟";
    confidenceNote.textContent = "—";
    pixelCtx.fillStyle = "#000";
    pixelCtx.fillRect(0, 0, 28, 28);
    buildBars(new Array(10).fill(0), -1);
    updateNetwork(null);
  }

  function drawPixelPreview(gray) {
    const imgData = pixelCtx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.max(0, Math.min(255, gray[i]));
      imgData.data[i * 4 + 0] = v;
      imgData.data[i * 4 + 1] = v;
      imgData.data[i * 4 + 2] = 255; // سایه‌ی آبی-بنفش برای حس "دید ماشین"
      imgData.data[i * 4 + 3] = 255;
    }
    pixelCtx.putImageData(imgData, 0, 0);
  }

  /* ---------------- نوار احتمال هر رقم ---------------- */
  function buildBars(probs, topIdx) {
    barsWrap.innerHTML = "";
    for (let d = 0; d < 10; d++) {
      const row = document.createElement("div");
      row.className = "bar-row" + (d === topIdx ? " top" : "");
      row.innerHTML =
        '<span class="bar-digit">' + toFarsiDigits(d) + '</span>' +
        '<span class="bar-track"><span class="bar-fill" style="width:' + (probs[d] * 100).toFixed(1) + '%"></span></span>' +
        '<span class="bar-pct">' + toFarsiDigits(Math.round(probs[d] * 100)) + '٪</span>';
      barsWrap.appendChild(row);
    }
  }

  function updateBars(probs) {
    let topIdx = 0, topVal = -1;
    for (let d = 0; d < 10; d++) if (probs[d] > topVal) { topVal = probs[d]; topIdx = d; }
    buildBars(probs, topIdx);

    bigDigitEl.textContent = toFarsiDigits(topIdx);
    bigDigitEl.classList.remove("pop");
    void bigDigitEl.offsetWidth;
    bigDigitEl.classList.add("pop");

    confidenceNote.textContent = "اطمینان " + toFarsiDigits(Math.round(topVal * 100)) + "٪";
  }

  /* ---------------- نمودار زنده‌ی شبکه (ساده و واضح) ---------------- */
  // طرح جدید: لایه‌ی ورودی = خودِ تصویر ۲۸×۲۸ (کامل، بدون نمونه‌برداری)
  // دو لایه‌ی پنهان = خلاصه‌شده به ۶ گره‌ی بزرگ و خوانا
  // لایه‌ی خروجی = هر ۱۰ رقم به‌طور کامل، با هایلایت مسیر تصمیم‌گیری
  const DIAG = { top: 30, bottom: 290 };
  const NET_LAYOUT = {
    tile: { x0: 800, x1: 950, y0: 85, y1: 235 },
    h1: { count: 6, x: 630 },
    h2: { count: 6, x: 380 },
    out: { count: 10, x: 110 },
  };

  let netNodes = { h1: [], h2: [], out: [] };
  let netLines = { inH1: [], h1H2: [], h2Out: [] };
  let inputImageEl, inputHint;
  let outerRing = null;

  function layerYs(n) {
    const arr = [];
    for (let i = 0; i < n; i++) {
      arr.push(n === 1 ? (DIAG.top + DIAG.bottom) / 2 : DIAG.top + ((DIAG.bottom - DIAG.top) * i) / (n - 1));
    }
    return arr;
  }

  function mkLine(ns, x1, y1, x2, y2) {
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", x1); line.setAttribute("y1", y1);
    line.setAttribute("x2", x2); line.setAttribute("y2", y2);
    line.setAttribute("stroke", "url(#lineGrad)");
    line.setAttribute("stroke-width", "1");
    line.setAttribute("opacity", "0.05");
    line.style.transition = "opacity .35s ease, stroke-width .35s ease";
    return line;
  }

  function buildNetwork() {
    const svg = document.getElementById("netSvg");
    const ns = "http://www.w3.org/2000/svg";
    svg.innerHTML = "";

    const defs = document.createElementNS(ns, "defs");
    defs.innerHTML =
      '<linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">' +
      '<stop offset="0%" stop-color="#29d9ee"/><stop offset="100%" stop-color="#8b6bff"/></linearGradient>' +
      '<filter id="nodeGlow" x="-150%" y="-150%" width="400%" height="400%">' +
      '<feGaussianBlur stdDeviation="5" result="b"/>' +
      '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
    svg.appendChild(defs);

    const h1Y = layerYs(NET_LAYOUT.h1.count);
    const h2Y = layerYs(NET_LAYOUT.h2.count);
    const outY = layerYs(NET_LAYOUT.out.count);
    const tile = NET_LAYOUT.tile;
    const inAnchorY = [tile.y0 + 18, (tile.y0 + tile.y1) / 2, tile.y1 - 18];

    const lineGroup = document.createElementNS(ns, "g");
    inAnchorY.forEach((ay) => {
      h1Y.forEach((hy, j) => {
        const line = mkLine(ns, tile.x0, ay, NET_LAYOUT.h1.x, hy);
        lineGroup.appendChild(line);
        netLines.inH1.push({ el: line, toIdx: j });
      });
    });
    h1Y.forEach((y1, i) => {
      h2Y.forEach((y2, j) => {
        const line = mkLine(ns, NET_LAYOUT.h1.x, y1, NET_LAYOUT.h2.x, y2);
        lineGroup.appendChild(line);
        netLines.h1H2.push({ el: line, fromIdx: i, toIdx: j });
      });
    });
    h2Y.forEach((y1, i) => {
      outY.forEach((y2, j) => {
        const line = mkLine(ns, NET_LAYOUT.h2.x, y1, NET_LAYOUT.out.x, y2);
        lineGroup.appendChild(line);
        netLines.h2Out.push({ el: line, fromIdx: i, toIdx: j });
      });
    });
    svg.appendChild(lineGroup);

    // --- کارت تصویر ورودی (لایه‌ی ورودی، کامل) ---
    const tileGroup = document.createElementNS(ns, "g");
    const frame = document.createElementNS(ns, "rect");
    frame.setAttribute("x", tile.x0); frame.setAttribute("y", tile.y0);
    frame.setAttribute("width", tile.x1 - tile.x0); frame.setAttribute("height", tile.y1 - tile.y0);
    frame.setAttribute("rx", 12);
    frame.setAttribute("fill", "#000");
    frame.setAttribute("stroke", "#29d9ee");
    frame.setAttribute("stroke-opacity", "0.5");
    frame.setAttribute("stroke-width", "1.5");
    tileGroup.appendChild(frame);

    inputImageEl = document.createElementNS(ns, "image");
    inputImageEl.setAttribute("x", tile.x0 + 4); inputImageEl.setAttribute("y", tile.y0 + 4);
    inputImageEl.setAttribute("width", tile.x1 - tile.x0 - 8); inputImageEl.setAttribute("height", tile.y1 - tile.y0 - 8);
    inputImageEl.setAttribute("preserveAspectRatio", "none");
    inputImageEl.style.imageRendering = "pixelated";
    tileGroup.appendChild(inputImageEl);

    inputHint = document.createElementNS(ns, "text");
    inputHint.setAttribute("x", (tile.x0 + tile.x1) / 2);
    inputHint.setAttribute("y", (tile.y0 + tile.y1) / 2 + 5);
    inputHint.setAttribute("text-anchor", "middle");
    inputHint.setAttribute("fill", "#6b6790");
    inputHint.setAttribute("font-size", "13");
    inputHint.setAttribute("font-family", "Vazirmatn, sans-serif");
    inputHint.textContent = "چیزی کشیده نشده";
    tileGroup.appendChild(inputHint);
    svg.appendChild(tileGroup);

    // --- گره‌های لایه‌های پنهان ---
    const nodeGroup = document.createElementNS(ns, "g");
    h1Y.forEach((y) => {
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", NET_LAYOUT.h1.x); c.setAttribute("cy", y); c.setAttribute("r", 8);
      c.setAttribute("fill", "#8b6bff"); c.setAttribute("fill-opacity", "0.18");
      c.style.transition = "r .3s ease, fill-opacity .3s ease";
      nodeGroup.appendChild(c);
      netNodes.h1.push(c);
    });
    h2Y.forEach((y) => {
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", NET_LAYOUT.h2.x); c.setAttribute("cy", y); c.setAttribute("r", 8);
      c.setAttribute("fill", "#8b6bff"); c.setAttribute("fill-opacity", "0.18");
      c.style.transition = "r .3s ease, fill-opacity .3s ease";
      nodeGroup.appendChild(c);
      netNodes.h2.push(c);
    });

    // --- گره‌های خروجی (کامل، با برچسب رقم) ---
    outY.forEach((y, idx) => {
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", NET_LAYOUT.out.x); c.setAttribute("cy", y); c.setAttribute("r", 11);
      c.setAttribute("fill", "#ff5c93"); c.setAttribute("fill-opacity", "0.16");
      c.style.transition = "r .3s ease, fill-opacity .3s ease, filter .3s ease";
      nodeGroup.appendChild(c);
      netNodes.out.push(c);

      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", NET_LAYOUT.out.x);
      label.setAttribute("y", y + 4.5);
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("fill", "#f1eefa");
      label.setAttribute("font-size", "10.5");
      label.setAttribute("font-weight", "700");
      label.setAttribute("font-family", "Vazirmatn, sans-serif");
      label.style.pointerEvents = "none";
      label.textContent = toFarsiDigits(idx);
      nodeGroup.appendChild(label);
    });
    svg.appendChild(nodeGroup);

    const layerLabels = [
      { x: (tile.x0 + tile.x1) / 2, t: "ورودی" },
      { x: NET_LAYOUT.h1.x, t: "پنهان ۱" },
      { x: NET_LAYOUT.h2.x, t: "پنهان ۲" },
      { x: NET_LAYOUT.out.x, t: "خروجی" },
    ];
    layerLabels.forEach((l) => {
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", l.x);
      t.setAttribute("y", DIAG.bottom + 22);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("fill", "#6b6790");
      t.setAttribute("font-size", "12");
      t.setAttribute("font-family", "Vazirmatn, sans-serif");
      t.textContent = l.t;
      svg.appendChild(t);
    });
  }

  function bucketAverage(arr, groups) {
    const n = arr.length;
    const per = n / groups;
    const out = new Array(groups);
    for (let g = 0; g < groups; g++) {
      const start = Math.floor(g * per), end = Math.floor((g + 1) * per);
      let sum = 0, cnt = 0;
      for (let i = start; i < end; i++) { sum += arr[i]; cnt++; }
      out[g] = cnt ? sum / cnt : 0;
    }
    return out;
  }

  function topIndices(values, k) {
    return values
      .map((v, i) => [v, i])
      .sort((a, b) => b[0] - a[0])
      .slice(0, k)
      .map((p) => p[1]);
  }

  function resetLines() {
    Object.values(netLines).flat().forEach(({ el }) => {
      el.setAttribute("opacity", "0.05");
      el.setAttribute("stroke-width", "1");
    });
  }

  function updateNetwork(activations) {
    if (!inputImageEl) return;
    if (!activations) {
      netNodes.h1.forEach((c) => { c.setAttribute("fill-opacity", "0.18"); c.setAttribute("r", 8); });
      netNodes.h2.forEach((c) => { c.setAttribute("fill-opacity", "0.18"); c.setAttribute("r", 8); });
      netNodes.out.forEach((c) => { c.setAttribute("fill-opacity", "0.16"); c.setAttribute("r", 11); c.removeAttribute("filter"); });
      resetLines();
      inputImageEl.setAttribute("href", "");
      inputHint.style.display = "";
      return;
    }
    const [, h1, h2, out] = activations;
    inputHint.style.display = "none";
    inputImageEl.setAttribute("href", pixelCanvas.toDataURL());

    const h1Sample = bucketAverage(h1, NET_LAYOUT.h1.count);
    const h2Sample = bucketAverage(h2, NET_LAYOUT.h2.count);
    const predictedIdx = out.indexOf(Math.max(...out));

    const setLayer = (nodes, values, isOutput) => {
      const maxV = Math.max(...values, 1e-6);
      nodes.forEach((circle, i) => {
        const norm = Math.max(0, Math.min(1, values[i] / maxV));
        circle.setAttribute("fill-opacity", (0.15 + norm * 0.75).toFixed(2));
        circle.setAttribute("r", isOutput ? (9 + values[i] * 16).toFixed(1) : (5 + norm * 7).toFixed(1));
      });
    };
    setLayer(netNodes.h1, h1Sample, false);
    setLayer(netNodes.h2, h2Sample, false);
    setLayer(netNodes.out, out, true);

    // حلقه‌ی درخشان دور رقم برنده — پاسخ روشن به «کدام رقم انتخاب شد»
    netNodes.out.forEach((c, i) => {
      if (i === predictedIdx) {
        c.setAttribute("filter", "url(#nodeGlow)");
        c.setAttribute("fill", "#ff8fb8");
      } else {
        c.removeAttribute("filter");
        c.setAttribute("fill", "#ff5c93");
      }
    });

    // مسیر تصمیم‌گیری: از میان تمام مسیرهای ممکن، آن‌هایی که واقعاً به
    // نورون‌های پرفعالیت و رقم برنده ختم می‌شوند را روشن می‌کنیم.
    const h1Top = new Set(topIndices(h1Sample, 2));
    const h2Top = new Set(topIndices(h2Sample, 2));

    resetLines();
    netLines.inH1.forEach(({ el, toIdx }) => {
      if (h1Top.has(toIdx)) { el.setAttribute("opacity", "0.55"); el.setAttribute("stroke-width", "2"); }
    });
    netLines.h1H2.forEach(({ el, fromIdx, toIdx }) => {
      if (h1Top.has(fromIdx) && h2Top.has(toIdx)) { el.setAttribute("opacity", "0.6"); el.setAttribute("stroke-width", "2"); }
    });
    netLines.h2Out.forEach(({ el, fromIdx, toIdx }) => {
      if (h2Top.has(fromIdx) && toIdx === predictedIdx) { el.setAttribute("opacity", "0.85"); el.setAttribute("stroke-width", "2.5"); }
    });
  }

  /* ---------------- پس‌زمینه‌ی متحرک: شبکه‌ی نقطه‌ها (فقط تزیینی) ---------------- */
  function initBackgroundNet() {
    const canvas = document.getElementById("bgNet");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w, h, dots;

    function resize() {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
      const count = Math.min(70, Math.round((w * h) / 22000));
      dots = new Array(count).fill(0).map(() => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
      }));
    }
    window.addEventListener("resize", resize);
    resize();

    function frame() {
      ctx.clearRect(0, 0, w, h);
      for (const d of dots) {
        if (!reduceMotion) {
          d.x += d.vx; d.y += d.vy;
          if (d.x < 0 || d.x > w) d.vx *= -1;
          if (d.y < 0 || d.y > h) d.vy *= -1;
        }
      }
      for (let i = 0; i < dots.length; i++) {
        for (let j = i + 1; j < dots.length; j++) {
          const dx = dots[i].x - dots[j].x, dy = dots[i].y - dots[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 150) {
            ctx.strokeStyle = `rgba(139,107,255,${(1 - dist / 150) * 0.15})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(dots[i].x, dots[i].y);
            ctx.lineTo(dots[j].x, dots[j].y);
            ctx.stroke();
          }
        }
      }
      for (const d of dots) {
        ctx.fillStyle = "rgba(41,217,238,0.5)";
        ctx.beginPath();
        ctx.arc(d.x, d.y, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
      if (!reduceMotion) requestAnimationFrame(frame);
    }
    frame();
  }

  /* ---------------- شروع ---------------- */
  initBackgroundNet();
  buildNetwork();
  resetOutputs();
})();
