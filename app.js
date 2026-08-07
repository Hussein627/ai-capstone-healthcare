// ── Sample patients injected from the server template ──
const SAMPLES = JSON.parse(document.getElementById("samplesData")?.textContent || "[]");

// ── Tabs ──
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "evaluation") loadEvaluation();
  });
});

// ── Sample patient quick-load ──
document.getElementById("sampleSelect").addEventListener("change", (e) => {
  const idx = e.target.value;
  if (idx === "") return;
  const p = SAMPLES[parseInt(idx, 10)];
  document.getElementById("age").value = p.age;
  document.getElementById("temperature").value = p.temperature;
  document.getElementById("heart_rate").value = p.heart_rate;
  document.getElementById("blood_pressure").value = p.blood_pressure;
  document.querySelectorAll(".symptom-check").forEach(cb => {
    cb.checked = p.symptoms.includes(cb.value);
  });
});

// ── Run diagnosis ──
document.getElementById("diagnoseBtn").addEventListener("click", async () => {
  const symptoms = Array.from(document.querySelectorAll(".symptom-check:checked")).map(cb => cb.value);
  const errorMsg = document.getElementById("errorMsg");
  const loadingMsg = document.getElementById("loadingMsg");
  const btn = document.getElementById("diagnoseBtn");

  errorMsg.classList.add("hidden");
  if (symptoms.length === 0) {
    errorMsg.textContent = "Select at least one symptom before running a diagnosis.";
    errorMsg.classList.remove("hidden");
    return;
  }

  const payload = {
    symptoms,
    age: document.getElementById("age").value,
    temperature: document.getElementById("temperature").value,
    heart_rate: document.getElementById("heart_rate").value,
    blood_pressure: document.getElementById("blood_pressure").value,
    patient_id: "WEB-" + Date.now().toString().slice(-6),
  };

  btn.disabled = true;
  loadingMsg.classList.remove("hidden");
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("resultsContent").classList.add("hidden");

  try {
    const res = await fetch("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResults(data);
  } catch (err) {
    errorMsg.textContent = "Error: " + err.message;
    errorMsg.classList.remove("hidden");
    document.getElementById("emptyState").classList.remove("hidden");
  } finally {
    btn.disabled = false;
    loadingMsg.classList.add("hidden");
  }
});

function severityColor(label) {
  return { LOW: "#1E8449", MILD: "#52BE80", MODERATE: "#F4D03F", HIGH: "#E67E22", CRITICAL: "#C0392B" }[label] || "#999";
}

function renderResults(data) {
  const el = document.getElementById("resultsContent");
  const sev = data.severity;
  const plan = data.treatment_plan;

  let html = "";
  html += `<span class="urgency-badge urgency-${data.urgency}">URGENCY: ${data.urgency}</span>`;
  html += `<div class="diag-headline">${data.diagnosis.replace(/_/g, " ")}</div>`;
  html += `<div class="diag-confidence">Agent confidence: ${(data.confidence * 100).toFixed(1)}% &middot; Patient ${data.patient.id}</div>`;

  html += `<div class="section-title">Severity (Fuzzy Logic)</div>`;
  html += `<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:2px;">
             <span>${sev.severity_label}</span><span>${sev.severity_score.toFixed(1)} / 100</span></div>`;
  html += `<div class="severity-bar-track"><div class="severity-bar-fill" style="width:${sev.severity_score}%;background:${severityColor(sev.severity_label)}"></div></div>`;

  html += `<div class="section-title">Module Opinions</div>`;
  for (const [name, res] of Object.entries(data.module_opinions)) {
    const conf = res.confidence != null ? (res.confidence * 100).toFixed(1) + "%" : "—";
    html += `<div class="opinion-row">
               <span class="opinion-name">${name}</span>
               <span class="opinion-diag">${(res.diagnosis || "unknown").replace(/_/g, " ")}</span>
               <span class="opinion-conf">${conf}</span>
             </div>`;
  }

  html += `<div class="section-title">Treatment Plan (${plan.steps} steps)</div>`;
  plan.plan.forEach(step => {
    html += `<div class="plan-step">
               <div class="plan-step-num">${step.step}</div>
               <div style="flex:1;">
                 <div class="plan-step-name">${step.action}</div>
                 <div class="plan-step-dur">${step.duration}</div>
               </div>
             </div>`;
  });

  html += `<div class="section-title">Recommendations</div><ul style="font-size:13px;color:var(--ink);padding-left:20px;">`;
  (data.recommendations || []).forEach(r => { html += `<li style="margin-bottom:4px;">${r}</li>`; });
  html += `</ul>`;

  el.innerHTML = html;
  el.classList.remove("hidden");
}

// ── Evaluation tab ──
let evalLoaded = false;
async function loadEvaluation() {
  if (evalLoaded) return;
  const loading = document.getElementById("evalLoading");
  const chart = document.getElementById("evalChart");
  loading.classList.remove("hidden");
  try {
    const res = await fetch("/api/evaluation");
    const data = await res.json();
    let html = "";
    const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]);
    for (const [name, acc] of sorted) {
      const pct = (acc * 100).toFixed(1);
      html += `<div class="eval-row">
                 <div class="eval-label">${name}</div>
                 <div class="eval-track"><div class="eval-fill" style="width:${pct}%"></div></div>
                 <div class="eval-pct">${pct}%</div>
               </div>`;
    }
    chart.innerHTML = html;
    evalLoaded = true;
  } catch (err) {
    chart.innerHTML = `<p style="color:#C0392B;">Failed to load evaluation results.</p>`;
  } finally {
    loading.classList.add("hidden");
  }
}
