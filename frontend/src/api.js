// ── Dataset ──────────────────────────────────────────────────────────────

export async function fetchDataset() {
  const res = await fetch("/api/dataset", { cache: "no-store" });
  return res.json();
}

export async function saveRecording(audioBlob, labelId, replaceFilename = null) {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.wav");
  form.append("label_id", labelId);
  form.append("replace", replaceFilename || "");
  const res = await fetch("/api/recordings", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || "Failed to save recording");
  }
  return res.json();
}

export async function deleteRecording(filename) {
  const res = await fetch(`/api/recordings/${filename}`, { method: "DELETE" });
  return res.json();
}

// ── Languages ────────────────────────────────────────────────────────────

export async function createLanguage(name) {
  const res = await fetch("/api/languages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return res.json();
}

export async function deleteLanguage(name) {
  const res = await fetch(`/api/languages/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  return res.json();
}

// ── Labels ───────────────────────────────────────────────────────────────

export async function addLabel(language, text) {
  const res = await fetch("/api/labels/add", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language, text }),
  });
  return res.json();
}

export async function removeLabel(id) {
  const res = await fetch("/api/labels/remove", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  return res.json();
}

export async function getDatasetCount() {
  const res = await fetch("/api/dataset/count");
  return res.json();
}

// ── Train config ────────────────────────────────────────────────────────

export async function getTrainConfig() {
  const res = await fetch("/api/train/config");
  return res.json();
}

export async function saveTrainConfig(config) {
  const res = await fetch("/api/train/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return res.json();
}

// ── Training ────────────────────────────────────────────────────────────

export async function startTraining(config) {
  const res = await fetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return res.json();
}

export async function stopTraining() {
  const res = await fetch("/api/train/stop", { method: "POST" });
  return res.json();
}

export async function getTrainStatus() {
  const res = await fetch("/api/train/status");
  return res.json();
}
