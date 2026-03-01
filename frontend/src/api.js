// ── Datasets ─────────────────────────────────────────────────────────────

export async function fetchDatasets() {
  const res = await fetch("/api/datasets", { cache: "no-store" });
  return res.json();
}

export async function createDataset(name) {
  const res = await fetch("/api/datasets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return res.json();
}

export async function deleteDataset(name) {
  const res = await fetch(`/api/datasets/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  return res.json();
}

export async function fetchDatasetSamples(name, offset = 0, limit = 100) {
  const params = new URLSearchParams({ offset, limit });
  const res = await fetch(
    `/api/datasets/${encodeURIComponent(name)}?${params}`,
    { cache: "no-store" }
  );
  return res.json();
}

export async function uploadToDataset(name, files) {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await fetch(`/api/datasets/${encodeURIComponent(name)}/upload`, {
    method: "POST",
    body: form,
  });
  return res.json();
}

// ── Recordings ───────────────────────────────────────────────────────────

export async function saveRecording(dataset, audioBlob, text, replace = "") {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.wav");
  form.append("text", text);
  form.append("replace", replace);
  const res = await fetch(
    `/api/datasets/${encodeURIComponent(dataset)}/recordings`,
    { method: "POST", body: form }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || "Failed to save recording");
  }
  return res.json();
}

export async function deleteRecording(dataset, sampleName) {
  const res = await fetch(
    `/api/datasets/${encodeURIComponent(dataset)}/recordings/${encodeURIComponent(sampleName)}`,
    { method: "DELETE" }
  );
  return res.json();
}

// ── Models ───────────────────────────────────────────────────────────────

export async function fetchModels() {
  const res = await fetch("/api/models", { cache: "no-store" });
  return res.json();
}

export async function createModel(name) {
  const res = await fetch("/api/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return res.json();
}

export async function deleteModel(name) {
  const res = await fetch(`/api/models/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  return res.json();
}

export async function getModelConfig(name) {
  const res = await fetch(`/api/models/${encodeURIComponent(name)}/config`);
  return res.json();
}

export async function saveModelConfig(name, config) {
  const res = await fetch(`/api/models/${encodeURIComponent(name)}/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  return res.json();
}

// ── Model Training ───────────────────────────────────────────────────────

export async function startModelTraining(name) {
  const res = await fetch(`/api/models/${encodeURIComponent(name)}/train`, {
    method: "POST",
  });
  return res.json();
}

export async function stopModelTraining(name) {
  const res = await fetch(`/api/models/${encodeURIComponent(name)}/stop`, {
    method: "POST",
  });
  return res.json();
}

export async function getModelStatus(name) {
  const res = await fetch(`/api/models/${encodeURIComponent(name)}/status`);
  return res.json();
}

export async function predictModelSamples(name, n = 5) {
  const res = await fetch(`/api/models/${encodeURIComponent(name)}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n }),
  });
  return res.json();
}

export async function transcribeAudio(name, audioBlob) {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.wav");
  const res = await fetch(
    `/api/models/${encodeURIComponent(name)}/transcribe`,
    { method: "POST", body: form }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || "Transcription failed");
  }
  return res.json();
}
