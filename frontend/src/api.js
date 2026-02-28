// ── Dataset ──────────────────────────────────────────────────────────────

export async function fetchDataset() {
  const res = await fetch("/api/dataset");
  return res.json();
}

export async function saveRecording(audioBlob, sentence, replaceFilename = null) {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.wav");
  form.append("sentence", sentence);
  if (replaceFilename) {
    form.append("replace", replaceFilename);
  }
  const res = await fetch("/api/recordings", { method: "POST", body: form });
  return res.json();
}

export async function deleteRecording(filename) {
  const res = await fetch(`/api/recordings/${filename}`, { method: "DELETE" });
  return res.json();
}

// ── Labels ──────────────────────────────────────────────────────────────

export async function createLabelFile(filename) {
  const res = await fetch("/api/labels", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  return res.json();
}

export async function deleteLabelFile(filename) {
  const res = await fetch(`/api/labels/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
  return res.json();
}

export async function addSentence(filename, sentence) {
  const res = await fetch(
    `/api/labels/${encodeURIComponent(filename)}/add`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence }),
    }
  );
  return res.json();
}

export async function removeSentence(filename, sentence) {
  const res = await fetch(
    `/api/labels/${encodeURIComponent(filename)}/remove`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence }),
    }
  );
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
