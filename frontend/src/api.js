export async function fetchCurrentSentence() {
  const res = await fetch("/api/sentences/current");
  return res.json();
}

export async function skipSentence() {
  const res = await fetch("/api/sentences/skip", { method: "POST" });
  return res.json();
}

export async function saveRecording(audioBlob, sentence) {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.wav");
  form.append("sentence", sentence);
  const res = await fetch("/api/recordings", { method: "POST", body: form });
  return res.json();
}

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
