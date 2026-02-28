import React, { useState, useEffect, useRef, useMemo } from "react";
import { fetchDataset, saveRecording } from "../api";

const styles = {
  langSelect: {
    display: "flex",
    gap: "12px",
    flexWrap: "wrap",
    marginBottom: "20px",
    alignItems: "center",
  },
  langLabel: {
    fontSize: "0.9rem",
    color: "#555",
    fontWeight: 500,
  },
  chip: (active) => ({
    padding: "6px 16px",
    borderRadius: "20px",
    fontSize: "0.85rem",
    fontWeight: 500,
    cursor: "pointer",
    border: active ? "1px solid #2563eb" : "1px solid #d1d5db",
    background: active ? "#eff6ff" : "#fff",
    color: active ? "#2563eb" : "#666",
    userSelect: "none",
  }),
  progress: {
    marginBottom: "20px",
  },
  progressText: {
    fontSize: "0.9rem",
    color: "#666",
    marginBottom: "6px",
  },
  progressBar: {
    width: "100%",
    height: "8px",
    background: "#e0e0e0",
    borderRadius: "4px",
    overflow: "hidden",
  },
  progressFill: (ratio) => ({
    width: `${ratio * 100}%`,
    height: "100%",
    background: "#2563eb",
    transition: "width 0.3s",
  }),
  sentence: {
    fontSize: "1.4rem",
    lineHeight: 1.6,
    padding: "32px 24px",
    background: "#f8f9fa",
    borderRadius: "8px",
    marginBottom: "20px",
    minHeight: "80px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    textAlign: "center",
  },
  audio: {
    width: "100%",
    marginBottom: "16px",
  },
  buttons: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
    marginBottom: "16px",
  },
  btn: (variant) => ({
    padding: "10px 20px",
    border: "none",
    borderRadius: "6px",
    fontSize: "0.95rem",
    cursor: "pointer",
    fontWeight: 500,
    ...(variant === "primary"
      ? { background: "#2563eb", color: "#fff" }
      : variant === "danger"
        ? { background: "#dc2626", color: "#fff" }
        : { background: "#e5e7eb", color: "#333" }),
  }),
  btnDisabled: {
    opacity: 0.5,
    cursor: "not-allowed",
  },
  status: {
    fontSize: "0.9rem",
    color: "#666",
    padding: "8px 12px",
    background: "#f3f4f6",
    borderRadius: "4px",
  },
  empty: {
    textAlign: "center",
    padding: "40px",
    color: "#9ca3af",
    fontSize: "0.95rem",
  },
  done: {
    textAlign: "center",
    padding: "48px 24px",
    color: "#16a34a",
    fontSize: "1.1rem",
    fontWeight: 500,
  },
};

function displayName(filename) {
  return filename.replace(/\.txt$/, "");
}

export default function Collect() {
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedLangs, setSelectedLangs] = useState(null);
  const [skipped, setSkipped] = useState(new Set());

  // Recording state
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [status, setStatus] = useState("Select languages and click Record.");
  const [saving, setSaving] = useState(false);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const loadData = async () => {
    const data = await fetchDataset();
    setGroups(data.groups);
    if (selectedLangs === null && data.groups.length > 0) {
      setSelectedLangs(new Set(data.groups.map((g) => g.filename)));
    }
    setLoading(false);
  };

  useEffect(() => {
    loadData();
  }, []);

  const activeLangs = selectedLangs || new Set();

  // Pending labels: from selected languages, no recordings, not skipped
  const pending = useMemo(() => {
    const items = [];
    for (const group of groups) {
      if (!activeLangs.has(group.filename)) continue;
      for (const sent of group.sentences) {
        if (sent.recordings.length === 0 && !skipped.has(sent.text)) {
          items.push(sent.text);
        }
      }
    }
    return items;
  }, [groups, activeLangs, skipped]);

  // Stats for selected languages
  const totalSelected = useMemo(() => {
    let count = 0;
    for (const g of groups) {
      if (activeLangs.has(g.filename)) count += g.sentences.length;
    }
    return count;
  }, [groups, activeLangs]);

  const recordedCount = useMemo(() => {
    let count = 0;
    for (const g of groups) {
      if (!activeLangs.has(g.filename)) continue;
      for (const s of g.sentences) {
        if (s.recordings.length > 0) count++;
      }
    }
    return count;
  }, [groups, activeLangs]);

  const currentLabel = pending.length > 0 ? pending[0] : null;
  const finished = totalSelected > 0 && pending.length === 0;
  const ratio = totalSelected > 0 ? recordedCount / totalSelected : 0;

  // ── Language selection ──

  const toggleLang = (filename) => {
    setSelectedLangs((prev) => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
      return next;
    });
    setSkipped(new Set());
    clearAudio();
  };

  // ── Recording ──

  const clearAudio = () => {
    setAudioURL(null);
    setAudioBlob(null);
  };

  const startRecording = async () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.ondataavailable = null;
      mediaRecorderRef.current.onstop = null;
      if (mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        setAudioBlob(blob);
        setAudioURL(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
        setStatus("Recording stopped. Listen back, then Save or Re-record.");
      };

      recorder.start();
      setRecording(true);
      clearAudio();
      setStatus("Recording...");
    } catch {
      setStatus("Microphone access denied. Please allow microphone access.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  const handleSave = async () => {
    if (!audioBlob || !currentLabel) return;
    setSaving(true);
    setStatus("Saving...");
    await saveRecording(audioBlob, currentLabel);
    clearAudio();
    setSaving(false);
    setStatus("Saved! Moved to next label.");
    loadData();
  };

  const handleSkip = () => {
    if (!currentLabel) return;
    setSkipped((prev) => new Set([...prev, currentLabel]));
    clearAudio();
    setStatus("Skipped.");
  };

  const handleRerecord = () => {
    clearAudio();
    setStatus("Discarded. Click Record to try again.");
  };

  if (loading) return <div style={styles.empty}>Loading...</div>;

  if (groups.length === 0) {
    return (
      <div style={styles.empty}>
        No languages found. Go to the Dataset tab to add languages and labels
        first.
      </div>
    );
  }

  const hasAudio = audioURL !== null;
  const noLangsSelected = activeLangs.size === 0;

  return (
    <div>
      {/* Language selection */}
      <div style={styles.langSelect}>
        <span style={styles.langLabel}>Languages:</span>
        {groups.map((g) => (
          <span
            key={g.filename}
            style={styles.chip(activeLangs.has(g.filename))}
            onClick={() => toggleLang(g.filename)}
          >
            {displayName(g.filename)}
          </span>
        ))}
      </div>

      {noLangsSelected ? (
        <div style={styles.empty}>Select at least one language to start.</div>
      ) : (
        <>
          {/* Progress */}
          <div style={styles.progress}>
            <div style={styles.progressText}>
              {recordedCount} / {totalSelected} labels recorded
              {skipped.size > 0 && ` (${skipped.size} skipped)`}
            </div>
            <div style={styles.progressBar}>
              <div style={styles.progressFill(ratio)} />
            </div>
          </div>

          {/* Current label or done */}
          {finished ? (
            <div style={styles.done}>
              All labels have been recorded!
            </div>
          ) : currentLabel ? (
            <>
              <div style={styles.sentence}>{currentLabel}</div>

              {audioURL && (
                <audio style={styles.audio} src={audioURL} controls />
              )}

              <div style={styles.buttons}>
                {!recording ? (
                  <button
                    style={styles.btn("primary")}
                    onClick={startRecording}
                  >
                    Record
                  </button>
                ) : (
                  <button
                    style={styles.btn("danger")}
                    onClick={stopRecording}
                  >
                    Stop
                  </button>
                )}
                <button
                  style={{
                    ...styles.btn("primary"),
                    ...(!hasAudio || saving ? styles.btnDisabled : {}),
                  }}
                  onClick={handleSave}
                  disabled={!hasAudio || saving}
                >
                  {saving ? "Saving..." : "Save"}
                </button>
                <button
                  style={{
                    ...styles.btn("default"),
                    ...(!hasAudio ? styles.btnDisabled : {}),
                  }}
                  onClick={handleRerecord}
                  disabled={!hasAudio}
                >
                  Re-record
                </button>
                <button
                  style={{
                    ...styles.btn("default"),
                    ...(recording ? styles.btnDisabled : {}),
                  }}
                  onClick={handleSkip}
                  disabled={recording}
                >
                  Skip
                </button>
              </div>

              <div style={styles.status}>{status}</div>
            </>
          ) : null}
        </>
      )}
    </div>
  );
}
