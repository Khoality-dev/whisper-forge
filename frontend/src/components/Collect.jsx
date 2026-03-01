import React, { useState, useEffect, useRef } from "react";
import { fetchDatasets, saveRecording } from "../api";

const styles = {
  row: {
    display: "flex",
    gap: "12px",
    alignItems: "center",
    marginBottom: "20px",
  },
  label: {
    fontSize: "0.9rem",
    color: "#555",
    fontWeight: 500,
  },
  select: {
    padding: "7px 12px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "0.9rem",
    outline: "none",
    minWidth: "200px",
  },
  textInput: {
    padding: "10px 14px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "1rem",
    outline: "none",
    width: "100%",
    marginBottom: "16px",
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
  badge: {
    fontSize: "0.75rem",
    color: "#6b7280",
    background: "#e5e7eb",
    padding: "2px 8px",
    borderRadius: "10px",
  },
};

export default function Collect() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(true);

  // Recording state
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [status, setStatus] = useState("Type a sentence and click Record.");
  const [saving, setSaving] = useState(false);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const loadDatasets = async () => {
    const data = await fetchDatasets();
    const dsList = data.datasets || [];
    setDatasets(dsList);
    if (!selectedDataset && dsList.length > 0) {
      setSelectedDataset(dsList[0].name);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  const selectedCount = datasets.find((d) => d.name === selectedDataset)?.count || 0;

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
    if (!audioBlob || !selectedDataset || !text.trim()) return;
    setSaving(true);
    setStatus("Saving...");
    try {
      await saveRecording(selectedDataset, audioBlob, text.trim());
      clearAudio();
      setText("");
      setStatus("Saved! Ready for next recording.");
      await loadDatasets();
    } catch (err) {
      setStatus(`Error saving: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleDiscard = () => {
    clearAudio();
    setStatus("Discarded. Click Record to try again.");
  };

  if (loading) return <div style={styles.empty}>Loading...</div>;

  if (datasets.length === 0) {
    return (
      <div style={styles.empty}>
        No datasets found. Go to the Dataset tab to create one first.
      </div>
    );
  }

  const hasAudio = audioURL !== null;
  const canSave = hasAudio && !saving && text.trim() && selectedDataset;

  return (
    <div>
      {/* Dataset selector */}
      <div style={styles.row}>
        <span style={styles.label}>Dataset:</span>
        <select
          style={styles.select}
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
        >
          {datasets.map((ds) => (
            <option key={ds.name} value={ds.name}>
              {ds.name} ({ds.count} samples)
            </option>
          ))}
        </select>
        <span style={styles.badge}>{selectedCount} samples</span>
      </div>

      {/* Text input */}
      <input
        style={styles.textInput}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type the sentence to record..."
        disabled={recording}
      />

      {/* Audio playback */}
      {audioURL && <audio style={styles.audio} src={audioURL} controls />}

      {/* Buttons */}
      <div style={styles.buttons}>
        {!recording ? (
          <button
            style={{
              ...styles.btn("primary"),
              ...(!text.trim() ? styles.btnDisabled : {}),
            }}
            onClick={startRecording}
            disabled={!text.trim()}
          >
            Record
          </button>
        ) : (
          <button style={styles.btn("danger")} onClick={stopRecording}>
            Stop
          </button>
        )}
        <button
          style={{
            ...styles.btn("primary"),
            ...(!canSave ? styles.btnDisabled : {}),
          }}
          onClick={handleSave}
          disabled={!canSave}
        >
          {saving ? "Saving..." : "Save"}
        </button>
        <button
          style={{
            ...styles.btn("default"),
            ...(!hasAudio ? styles.btnDisabled : {}),
          }}
          onClick={handleDiscard}
          disabled={!hasAudio}
        >
          Discard
        </button>
      </div>

      <div style={styles.status}>{status}</div>
    </div>
  );
}
