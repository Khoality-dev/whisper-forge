import React, { useState, useEffect, useRef } from "react";
import { fetchCurrentSentence, skipSentence, saveRecording } from "../api";

const styles = {
  progress: {
    marginBottom: "20px",
  },
  progressBar: {
    width: "100%",
    height: "8px",
    background: "#e0e0e0",
    borderRadius: "4px",
    overflow: "hidden",
    marginTop: "6px",
  },
  progressFill: (ratio) => ({
    width: `${ratio * 100}%`,
    height: "100%",
    background: "#2563eb",
    transition: "width 0.3s",
  }),
  progressText: {
    fontSize: "0.9rem",
    color: "#666",
  },
  sentence: {
    fontSize: "1.4rem",
    lineHeight: 1.6,
    padding: "24px",
    background: "#f8f9fa",
    borderRadius: "8px",
    marginBottom: "20px",
    minHeight: "80px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    textAlign: "center",
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
  audio: {
    width: "100%",
    marginBottom: "16px",
  },
  status: {
    fontSize: "0.9rem",
    color: "#666",
    padding: "8px 12px",
    background: "#f3f4f6",
    borderRadius: "4px",
  },
};

export default function CollectData() {
  const [sentence, setSentence] = useState("");
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [finished, setFinished] = useState(false);
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [status, setStatus] = useState("Ready. Click Record to start.");
  const [saving, setSaving] = useState(false);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  useEffect(() => {
    fetchCurrentSentence().then((data) => {
      setSentence(data.sentence);
      setProgress(data.progress);
      setFinished(data.finished);
    });
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        setAudioBlob(blob);
        setAudioURL(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
        setStatus("Recording stopped. Listen back, then Save or Re-record.");
      };

      mediaRecorder.start();
      setRecording(true);
      setAudioURL(null);
      setAudioBlob(null);
      setStatus("Recording...");
    } catch (err) {
      setStatus("Microphone access denied. Please allow microphone access.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  const handleSave = async () => {
    if (!audioBlob || finished) return;
    setSaving(true);
    setStatus("Saving...");
    const data = await saveRecording(audioBlob, sentence);
    setSentence(data.sentence);
    setProgress(data.progress);
    setFinished(data.finished);
    setAudioURL(null);
    setAudioBlob(null);
    setSaving(false);
    setStatus("Saved! Moved to next sentence.");
  };

  const handleSkip = async () => {
    const data = await skipSentence();
    setSentence(data.sentence);
    setProgress(data.progress);
    setFinished(data.finished);
    setAudioURL(null);
    setAudioBlob(null);
    setStatus("Skipped. Moved to next sentence.");
  };

  const handleRerecord = () => {
    setAudioURL(null);
    setAudioBlob(null);
    setStatus("Discarded. Click Record to try again.");
  };

  const ratio = progress.total > 0 ? progress.done / progress.total : 0;
  const hasAudio = audioURL !== null;

  return (
    <div>
      <div style={styles.progress}>
        <div style={styles.progressText}>
          {progress.done} / {progress.total} sentences recorded
        </div>
        <div style={styles.progressBar}>
          <div style={styles.progressFill(ratio)} />
        </div>
      </div>

      <div style={styles.sentence}>{sentence}</div>

      {audioURL && <audio style={styles.audio} src={audioURL} controls />}

      <div style={styles.buttons}>
        {!recording ? (
          <button
            style={{ ...styles.btn("primary"), ...(finished ? styles.btnDisabled : {}) }}
            onClick={startRecording}
            disabled={finished}
          >
            Record
          </button>
        ) : (
          <button style={styles.btn("danger")} onClick={stopRecording}>
            Stop
          </button>
        )}
        <button
          style={{ ...styles.btn("primary"), ...(!hasAudio || saving ? styles.btnDisabled : {}) }}
          onClick={handleSave}
          disabled={!hasAudio || saving}
        >
          Save
        </button>
        <button
          style={{ ...styles.btn("default"), ...(!hasAudio ? styles.btnDisabled : {}) }}
          onClick={handleRerecord}
          disabled={!hasAudio}
        >
          Re-record
        </button>
        <button
          style={{ ...styles.btn("default"), ...(recording ? styles.btnDisabled : {}) }}
          onClick={handleSkip}
          disabled={recording}
        >
          Skip
        </button>
      </div>

      <div style={styles.status}>{status}</div>
    </div>
  );
}
