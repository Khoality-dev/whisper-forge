import React, { useState, useEffect, useRef, useMemo } from "react";
import { startTraining, stopTraining, getTrainStatus, getDatasetCount, getTrainConfig, saveTrainConfig, predictSamples } from "../api";

const styles = {
  form: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "16px",
    marginBottom: "20px",
  },
  field: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  label: {
    fontSize: "0.85rem",
    fontWeight: 500,
    color: "#444",
  },
  input: {
    padding: "8px 10px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "0.95rem",
  },
  checkbox: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  buttons: {
    display: "flex",
    gap: "10px",
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
  logArea: {
    width: "100%",
    minHeight: "300px",
    maxHeight: "500px",
    overflow: "auto",
    background: "#1e1e1e",
    color: "#d4d4d4",
    fontFamily: "monospace",
    fontSize: "0.85rem",
    padding: "12px",
    borderRadius: "8px",
    whiteSpace: "pre-wrap",
    wordBreak: "break-all",
  },
};

const defaultConfig = {
  epochs: 5,
  learning_rate: 1e-5,
  train_batch_size: 8,
  eval_batch_size: 8,
  fp16: false,
  logging_steps: 100,
  save_steps: 500,
  eval_steps: 500,
  output_dir: "userdata/outputs",
};

export default function Train() {
  const [config, setConfig] = useState(defaultConfig);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState("");
  const [sampleCount, setSampleCount] = useState(0);
  const [predictions, setPredictions] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const pollRef = useRef(null);
  const logEndRef = useRef(null);

  const set = (key) => (e) => {
    const val = e.target.type === "checkbox" ? e.target.checked : e.target.value;
    setConfig((prev) => {
      const next = { ...prev, [key]: val };
      saveTrainConfig(next);
      return next;
    });
  };

  const pollStatus = async () => {
    const data = await getTrainStatus();
    setRunning(data.running);
    setLog(data.log || "");
    if (!data.running && data.exit_code !== null) {
      setLog((prev) => prev + `\n\nTraining finished (exit code ${data.exit_code}).`);
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const handleStart = async () => {
    const res = await startTraining(config);
    if (res.error) {
      setLog(res.error);
      return;
    }
    setRunning(true);
    setLog(res.message + "\n");
    pollRef.current = setInterval(pollStatus, 2000);
  };

  const handleStop = async () => {
    const res = await stopTraining();
    setRunning(false);
    setLog((prev) => prev + "\n" + res.message);
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  useEffect(() => {
    pollStatus();
    getDatasetCount().then((d) => setSampleCount(d.count || 0));
    getTrainConfig().then((saved) => {
      if (saved && Object.keys(saved).length > 0) {
        setConfig((prev) => ({ ...prev, ...saved }));
      }
    });
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  useEffect(() => {
    // Start polling if we discover training is running on mount
    if (running && !pollRef.current) {
      pollRef.current = setInterval(pollStatus, 2000);
    }
  }, [running]);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollTop = logEndRef.current.scrollHeight;
    }
  }, [log]);

  const fields = [
    { key: "epochs", label: "Epochs", type: "number" },
    { key: "learning_rate", label: "Learning rate", type: "number", step: "any" },
    { key: "train_batch_size", label: "Train batch size", type: "number" },
    { key: "eval_batch_size", label: "Eval batch size", type: "number" },
    { key: "logging_steps", label: "Logging steps", type: "number" },
    { key: "save_steps", label: "Save steps", type: "number" },
    { key: "eval_steps", label: "Eval steps", type: "number" },
  ];

  return (
    <div>
      <p style={{ color: "#666", marginBottom: "16px" }}>
        Configure and launch Whisper fine-tuning. Training runs train.py as a subprocess.
      </p>

      <div style={styles.form}>
        {fields.map(({ key, label, type, step }) => (
          <div key={key} style={styles.field}>
            <label style={styles.label}>{label}</label>
            <input
              style={styles.input}
              type={type}
              step={step}
              value={config[key]}
              onChange={set(key)}
              disabled={running}
            />
          </div>
        ))}
        <div style={{ ...styles.field, ...styles.checkbox, justifyContent: "flex-start" }}>
          <input
            type="checkbox"
            checked={config.fp16}
            onChange={set("fp16")}
            disabled={running}
          />
          <label style={styles.label}>FP16 (mixed precision)</label>
        </div>
      </div>

      {sampleCount > 0 && (() => {
        const batchSize = Number(config.train_batch_size) || 1;
        const epochs = Number(config.epochs) || 1;
        const trainSamples = Math.round(sampleCount * 0.9);
        const stepsPerEpoch = Math.ceil(trainSamples / batchSize);
        const totalSteps = stepsPerEpoch * epochs;
        return (
          <p style={{ color: "#666", fontSize: "0.85rem", marginBottom: "12px" }}>
            {sampleCount} samples ({trainSamples} train) &middot; {stepsPerEpoch} steps/epoch &middot; <strong>{totalSteps} total steps</strong>
          </p>
        );
      })()}

      <div style={styles.buttons}>
        <button
          style={{ ...styles.btn("primary"), ...(running ? styles.btnDisabled : {}) }}
          onClick={handleStart}
          disabled={running}
        >
          Start Training
        </button>
        <button
          style={{ ...styles.btn("danger"), ...(!running ? styles.btnDisabled : {}) }}
          onClick={handleStop}
          disabled={!running}
        >
          Stop Training
        </button>
        <button
          style={{ ...styles.btn(), ...(running ? styles.btnDisabled : {}) }}
          onClick={() => { window.location.href = "/api/train/download"; }}
          disabled={running}
        >
          Download Model
        </button>
        <button
          style={{ ...styles.btn(), ...(running || predicting ? styles.btnDisabled : {}) }}
          onClick={async () => {
            setPredicting(true);
            setPredictions(null);
            try {
              const res = await predictSamples(5);
              if (res.error) { setPredictions([]); setLog((prev) => prev + "\n" + res.error); }
              else { setPredictions(res); }
            } catch { setPredictions([]); }
            setPredicting(false);
          }}
          disabled={running || predicting}
        >
          {predicting ? "Testing..." : "Test Model"}
        </button>
      </div>

      <div style={styles.logArea} ref={logEndRef}>
        {log || "No training output yet."}
      </div>

      {predictions && predictions.length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h3 style={{ fontSize: "1rem", marginBottom: "10px" }}>Sample Predictions</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #e5e7eb", textAlign: "left" }}>
                <th style={{ padding: "8px" }}>Expected</th>
                <th style={{ padding: "8px" }}>Predicted</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((p, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #e5e7eb" }}>
                  <td style={{ padding: "8px", color: "#444" }}>{p.expected}</td>
                  <td style={{ padding: "8px", color: p.expected.toLowerCase() === p.predicted.toLowerCase() ? "#16a34a" : "#dc2626" }}>
                    {p.predicted}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
