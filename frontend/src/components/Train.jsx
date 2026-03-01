import React, { useState, useEffect, useRef } from "react";
import {
  fetchModels,
  createModel,
  deleteModel,
  getModelConfig,
  saveModelConfig,
  startModelTraining,
  stopModelTraining,
  getModelStatus,
  predictModelSamples,
  fetchDatasets,
} from "../api";

const styles = {
  card: {
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    marginBottom: "12px",
    padding: "14px 18px",
    display: "flex",
    alignItems: "center",
    gap: "12px",
    cursor: "pointer",
    transition: "background 0.15s",
  },
  cardName: {
    fontWeight: 600,
    fontSize: "1rem",
    flex: 1,
  },
  badge: (color) => ({
    fontSize: "0.75rem",
    color: color || "#6b7280",
    background:
      color === "#16a34a" ? "#f0fdf4" : color === "#ea580c" ? "#fff7ed" : "#e5e7eb",
    padding: "2px 8px",
    borderRadius: "10px",
    fontWeight: 500,
  }),
  arrow: {
    color: "#9ca3af",
    fontSize: "1.1rem",
  },
  empty: {
    textAlign: "center",
    padding: "40px",
    color: "#9ca3af",
    fontSize: "0.95rem",
  },
  addForm: {
    display: "flex",
    gap: "8px",
    alignItems: "center",
    marginBottom: "16px",
  },
  input: {
    padding: "7px 12px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "0.9rem",
    outline: "none",
  },
  backBtn: {
    padding: "5px 12px",
    border: "1px solid #d1d5db",
    borderRadius: "5px",
    fontSize: "0.85rem",
    cursor: "pointer",
    fontWeight: 500,
    background: "#fff",
    color: "#333",
    marginBottom: "16px",
  },
  detailHeader: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    marginBottom: "16px",
  },
  detailTitle: {
    fontWeight: 700,
    fontSize: "1.2rem",
    flex: 1,
  },
  section: {
    marginBottom: "20px",
  },
  sectionTitle: {
    fontSize: "0.95rem",
    fontWeight: 600,
    marginBottom: "8px",
    color: "#333",
  },
  checkboxRow: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    marginBottom: "6px",
    fontSize: "0.9rem",
  },
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
  fieldInput: {
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
    flexWrap: "wrap",
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
  btnPrimary: {
    padding: "5px 12px",
    border: "none",
    borderRadius: "5px",
    fontSize: "0.8rem",
    cursor: "pointer",
    fontWeight: 500,
    background: "#2563eb",
    color: "#fff",
  },
  btnDefault: {
    padding: "5px 12px",
    border: "none",
    borderRadius: "5px",
    fontSize: "0.8rem",
    cursor: "pointer",
    fontWeight: 500,
    background: "#e5e7eb",
    color: "#333",
  },
  btnSmallDanger: {
    padding: "3px 8px",
    border: "none",
    borderRadius: "4px",
    fontSize: "0.75rem",
    cursor: "pointer",
    background: "#fef2f2",
    color: "#dc2626",
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

const statusColors = {
  untrained: "#6b7280",
  trained: "#16a34a",
  training: "#ea580c",
};

const defaultConfig = {
  datasets: [],
  epochs: 5,
  learning_rate: 1e-5,
  train_batch_size: 8,
  eval_batch_size: 8,
  fp16: false,
  logging_steps: 100,
  save_steps: 500,
  eval_steps: 500,
};

export default function Train() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  // List view state
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState("");

  // Detail view state
  const [config, setConfig] = useState(defaultConfig);
  const [datasets, setDatasets] = useState([]);
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const pollRef = useRef(null);
  const logEndRef = useRef(null);
  const saveTimeoutRef = useRef(null);

  const loadModels = async () => {
    const data = await fetchModels();
    setModels(data.models || []);
    setLoading(false);
  };

  const loadDetail = async (name) => {
    const [configData, dsData, statusData] = await Promise.all([
      getModelConfig(name),
      fetchDatasets(),
      getModelStatus(name),
    ]);
    setConfig({ ...defaultConfig, ...configData });
    setDatasets(dsData.datasets || []);
    setRunning(statusData.running);
    setLog(statusData.log || "");
    if (statusData.running && !pollRef.current) {
      pollRef.current = setInterval(() => pollStatus(name), 2000);
    }
  };

  useEffect(() => {
    loadModels();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  useEffect(() => {
    if (selected) {
      loadDetail(selected);
    } else {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      setLog("");
      setPredictions(null);
    }
  }, [selected]);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollTop = logEndRef.current.scrollHeight;
    }
  }, [log]);

  const pollStatus = async (name) => {
    const data = await getModelStatus(name);
    setRunning(data.running);
    setLog(data.log || "");
    if (!data.running && data.exit_code !== null) {
      setLog(
        (prev) =>
          prev + `\n\nTraining finished (exit code ${data.exit_code}).`
      );
      clearInterval(pollRef.current);
      pollRef.current = null;
      loadModels();
    }
  };

  // Auto-save config with debounce
  const persistConfig = (newConfig) => {
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = setTimeout(() => {
      if (selected) saveModelConfig(selected, newConfig);
    }, 500);
  };

  const updateConfig = (key, value) => {
    setConfig((prev) => {
      const next = { ...prev, [key]: value };
      persistConfig(next);
      return next;
    });
  };

  const toggleDataset = (dsName) => {
    setConfig((prev) => {
      const ds = prev.datasets || [];
      const next = ds.includes(dsName)
        ? { ...prev, datasets: ds.filter((d) => d !== dsName) }
        : { ...prev, datasets: [...ds, dsName] };
      persistConfig(next);
      return next;
    });
  };

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    const res = await createModel(name);
    if (res.error) {
      alert(res.error);
      return;
    }
    setNewName("");
    setShowAdd(false);
    loadModels();
  };

  const handleDelete = async (name) => {
    if (!confirm(`Delete model version "${name}" and all its files?`)) return;
    await deleteModel(name);
    if (selected === name) setSelected(null);
    loadModels();
  };

  const handleStart = async () => {
    const res = await startModelTraining(selected);
    if (res.error) {
      setLog(res.error);
      return;
    }
    setRunning(true);
    setLog(res.message + "\n");
    pollRef.current = setInterval(() => pollStatus(selected), 2000);
  };

  const handleStop = async () => {
    const res = await stopModelTraining(selected);
    setRunning(false);
    setLog((prev) => prev + "\n" + res.message);
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const handlePredict = async () => {
    setPredicting(true);
    setPredictions(null);
    try {
      const res = await predictModelSamples(selected, 5);
      if (res.error) {
        setPredictions([]);
        setLog((prev) => prev + "\n" + res.error);
      } else {
        setPredictions(res);
      }
    } catch {
      setPredictions([]);
    }
    setPredicting(false);
  };

  if (loading) return <div style={styles.empty}>Loading models...</div>;

  // ═══════════════════════════════════════════════════
  //  Detail view — model version config + training
  // ═══════════════════════════════════════════════════

  if (selected) {
    const selectedDatasets = config.datasets || [];
    const totalSamples = datasets
      .filter((d) => selectedDatasets.includes(d.name))
      .reduce((sum, d) => sum + d.count, 0);

    const fields = [
      { key: "epochs", label: "Epochs", type: "number" },
      {
        key: "learning_rate",
        label: "Learning rate",
        type: "number",
        step: "any",
      },
      { key: "train_batch_size", label: "Train batch size", type: "number" },
      { key: "eval_batch_size", label: "Eval batch size", type: "number" },
      { key: "logging_steps", label: "Logging steps", type: "number" },
      { key: "save_steps", label: "Save steps", type: "number" },
      { key: "eval_steps", label: "Eval steps", type: "number" },
    ];

    return (
      <div>
        <button style={styles.backBtn} onClick={() => setSelected(null)}>
          &larr; Back
        </button>

        <div style={styles.detailHeader}>
          <span style={styles.detailTitle}>{selected}</span>
          <button
            style={styles.btnSmallDanger}
            onClick={() => handleDelete(selected)}
          >
            Delete Model
          </button>
        </div>

        {/* Dataset selection */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Datasets</div>
          {datasets.length === 0 && (
            <div style={{ color: "#9ca3af", fontSize: "0.9rem" }}>
              No datasets available. Create one in the Dataset tab.
            </div>
          )}
          {datasets.map((ds) => (
            <label key={ds.name} style={styles.checkboxRow}>
              <input
                type="checkbox"
                checked={selectedDatasets.includes(ds.name)}
                onChange={() => toggleDataset(ds.name)}
                disabled={running}
              />
              {ds.name}{" "}
              <span style={{ color: "#9ca3af", fontSize: "0.8rem" }}>
                ({ds.count} samples)
              </span>
            </label>
          ))}
          {totalSamples > 0 && (
            <div
              style={{
                color: "#666",
                fontSize: "0.85rem",
                marginTop: "4px",
              }}
            >
              {totalSamples} total samples selected
            </div>
          )}
        </div>

        {/* Hyperparameters */}
        <div style={styles.form}>
          {fields.map(({ key, label, type, step }) => (
            <div key={key} style={styles.field}>
              <label style={styles.label}>{label}</label>
              <input
                style={styles.fieldInput}
                type={type}
                step={step}
                value={config[key]}
                onChange={(e) => updateConfig(key, e.target.value)}
                disabled={running}
              />
            </div>
          ))}
          <div
            style={{
              ...styles.field,
              ...styles.checkbox,
              justifyContent: "flex-start",
            }}
          >
            <input
              type="checkbox"
              checked={config.fp16}
              onChange={(e) => updateConfig("fp16", e.target.checked)}
              disabled={running}
            />
            <label style={styles.label}>FP16 (mixed precision)</label>
          </div>
        </div>

        {/* Training step info */}
        {totalSamples > 0 &&
          (() => {
            const batchSize = Number(config.train_batch_size) || 1;
            const epochs = Number(config.epochs) || 1;
            const trainSamples = Math.round(totalSamples * 0.9);
            const stepsPerEpoch = Math.ceil(trainSamples / batchSize);
            const totalSteps = stepsPerEpoch * epochs;
            return (
              <p
                style={{
                  color: "#666",
                  fontSize: "0.85rem",
                  marginBottom: "12px",
                }}
              >
                {totalSamples} samples ({trainSamples} train) &middot;{" "}
                {stepsPerEpoch} steps/epoch &middot;{" "}
                <strong>{totalSteps} total steps</strong>
              </p>
            );
          })()}

        {/* Buttons */}
        <div style={styles.buttons}>
          <button
            style={{
              ...styles.btn("primary"),
              ...(running ? styles.btnDisabled : {}),
            }}
            onClick={handleStart}
            disabled={running}
          >
            Start Training
          </button>
          <button
            style={{
              ...styles.btn("danger"),
              ...(!running ? styles.btnDisabled : {}),
            }}
            onClick={handleStop}
            disabled={!running}
          >
            Stop Training
          </button>
          <button
            style={{
              ...styles.btn(),
              ...(running ? styles.btnDisabled : {}),
            }}
            onClick={() => {
              window.location.href = `/api/models/${encodeURIComponent(selected)}/download`;
            }}
            disabled={running}
          >
            Download Model
          </button>
          <button
            style={{
              ...styles.btn(),
              ...(running || predicting ? styles.btnDisabled : {}),
            }}
            onClick={handlePredict}
            disabled={running || predicting}
          >
            {predicting ? "Testing..." : "Test Model"}
          </button>
        </div>

        {/* Log */}
        <div style={styles.logArea} ref={logEndRef}>
          {log || "No training output yet."}
        </div>

        {/* Predictions */}
        {predictions && predictions.length > 0 && (
          <div style={{ marginTop: "20px" }}>
            <h3 style={{ fontSize: "1rem", marginBottom: "10px" }}>
              Sample Predictions
            </h3>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "0.9rem",
              }}
            >
              <thead>
                <tr
                  style={{
                    borderBottom: "2px solid #e5e7eb",
                    textAlign: "left",
                  }}
                >
                  <th style={{ padding: "8px" }}>Expected</th>
                  <th style={{ padding: "8px" }}>Predicted</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p, i) => (
                  <tr
                    key={i}
                    style={{ borderBottom: "1px solid #e5e7eb" }}
                  >
                    <td style={{ padding: "8px", color: "#444" }}>
                      {p.expected}
                    </td>
                    <td
                      style={{
                        padding: "8px",
                        color:
                          p.expected.toLowerCase() ===
                          p.predicted.toLowerCase()
                            ? "#16a34a"
                            : "#dc2626",
                      }}
                    >
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

  // ═══════════════════════════════════════════════════
  //  List view — all model versions
  // ═══════════════════════════════════════════════════

  return (
    <div>
      <p style={{ color: "#666", marginBottom: "16px" }}>
        Create model versions, configure datasets and hyperparameters, then
        train.
      </p>

      <div style={{ marginBottom: "16px" }}>
        {showAdd ? (
          <div style={styles.addForm}>
            <input
              style={styles.input}
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Model version name"
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              autoFocus
            />
            <button style={styles.btnPrimary} onClick={handleCreate}>
              Create
            </button>
            <button
              style={styles.btnDefault}
              onClick={() => {
                setShowAdd(false);
                setNewName("");
              }}
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            style={styles.btnPrimary}
            onClick={() => setShowAdd(true)}
          >
            + New Model Version
          </button>
        )}
      </div>

      {models.length === 0 && (
        <div style={styles.empty}>
          No model versions yet. Create one to get started.
        </div>
      )}

      {models.map((m) => (
        <div
          key={m.name}
          style={styles.card}
          onClick={() => setSelected(m.name)}
          onMouseEnter={(e) =>
            (e.currentTarget.style.background = "#f9fafb")
          }
          onMouseLeave={(e) => (e.currentTarget.style.background = "")}
        >
          <span style={styles.cardName}>{m.name}</span>
          <span style={styles.badge(statusColors[m.status] || "#6b7280")}>
            {m.status}
          </span>
          <span style={styles.arrow}>&rsaquo;</span>
        </div>
      ))}
    </div>
  );
}
