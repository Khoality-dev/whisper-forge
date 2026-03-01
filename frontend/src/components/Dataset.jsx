import React, { useState, useEffect, useRef } from "react";
import {
  fetchDatasets,
  createDataset,
  deleteDataset,
  fetchDatasetSamples,
  deleteRecording,
  uploadToDataset,
} from "../api";

const styles = {
  summary: {
    display: "flex",
    gap: "16px",
    marginBottom: "20px",
    flexWrap: "wrap",
  },
  summaryItem: {
    padding: "8px 16px",
    background: "#f3f4f6",
    borderRadius: "6px",
    fontSize: "0.85rem",
    color: "#555",
  },
  summaryValue: {
    fontWeight: 600,
    color: "#111",
  },
  addForm: {
    display: "flex",
    gap: "8px",
    alignItems: "center",
  },
  input: {
    padding: "7px 12px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "0.9rem",
    outline: "none",
  },
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
  badge: {
    fontSize: "0.75rem",
    color: "#6b7280",
    background: "#e5e7eb",
    padding: "2px 8px",
    borderRadius: "10px",
  },
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
  sampleRow: {
    border: "1px solid #f3f4f6",
    borderRadius: "6px",
    padding: "10px 14px",
    marginBottom: "8px",
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  sampleText: {
    flex: 1,
    fontSize: "0.9rem",
    lineHeight: 1.5,
    color: "#1f2937",
  },
  audioPlayer: {
    height: "30px",
    maxWidth: "300px",
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
  btnDanger: {
    padding: "5px 12px",
    border: "none",
    borderRadius: "5px",
    fontSize: "0.8rem",
    cursor: "pointer",
    fontWeight: 500,
    background: "#dc2626",
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
  uploadArea: {
    display: "flex",
    gap: "8px",
    alignItems: "center",
    marginBottom: "16px",
  },
};

const PAGE_SIZE = 100;

export default function Dataset() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [samples, setSamples] = useState([]);
  const [totalSamples, setTotalSamples] = useState(0);
  const [page, setPage] = useState(0);

  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  const loadDatasets = async () => {
    const data = await fetchDatasets();
    setDatasets(data.datasets || []);
    setLoading(false);
  };

  const loadSamples = async (name, p = 0) => {
    const data = await fetchDatasetSamples(name, p * PAGE_SIZE, PAGE_SIZE);
    setSamples(data.samples || []);
    setTotalSamples(data.total || 0);
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selected) {
      setPage(0);
      loadSamples(selected, 0);
    }
  }, [selected]);

  useEffect(() => {
    if (selected) loadSamples(selected, page);
  }, [page]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    const res = await createDataset(name);
    if (res.error) {
      alert(res.error);
      return;
    }
    setNewName("");
    setShowAdd(false);
    loadDatasets();
  };

  const handleDelete = async (name) => {
    if (!confirm(`Delete dataset "${name}" and all its samples?`)) return;
    await deleteDataset(name);
    if (selected === name) setSelected(null);
    loadDatasets();
  };

  const handleDeleteSample = async (sampleName) => {
    if (!confirm("Delete this sample?")) return;
    await deleteRecording(selected, sampleName);
    loadSamples(selected, page);
    loadDatasets();
  };

  const handleUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    await uploadToDataset(selected, files);
    setUploading(false);
    loadSamples(selected, page);
    loadDatasets();
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  if (loading) return <div style={styles.empty}>Loading datasets...</div>;

  // ═══════════════════════════════════════════════════
  //  Detail view — samples for one dataset
  // ═══════════════════════════════════════════════════

  if (selected) {
    return (
      <div>
        <button style={styles.backBtn} onClick={() => setSelected(null)}>
          &larr; Back
        </button>

        <div style={styles.detailHeader}>
          <span style={styles.detailTitle}>{selected}</span>
          <span style={styles.badge}>
            {totalSamples} sample{totalSamples !== 1 ? "s" : ""}
          </span>
          <button
            style={styles.btnDefault}
            onClick={() => {
              window.location.href = `/api/datasets/${encodeURIComponent(selected)}/download`;
            }}
          >
            Export
          </button>
          <button
            style={styles.btnSmallDanger}
            onClick={() => handleDelete(selected)}
          >
            Delete Dataset
          </button>
        </div>

        {/* Upload */}
        <div style={styles.uploadArea}>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".wav,.txt,.zip"
            onChange={handleUpload}
            style={{ fontSize: "0.85rem" }}
          />
          {uploading && (
            <span style={{ fontSize: "0.85rem", color: "#666" }}>
              Uploading...
            </span>
          )}
        </div>

        {samples.length === 0 && totalSamples === 0 && (
          <div style={styles.empty}>
            No samples yet. Upload files or record in the Collect tab.
          </div>
        )}

        {samples.map((s) => (
          <div key={s.filename} style={styles.sampleRow}>
            <audio
              src={`/api/audio/${encodeURIComponent(selected)}/${s.filename}`}
              controls
              preload="none"
              style={styles.audioPlayer}
            />
            <span style={styles.sampleText}>{s.text || "(no text)"}</span>
            <button
              style={styles.btnSmallDanger}
              onClick={() => handleDeleteSample(s.filename)}
            >
              Delete
            </button>
          </div>
        ))}

        {/* Pagination */}
        {totalSamples > PAGE_SIZE && (
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "12px",
            marginTop: "16px",
            fontSize: "0.9rem",
          }}>
            <button
              style={{ ...styles.btnDefault, ...(page === 0 ? { opacity: 0.4, cursor: "not-allowed" } : {}) }}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
            >
              Prev
            </button>
            <span style={{ color: "#666" }}>
              {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, totalSamples)} of {totalSamples}
            </span>
            <button
              style={{ ...styles.btnDefault, ...((page + 1) * PAGE_SIZE >= totalSamples ? { opacity: 0.4, cursor: "not-allowed" } : {}) }}
              onClick={() => setPage((p) => p + 1)}
              disabled={(page + 1) * PAGE_SIZE >= totalSamples}
            >
              Next
            </button>
          </div>
        )}
      </div>
    );
  }

  // ═══════════════════════════════════════════════════
  //  List view — all datasets
  // ═══════════════════════════════════════════════════

  const allSamplesCount = datasets.reduce((sum, d) => sum + d.count, 0);

  return (
    <div>
      <div style={styles.summary}>
        <div style={styles.summaryItem}>
          <span style={styles.summaryValue}>{datasets.length}</span> dataset
          {datasets.length !== 1 ? "s" : ""}
        </div>
        <div style={styles.summaryItem}>
          <span style={styles.summaryValue}>{allSamplesCount}</span> total sample
          {allSamplesCount !== 1 ? "s" : ""}
        </div>
      </div>

      <div style={{ marginBottom: "16px" }}>
        {showAdd ? (
          <div style={styles.addForm}>
            <input
              style={styles.input}
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Dataset name"
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
          <button style={styles.btnPrimary} onClick={() => setShowAdd(true)}>
            + New Dataset
          </button>
        )}
      </div>

      {datasets.length === 0 && (
        <div style={styles.empty}>
          No datasets yet. Create a dataset to get started.
        </div>
      )}

      {datasets.map((ds) => (
        <div
          key={ds.name}
          style={styles.card}
          onClick={() => setSelected(ds.name)}
          onMouseEnter={(e) =>
            (e.currentTarget.style.background = "#f9fafb")
          }
          onMouseLeave={(e) => (e.currentTarget.style.background = "")}
        >
          <span style={styles.cardName}>{ds.name}</span>
          <span style={styles.badge}>
            {ds.count} sample{ds.count !== 1 ? "s" : ""}
          </span>
          <span style={styles.arrow}>&rsaquo;</span>
        </div>
      ))}
    </div>
  );
}
