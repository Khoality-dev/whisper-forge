import React, { useState, useEffect, useRef } from "react";
import {
  fetchDataset,
  createLanguage,
  deleteLanguage,
  addLabel,
  removeLabel,
  saveRecording,
  deleteRecording,
} from "../api";

const styles = {
  // ── List view ──
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
  addLangForm: {
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
  langCard: {
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
  langName: {
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

  // ── Detail view ──
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
  inputWide: {
    padding: "7px 12px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    fontSize: "0.9rem",
    outline: "none",
    flex: 1,
  },
  labelRow: {
    border: "1px solid #f3f4f6",
    borderRadius: "6px",
    padding: "10px 14px",
    marginBottom: "8px",
  },
  labelHeader: {
    display: "flex",
    alignItems: "flex-start",
    gap: "8px",
  },
  labelText: {
    flex: 1,
    fontSize: "0.9rem",
    lineHeight: 1.5,
    color: "#1f2937",
  },
  recordingRow: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    marginTop: "6px",
    paddingLeft: "12px",
  },
  audioPlayer: {
    height: "30px",
    flex: 1,
    maxWidth: "400px",
  },
  recorderInline: {
    marginTop: "8px",
    paddingLeft: "12px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    flexWrap: "wrap",
  },
  recordingIndicator: {
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    color: "#dc2626",
    fontSize: "0.85rem",
    fontWeight: 500,
  },
  redDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#dc2626",
    display: "inline-block",
  },
  recordBtn: {
    marginTop: "6px",
    padding: "4px 12px",
    fontSize: "0.8rem",
    background: "none",
    border: "1px solid #2563eb",
    color: "#2563eb",
    borderRadius: "4px",
    cursor: "pointer",
  },
  addLabelRow: {
    display: "flex",
    gap: "8px",
    marginTop: "12px",
    alignItems: "center",
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
  btnSmall: {
    padding: "3px 8px",
    border: "none",
    borderRadius: "4px",
    fontSize: "0.75rem",
    cursor: "pointer",
    background: "#f3f4f6",
    color: "#6b7280",
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
  disabled: {
    opacity: 0.5,
    cursor: "not-allowed",
  },
};

export default function Dataset() {
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedLang, setSelectedLang] = useState(null);

  // Language management
  const [showAddLang, setShowAddLang] = useState(false);
  const [newLangName, setNewLangName] = useState("");
  const [labelInput, setLabelInput] = useState("");

  // Recording
  const [recordingFor, setRecordingFor] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [saving, setSaving] = useState(false);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const loadData = async () => {
    const data = await fetchDataset();
    setGroups(data.groups);
    setLoading(false);
  };

  useEffect(() => {
    loadData();
  }, []);

  const currentGroup = groups.find((g) => g.language === selectedLang);

  // ── Language management ──

  const handleCreateLang = async () => {
    const name = newLangName.trim();
    if (!name) return;
    await createLanguage(name);
    setNewLangName("");
    setShowAddLang(false);
    loadData();
  };

  const handleDeleteLang = async (language) => {
    if (
      !confirm(
        `Delete "${language}" and all its labels and recordings?`
      )
    )
      return;
    await deleteLanguage(language);
    setSelectedLang(null);
    loadData();
  };

  // ── Label management ──

  const handleAddLabel = async () => {
    const text = labelInput.trim();
    if (!text || !selectedLang) return;
    await addLabel(selectedLang, text);
    setLabelInput("");
    loadData();
  };

  const handleRemoveLabel = async (id) => {
    if (!confirm("Remove this label and its recordings?")) return;
    await removeLabel(id);
    loadData();
  };

  // ── Recording ──

  const startRec = async (labelId, replaceFilename = null) => {
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
        setRecordedAudio({ url: URL.createObjectURL(blob), blob });
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      setRecordingFor({ labelId, replaceFilename });
      setIsRecording(true);
      setRecordedAudio(null);
    } catch {
      alert("Microphone access denied. Please allow microphone access.");
    }
  };

  const stopRec = () => {
    if (mediaRecorderRef.current?.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const handleSaveRecording = async () => {
    if (!recordedAudio || !recordingFor) return;
    setSaving(true);
    try {
      await saveRecording(
        recordedAudio.blob,
        recordingFor.labelId,
        recordingFor.replaceFilename
      );
      setRecordingFor(null);
      setRecordedAudio(null);
      await loadData();
    } catch (err) {
      alert(`Error saving: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const cancelRec = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.ondataavailable = null;
      mediaRecorderRef.current.onstop = null;
      if (mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    }
    setIsRecording(false);
    setRecordingFor(null);
    setRecordedAudio(null);
  };

  const handleDeleteRecording = async (filename) => {
    if (!confirm("Delete this recording?")) return;
    await deleteRecording(filename);
    loadData();
  };

  // ── Stats ──

  const totalLabels = groups.reduce((sum, g) => sum + g.sentences.length, 0);
  const totalRecordings = groups.reduce(
    (sum, g) =>
      sum + g.sentences.reduce((s, sent) => s + sent.recordings.length, 0),
    0
  );
  const recordedLabels = groups.reduce(
    (sum, g) =>
      sum + g.sentences.filter((s) => s.recordings.length > 0).length,
    0
  );

  if (loading) return <div style={styles.empty}>Loading dataset...</div>;

  // ═══════════════════════════════════════════════════
  //  Detail view — labels & recordings for one language
  // ═══════════════════════════════════════════════════

  if (selectedLang && currentGroup) {
    const groupLabels = currentGroup.sentences;
    const groupRecordings = groupLabels.reduce(
      (s, l) => s + l.recordings.length,
      0
    );
    const groupRecorded = groupLabels.filter(
      (l) => l.recordings.length > 0
    ).length;

    return (
      <div>
        <button
          style={styles.backBtn}
          onClick={() => {
            cancelRec();
            setSelectedLang(null);
          }}
        >
          &larr; Back
        </button>

        <div style={styles.detailHeader}>
          <span style={styles.detailTitle}>
            {currentGroup.language}
          </span>
          <span style={styles.badge}>
            {groupRecorded} / {groupLabels.length} recorded
          </span>
          <span style={styles.badge}>
            {groupRecordings} recording{groupRecordings !== 1 ? "s" : ""}
          </span>
          <button
            style={styles.btnSmallDanger}
            onClick={() => handleDeleteLang(currentGroup.language)}
          >
            Delete Language
          </button>
        </div>

        {/* Labels */}
        {groupLabels.length === 0 && (
          <div style={styles.empty}>No labels yet. Add one below.</div>
        )}

        {groupLabels.map((sent) => (
          <div key={sent.id} style={styles.labelRow}>
            <div style={styles.labelHeader}>
              <span style={styles.labelText}>{sent.text}</span>
              <button
                style={styles.btnSmallDanger}
                onClick={() => handleRemoveLabel(sent.id)}
              >
                Remove
              </button>
            </div>

            {/* Existing recordings */}
            {sent.recordings.map((rec) => (
              <div key={rec.filename} style={styles.recordingRow}>
                <audio
                  src={`/api/audio/${rec.filename}`}
                  controls
                  preload="none"
                  style={styles.audioPlayer}
                />
                <button
                  style={styles.btnSmall}
                  onClick={() => startRec(sent.id, rec.filename)}
                >
                  Re-record
                </button>
                <button
                  style={styles.btnSmallDanger}
                  onClick={() => handleDeleteRecording(rec.filename)}
                >
                  Delete
                </button>
              </div>
            ))}

            {/* Inline recorder */}
            {recordingFor && recordingFor.labelId === sent.id ? (
              <div style={styles.recorderInline}>
                {isRecording ? (
                  <>
                    <span style={styles.recordingIndicator}>
                      <span style={styles.redDot} />
                      Recording...
                    </span>
                    <button style={styles.btnDanger} onClick={stopRec}>
                      Stop
                    </button>
                    <button style={styles.btnDefault} onClick={cancelRec}>
                      Cancel
                    </button>
                  </>
                ) : recordedAudio ? (
                  <>
                    <audio
                      src={recordedAudio.url}
                      controls
                      style={styles.audioPlayer}
                    />
                    <button
                      style={{
                        ...styles.btnPrimary,
                        ...(saving ? styles.disabled : {}),
                      }}
                      onClick={handleSaveRecording}
                      disabled={saving}
                    >
                      {saving ? "Saving..." : "Save"}
                    </button>
                    <button style={styles.btnDefault} onClick={cancelRec}>
                      Cancel
                    </button>
                  </>
                ) : null}
              </div>
            ) : (
              <button
                style={styles.recordBtn}
                onClick={() => startRec(sent.id)}
              >
                Record
              </button>
            )}
          </div>
        ))}

        {/* Add label */}
        <div style={styles.addLabelRow}>
          <input
            style={styles.inputWide}
            value={labelInput}
            onChange={(e) => setLabelInput(e.target.value)}
            placeholder="Add a label..."
            onKeyDown={(e) => e.key === "Enter" && handleAddLabel()}
          />
          <button style={styles.btnPrimary} onClick={handleAddLabel}>
            Add
          </button>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════
  //  List view — all languages
  // ═══════════════════════════════════════════════════

  return (
    <div>
      {/* Summary */}
      <div style={styles.summary}>
        <div style={styles.summaryItem}>
          <span style={styles.summaryValue}>{groups.length}</span> language
          {groups.length !== 1 ? "s" : ""}
        </div>
        <div style={styles.summaryItem}>
          <span style={styles.summaryValue}>{totalLabels}</span> label
          {totalLabels !== 1 ? "s" : ""}
        </div>
        <div style={styles.summaryItem}>
          <span style={styles.summaryValue}>{recordedLabels}</span> /{" "}
          {totalLabels} recorded
        </div>
        <div style={styles.summaryItem}>
          <span style={styles.summaryValue}>{totalRecordings}</span> recording
          {totalRecordings !== 1 ? "s" : ""}
        </div>
      </div>

      {/* Add language */}
      <div style={{ marginBottom: "16px" }}>
        {showAddLang ? (
          <div style={styles.addLangForm}>
            <input
              style={styles.input}
              value={newLangName}
              onChange={(e) => setNewLangName(e.target.value)}
              placeholder="Language name"
              onKeyDown={(e) => e.key === "Enter" && handleCreateLang()}
              autoFocus
            />
            <button style={styles.btnPrimary} onClick={handleCreateLang}>
              Create
            </button>
            <button
              style={styles.btnDefault}
              onClick={() => {
                setShowAddLang(false);
                setNewLangName("");
              }}
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            style={styles.btnPrimary}
            onClick={() => setShowAddLang(true)}
          >
            + Add Language
          </button>
        )}
      </div>

      {/* Empty state */}
      {groups.length === 0 && (
        <div style={styles.empty}>
          No languages yet. Add a language to get started.
        </div>
      )}

      {/* Language cards */}
      {groups.map((group) => {
        const labelCount = group.sentences.length;
        const recCount = group.sentences.reduce(
          (s, l) => s + l.recordings.length,
          0
        );
        const recLabels = group.sentences.filter(
          (l) => l.recordings.length > 0
        ).length;

        return (
          <div
            key={group.language}
            style={styles.langCard}
            onClick={() => setSelectedLang(group.language)}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "#f9fafb")
            }
            onMouseLeave={(e) => (e.currentTarget.style.background = "")}
          >
            <span style={styles.langName}>
              {group.language}
            </span>
            <span style={styles.badge}>
              {labelCount} label{labelCount !== 1 ? "s" : ""}
            </span>
            <span style={styles.badge}>
              {recLabels} / {labelCount} recorded
            </span>
            <span style={styles.badge}>
              {recCount} recording{recCount !== 1 ? "s" : ""}
            </span>
            <span style={styles.arrow}>&rsaquo;</span>
          </div>
        );
      })}
    </div>
  );
}
