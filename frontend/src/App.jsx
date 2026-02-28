import React, { useState } from "react";
import Dataset from "./components/Dataset";
import Collect from "./components/Collect";
import Train from "./components/Train";

const styles = {
  app: {
    maxWidth: 900,
    margin: "0 auto",
    padding: "20px",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  title: {
    fontSize: "2rem",
    fontWeight: 700,
    marginBottom: "20px",
  },
  tabs: {
    display: "flex",
    gap: "0",
    borderBottom: "2px solid #e0e0e0",
    marginBottom: "24px",
  },
  tab: (active) => ({
    padding: "10px 24px",
    cursor: "pointer",
    border: "none",
    background: "none",
    fontSize: "1rem",
    fontWeight: active ? 600 : 400,
    color: active ? "#2563eb" : "#666",
    borderBottom: active ? "2px solid #2563eb" : "2px solid transparent",
    marginBottom: "-2px",
  }),
};

export default function App() {
  const [tab, setTab] = useState("dataset");

  return (
    <div style={styles.app}>
      <h1 style={styles.title}>WhisperForge</h1>
      <div style={styles.tabs}>
        <button style={styles.tab(tab === "dataset")} onClick={() => setTab("dataset")}>
          Dataset
        </button>
        <button style={styles.tab(tab === "collect")} onClick={() => setTab("collect")}>
          Collect
        </button>
        <button style={styles.tab(tab === "train")} onClick={() => setTab("train")}>
          Train
        </button>
      </div>
      {tab === "dataset" && <Dataset />}
      {tab === "collect" && <Collect />}
      {tab === "train" && <Train />}
    </div>
  );
}
