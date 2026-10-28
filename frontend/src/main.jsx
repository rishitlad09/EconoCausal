
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Upload,
  Wallet,
  FileSpreadsheet,
  CheckCircle,
  AlertCircle,
  BarChart3
} from "lucide-react";
import "./style.css";

function App() {
  const [budget, setBudget] = useState("50000");
  const [fileName, setFileName] = useState("");
  const [rows, setRows] = useState([]);
  const [message, setMessage] = useState("");

  

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".csv")) {
    setMessage("Please upload a CSV file.");
    setFileName("");
    setRows([]);
    return;
  }

  setFileName(file.name);
  setMessage("Reading CSV and connecting to backend...");
  setRows([]);

  // Read the CSV for the existing five-row preview
  const reader = new FileReader();

  reader.onload = (e) => {
    const text = e.target.result;
    const lines = text.trim().split(/\r?\n/);

    if (lines.length < 2) {
      setRows([]);
      setMessage("The CSV has no data rows.");
      return;
    }

    const headers = lines[0].split(",").map((h) => h.trim());

    const preview = lines.slice(1, 6).map((line) => {
      const values = line.split(",");
      return Object.fromEntries(
        headers.map((header, index) => [
          header,
          values[index] ?? ""
        ])
      );
    });

    setRows(preview);
  };

  reader.readAsText(file);

  // Send the complete CSV to FastAPI
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(
      "https://wild-cemetery-4jwg7wv75pj534rw-8000.app.github.dev/dataset/summary",
      {
        method: "POST",
        body: formData
      }
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Backend request failed.");
    }

    setMessage(
      `Backend success: ${result.rows} rows, ${result.columns} columns.`
    );

    console.log("Backend dataset summary:", result);
  } catch (error) {
    setMessage(`Backend upload failed: ${error.message}`);
  }
}


  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">E</div>
          <div>
            <h2>EconoCausal</h2>
            <p>Pricing intelligence</p>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>
        <div className="nav-item active">
          <BarChart3 size={18} /> Dashboard
        </div>
        <div className="nav-item">
          <FileSpreadsheet size={18} /> Campaign data
        </div>
        <div className="nav-item">
          <Wallet size={18} /> Budget settings
        </div>

        <div className="sidebar-footer">
          Week 1 · Dashboard scaffold
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">CAMPAIGN WORKSPACE</p>
            <h1>Dashboard</h1>
          </div>
          <span className="status">Prototype</span>
        </header>

        <section className="welcome">
          <div>
            <p className="eyebrow">ECONOMICS + CAUSAL INFERENCE</p>
            <h2>Understand your campaign data.</h2>
            <p>
              Upload historical campaigns and set a budget constraint
              to prepare for future pricing analysis.
            </p>
          </div>
          <div className="welcome-icon">
            <BarChart3 size={36} />
          </div>
        </section>

        <section className="cards">
          <article className="card">
            <div className="card-heading">
              <span>Historical campaign data</span>
              <FileSpreadsheet size={20} />
            </div>
            <h3>Upload a CSV file</h3>
            <p className="muted">
              Choose a historical campaign dataset from your computer.
            </p>

            <label className="upload-area">
              <Upload size={26} />
              <strong>Choose CSV file</strong>
              <span>CSV format only</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={handleFileUpload}
              />
            </label>

            {fileName && (
              <p className="file-info">
                <CheckCircle size={16} /> {fileName}
              </p>
            )}

            {message && (
              <p className={message.startsWith("CSV loaded") ? "file-info" : "message"}>
                <AlertCircle size={16} /> {message}
              </p>
            )}
          </article>

          <article className="card">
            <div className="card-heading">
              <span>Budget constraints</span>
              <Wallet size={20} />
            </div>
            <h3>Set campaign budget</h3>
            <p className="muted">
              Enter the maximum budget available for a campaign.
            </p>

            <label className="field-label" htmlFor="budget">
              Maximum budget (₹)
            </label>
            <input
              id="budget"
              className="budget-input"
              type="number"
              min="0"
              value={budget}
              onChange={(event) => setBudget(event.target.value)}
            />

            <div className="budget-summary">
              <span>Selected budget</span>
              <strong>
                ₹{Number(budget || 0).toLocaleString("en-IN")}
              </strong>
            </div>

            <p className="muted small">
              This setting is local to this prototype; it is not yet
              connected to an optimization engine.
            </p>
          </article>
        </section>

        <section className="card preview-card">
          <div className="preview-heading">
            <div>
              <h3>Dataset preview</h3>
              <p className="muted">
                The first five data rows appear here after CSV upload.
              </p>
            </div>
            {rows.length > 0 && (
              <span className="status">{rows.length} rows shown</span>
            )}
          </div>

          {rows.length === 0 ? (
            <div className="empty-state">
              <FileSpreadsheet size={30} />
              <p>No dataset uploaded yet.</p>
            </div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {Object.keys(rows[0]).map((header) => (
                      <th key={header}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {Object.values(row).map((value, index) => (
                        <td key={index}>{value}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <footer>
          EconoCausal · Week 1 frontend prototype
        </footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
