import { useState, useRef, useEffect } from "react";

// CRA env (put REACT_APP_API_BASE in .env if you want to override)
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

export default function App() {
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [targetLang, setTargetLang] = useState("en");
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const pollRef = useRef(null);

  const startJob = async (e) => {
    e.preventDefault();
    setError("");
    setStatus(null);
    setJobId(null);

    try {
      const res = await fetch(`${API_BASE}/api/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          youtube_url: youtubeUrl.trim(),
          target_lang: targetLang.trim(),
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setJobId(data.job_id); // your backend should return { job_id: "..." }
    } catch (err) {
      setError(err.message || "Failed to start job");
    }
  };

  // Poll job status
  useEffect(() => {
    if (!jobId) return;
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${jobId}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        setStatus(data);
        if (data.status === "done" || data.status === "error") {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch (err) {
        setError(err.message || "Polling failed");
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
    pollRef.current = setInterval(poll, 3000);
    poll(); // also fire immediately
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobId]);

  const downloadHref = jobId ? `${API_BASE}/api/file/${jobId}` : null;

  return (
    <div style={{ minHeight: "100vh", background: "#f5f5f5" }}>
      <div style={{
        maxWidth: 640, margin: "0 auto", padding: 24,
        background: "#fff", borderRadius: 16, boxShadow: "0 10px 30px rgba(0,0,0,0.08)"
      }}>
        <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 12 }}>Lyra Translator</h1>
        <p style={{ color: "#555", fontSize: 14, marginBottom: 20 }}>
          Paste a YouTube URL and choose a target language. We’ll translate the transcript, synthesize TTS,
          align to vocal windows, and remix with instrumental.
        </p>

        <form onSubmit={startJob}>
          <label style={{ display: "block", fontWeight: 600, fontSize: 14, marginBottom: 6 }}>
            YouTube URL
          </label>
          <input
            type="url"
            required
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            style={{ width: "100%", padding: 10, borderRadius: 8, border: "1px solid #ddd", marginBottom: 14 }}
          />

          <label style={{ display: "block", fontWeight: 600, fontSize: 14, marginBottom: 6 }}>
            Target language (BCP-47 / ISO code)
          </label>
          <input
            type="text"
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            placeholder="en, es, fr, hi, ja ..."
            style={{ width: "100%", padding: 10, borderRadius: 8, border: "1px solid #ddd", marginBottom: 16 }}
          />

          <button
            type="submit"
            disabled={!youtubeUrl}
            style={{
              width: "100%", padding: 12, border: "none", borderRadius: 10,
              background: "#111", color: "#fff", fontWeight: 600, cursor: "pointer"
            }}
          >
            Translate & Mix
          </button>
        </form>

        {error && (
          <div style={{ marginTop: 12, color: "#b00020", fontSize: 14 }}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {jobId && (
          <div style={{ marginTop: 18, paddingTop: 12, borderTop: "1px solid #eee", fontSize: 14 }}>
            <div><strong>Job ID:</strong> {jobId}</div>
            <div style={{ marginTop: 4 }}>
              <strong>Status:</strong>{" "}
              <span style={{
                color:
                  status?.status === "done" ? "#0a7d18" :
                  status?.status === "error" ? "#b00020" : "#1554b3"
              }}>
                {status?.status || "starting..."}
              </span>
            </div>
            {status?.error && (
              <div style={{ marginTop: 4, color: "#b00020", wordBreak: "break-word" }}>
                Details: {status.error}
              </div>
            )}
            {status?.status === "done" && downloadHref && (
              <a
                href={downloadHref}
                style={{
                  display: "inline-block", marginTop: 12, background: "#0a7d18",
                  color: "#fff", borderRadius: 10, padding: "10px 16px", textDecoration: "none"
                }}
              >
                Download Final Track (WAV)
              </a>
            )}
          </div>
        )}

        <div style={{ marginTop: 24, color: "#777", fontSize: 12 }}>
          Tip: leave the page open while processing.
        </div>
      </div>
    </div>
  );
}
