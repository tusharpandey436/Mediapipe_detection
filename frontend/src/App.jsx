import React, { useEffect, useRef, useState } from 'react';
import useWebSocketLib, { ReadyState } from 'react-use-websocket';
import { Activity, Brain, Camera, FolderOpen, RefreshCcw, StopCircle, Upload } from 'lucide-react';
import './App.css';

const useWebSocket = typeof useWebSocketLib === 'function'
  ? useWebSocketLib
  : useWebSocketLib?.default;

const API_BASE_URL = 'http://127.0.0.1:8000';
const WS_URL = `${API_BASE_URL.replace('http', 'ws')}/ws/live`;
const VIDEO_SUMMARY_URL = `${API_BASE_URL}/upload/video-summary`;
const RESULTS_LIST_URL = `${API_BASE_URL}/results`;

const defaultStats = {
  face_detected: false,
  attention: 100,
  attention_final: 100,
  focus_pct: 0,
  blinks: 0,
  blink_rate: 0,
  microsleeps: 0,
  perclos: 0,
  eye_dir: 'Center',
  head_dir: 'Center',
  ear: 0.3,
  alerts: [],
  eye_counts: { L: 0, R: 0, U: 0, D: 0 },
  head_counts: { L: 0, R: 0, U: 0, D: 0 },
  pose: { yaw: 0, pitch: 0, roll: 0 },
  duration_s: 0,
  processed_frames: 0,
  source_video: '',
  session_file: '',
  face_box: null,
};

function normalizeSummary(summary) {
  const eyeCounts = summary.eye_counts || {};
  const headCounts = summary.head_counts || {};

  return {
    ...defaultStats,
    ...summary,
    attention: summary.attention ?? summary.attention_final ?? defaultStats.attention,
    attention_final: summary.attention_final ?? summary.attention ?? defaultStats.attention,
    blink_rate: summary.blink_rate ?? summary.blinks ?? 0,
    eye_counts: {
      L: eyeCounts.L ?? eyeCounts.Left ?? 0,
      R: eyeCounts.R ?? eyeCounts.Right ?? 0,
      U: eyeCounts.U ?? eyeCounts.Up ?? 0,
      D: eyeCounts.D ?? eyeCounts.Down ?? 0,
    },
    head_counts: {
      L: headCounts.L ?? headCounts.Left ?? 0,
      R: headCounts.R ?? headCounts.Right ?? 0,
      U: headCounts.U ?? headCounts.Up ?? 0,
      D: headCounts.D ?? headCounts.Down ?? 0,
    },
  };
}

function App() {
  const [isLive, setIsLive] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [stats, setStats] = useState(defaultStats);
  const [stream, setStream] = useState(null);
  const [uploadedVideoName, setUploadedVideoName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [analysisMode, setAnalysisMode] = useState('live');
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState('');
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState('');
  const [eventTimeline, setEventTimeline] = useState([]);
  const [currentUploadTime, setCurrentUploadTime] = useState(0);
  const [uploadTrace, setUploadTrace] = useState([]);
  const [liveTimeline, setLiveTimeline] = useState([]);
  const [liveSessionTime, setLiveSessionTime] = useState(0);
  const [savedResults, setSavedResults] = useState([]);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [selectedResultId, setSelectedResultId] = useState('');
  const reloadTimerRef = useRef(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const uploadedVideoRef = useRef(null);
  const fileInputRef = useRef(null);
  const intervalRef = useRef(null);
  const readyStateRef = useRef(ReadyState.UNINSTANTIATED);

  const { sendMessage, lastJsonMessage, readyState } = useWebSocket(WS_URL, {
    shouldReconnect: () => true,
    reconnectAttempts: 10,
    reconnectInterval: 3000,
  });

  useEffect(() => {
    readyStateRef.current = readyState;
  }, [readyState]);

  useEffect(() => {
    if (lastJsonMessage && !lastJsonMessage.error) {
      setAnalysisMode('live');
      setStats((prev) => ({ ...prev, ...lastJsonMessage }));
      if (typeof lastJsonMessage.session_t === 'number') {
        setLiveSessionTime(lastJsonMessage.session_t);
      }
      if (Array.isArray(lastJsonMessage.frame_events) && lastJsonMessage.frame_events.length > 0) {
        setLiveTimeline((prev) => {
          const next = [...prev, ...lastJsonMessage.frame_events];
          return next.slice(-300);
        });
      }
    }
  }, [lastJsonMessage]);

  useEffect(() => {
    if (isLive) {
      intervalRef.current = setInterval(() => {
        const rs = readyStateRef.current;
        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (rs !== ReadyState.OPEN || !video || !canvas || video.readyState < 2) {
          return;
        }

        const ctx = canvas.getContext('2d');
        if (video.videoWidth > 0) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          const data = canvas.toDataURL('image/jpeg', 0.6);
          sendMessage(data);
        }
      }, 100);
    }

    return () => {
      clearInterval(intervalRef.current);
    };
  }, [isLive, sendMessage]);

  useEffect(() => () => {
    window.clearTimeout(reloadTimerRef.current);
  }, []);

  useEffect(() => {
    const loadResults = async () => {
      setResultsLoading(true);
      try {
        const response = await fetch(RESULTS_LIST_URL);
        const payload = await response.json();
        if (response.ok) {
          setSavedResults(Array.isArray(payload.results) ? payload.results : []);
        }
      } catch {
        // no-op
      } finally {
        setResultsLoading(false);
      }
    };
    loadResults();
  }, []);

  useEffect(() => {
    return () => {
      if (uploadedVideoUrl && uploadedVideoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(uploadedVideoUrl);
      }
    };
  }, [uploadedVideoUrl]);

  const resetStats = () => {
    setStats(defaultStats);
    setShowReport(false);
    setUploadError('');
    setEventTimeline([]);
    setUploadTrace([]);
    setLiveTimeline([]);
    setLiveSessionTime(0);
    setSelectedResultId('');
    setAnnotatedVideoUrl('');
    window.clearTimeout(reloadTimerRef.current);
  };

  const startLive = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      videoRef.current.srcObject = mediaStream;
      setStream(mediaStream);
      resetStats();
      setUploadedVideoName('');
      setCurrentUploadTime(0);
      setLiveTimeline([]);
      setLiveSessionTime(0);
      setAnalysisMode('live');
      setIsLive(true);
    } catch {
      alert('Could not access webcam. Please allow camera permissions.');
    }
  };

  const stopLive = () => {
    setIsLive(false);
    stream?.getTracks().forEach((track) => track.stop());
    setStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    clearInterval(intervalRef.current);
    setShowReport(true);
    window.clearTimeout(reloadTimerRef.current);
    reloadTimerRef.current = window.setTimeout(() => {
      window.location.reload();
    }, 3000);
  };

  const handleVideoUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    setUploadError('');
    setShowReport(false);
    setAnalysisMode('upload');
    setUploadedVideoName(file.name);
    setCurrentUploadTime(0);
    setLiveTimeline([]);
    setLiveSessionTime(0);
    setAnnotatedVideoUrl('');
    if (uploadedVideoUrl && uploadedVideoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(uploadedVideoUrl);
    }
    setUploadedVideoUrl('');
    window.clearTimeout(reloadTimerRef.current);

    try {
      const response = await fetch(VIDEO_SUMMARY_URL, {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Video upload failed.');
      }

      const summary = normalizeSummary(payload.summary || {});
      setStats(summary);
      setEventTimeline(Array.isArray(summary.events) ? summary.events : []);
      setUploadTrace(Array.isArray(summary.frame_trace) ? summary.frame_trace : []);
      setUploadedVideoUrl(payload.video_url ? `${API_BASE_URL}${payload.video_url}` : '');
      setAnnotatedVideoUrl(payload.annotated_video_url ? `${API_BASE_URL}${payload.annotated_video_url}` : '');
      setShowReport(true);
      setSelectedResultId(payload.result_id || '');
      try {
        const listResp = await fetch(RESULTS_LIST_URL);
        const listPayload = await listResp.json();
        if (listResp.ok) {
          setSavedResults(Array.isArray(listPayload.results) ? listPayload.results : []);
        }
      } catch {
        // no-op
      }
    } catch (error) {
      setUploadError(error.message || 'Video upload failed.');
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const wsStatus = {
    [ReadyState.CONNECTING]: 'Connecting...',
    [ReadyState.OPEN]: 'Connected',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Disconnected',
    [ReadyState.UNINSTANTIATED]: 'Not started',
  }[readyState];

  const isConnected = readyState === ReadyState.OPEN;
  const showLiveDirection = isLive && stats.face_detected;
  const eyeWarn = stats.eye_dir && stats.eye_dir !== 'Center';
  const headWarn = stats.head_dir && stats.head_dir !== 'Center';

  const formatSeconds = (value) => {
    const safe = Math.max(0, Number(value) || 0);
    const mins = Math.floor(safe / 60);
    const secs = Math.floor(safe % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const jumpToEvent = (t) => {
    if (!uploadedVideoRef.current) return;
    uploadedVideoRef.current.currentTime = Number(t) || 0;
    uploadedVideoRef.current.play();
  };

  const getActiveTracePoint = () => {
    if (uploadTrace.length === 0) return null;
    const t = currentUploadTime;
    let active = uploadTrace[0];
    for (let i = 1; i < uploadTrace.length; i += 1) {
      if (uploadTrace[i].t <= t) active = uploadTrace[i];
      else break;
    }
    return active;
  };

  const activeTrace = getActiveTracePoint();
  const uploadFaceBox = activeTrace?.face_box;

  const refreshResults = async () => {
    setResultsLoading(true);
    try {
      const response = await fetch(RESULTS_LIST_URL);
      const payload = await response.json();
      if (response.ok) {
        setSavedResults(Array.isArray(payload.results) ? payload.results : []);
      }
    } catch {
      // no-op
    } finally {
      setResultsLoading(false);
    }
  };

  const openSavedResult = async (resultId) => {
    setSelectedResultId(resultId);
    setUploading(true);
    setUploadError('');
    setShowReport(false);
    try {
      const response = await fetch(`${API_BASE_URL}/results/${resultId}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Could not load saved result.');
      }

      const result = payload.result || {};
      const summary = normalizeSummary(result.summary || {});
      setStats(summary);
      setEventTimeline(Array.isArray(summary.events) ? summary.events : []);
      setUploadTrace(Array.isArray(summary.frame_trace) ? summary.frame_trace : []);
      setUploadedVideoName(result.video_file || summary.source_video || 'saved video');
      if (uploadedVideoUrl && uploadedVideoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(uploadedVideoUrl);
      }
      const relativeVideoUrl = result.video_url || `/results/${resultId}/video`;
      setUploadedVideoUrl(`${API_BASE_URL}${relativeVideoUrl}`);
      const relativeAnnotatedUrl = result.annotated_video_url || `/results/${resultId}/video/annotated`;
      setAnnotatedVideoUrl(`${API_BASE_URL}${relativeAnnotatedUrl}`);
      setCurrentUploadTime(0);
      setAnalysisMode('upload');
    } catch (error) {
      setUploadError(error.message || 'Could not load saved result.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="title">
          <Brain size={36} color="var(--primary)" />
          CogniTracker
        </div>

        <div className="controls controls-wrap">
          <div className="connection-status">
            <span className={`status-dot ${isConnected ? 'connected' : 'error'}`} />
            {wsStatus}
          </div>

          {!isLive ? (
            <button className="btn btn-primary" onClick={startLive} disabled={!isConnected || uploading}>
              <Camera size={20} /> Start Session
            </button>
          ) : (
            <button className="btn btn-danger" onClick={stopLive}>
              <StopCircle size={20} /> End Session
            </button>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleVideoUpload}
            style={{ display: 'none' }}
          />
          <button
            className="btn btn-secondary"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || isLive}
          >
            <Upload size={20} /> {uploading ? 'Uploading...' : 'Upload Video'}
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => window.location.reload()}
            disabled={uploading}
          >
            <RefreshCcw size={18} /> Refresh
          </button>
          <button
            className="btn btn-secondary"
            onClick={refreshResults}
            disabled={uploading || resultsLoading}
          >
            <FolderOpen size={18} /> {resultsLoading ? 'Loading...' : 'Saved Files'}
          </button>
        </div>
      </header>

      <main className="dashboard">
        <div className="glass-panel video-section">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="video-feed"
            style={{ display: isLive ? 'block' : 'none' }}
          />
          {showLiveDirection && stats.face_box && (
            <div
              className="live-face-box"
              style={{
                left: `${(1 - (stats.face_box.x + stats.face_box.w)) * 100}%`,
                top: `${stats.face_box.y * 100}%`,
                width: `${stats.face_box.w * 100}%`,
                height: `${stats.face_box.h * 100}%`,
              }}
            />
          )}
          {isLive && (
            <div className="live-status-panel">
              <div className={`live-status-chip ${eyeWarn ? 'warn' : ''}`}>
                Eye: {stats.eye_dir || 'Center'}
              </div>
              <div className={`live-status-chip ${headWarn ? 'warn' : ''}`}>
                Head: {stats.head_dir || 'Center'}
              </div>
            </div>
          )}
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {!isLive && analysisMode === 'upload' && uploading ? (
            <div className="upload-loading">
              <div className="loader-ring" />
              <h3>Processing Video Frames</h3>
              <p>Your report is being generated. Playback will appear when analysis is ready.</p>
            </div>
          ) : !isLive && analysisMode === 'upload' && uploadedVideoUrl ? (
            <>
              <video
                ref={uploadedVideoRef}
                className="upload-playback"
                src={uploadedVideoUrl}
                controls
                onTimeUpdate={() => setCurrentUploadTime(uploadedVideoRef.current?.currentTime || 0)}
                onError={() => {
                  if (annotatedVideoUrl && uploadedVideoUrl !== annotatedVideoUrl) {
                    setUploadedVideoUrl(annotatedVideoUrl);
                  }
                }}
              />
              <div className="upload-detection-zone" />
              {uploadFaceBox && activeTrace?.face_detected && (
                <>
                  <div
                    className="upload-face-box"
                    style={{
                      left: `${uploadFaceBox.x * 100}%`,
                      top: `${uploadFaceBox.y * 100}%`,
                      width: `${uploadFaceBox.w * 100}%`,
                      height: `${uploadFaceBox.h * 100}%`,
                    }}
                  />
                  <div
                    className="upload-eye-zone"
                    style={{
                      left: `${(uploadFaceBox.x + uploadFaceBox.w * 0.22) * 100}%`,
                      top: `${(uploadFaceBox.y + uploadFaceBox.h * 0.3) * 100}%`,
                      width: `${(uploadFaceBox.w * 0.56) * 100}%`,
                      height: `${(uploadFaceBox.h * 0.18) * 100}%`,
                    }}
                  />
                </>
              )}
              <div className="upload-overlay-labels">
                <span className="overlay-chip">Face detection zone</span>
                <span className={`overlay-chip ${activeTrace?.eye_dir !== 'Center' ? 'warn' : ''}`}>
                  Eye: {activeTrace?.eye_dir || 'Center'}
                </span>
                <span className={`overlay-chip ${activeTrace?.head_dir !== 'Center' ? 'warn' : ''}`}>
                  Head: {activeTrace?.head_dir || 'Center'}
                </span>
              </div>
            </>
          ) : !isLive && (
            <div className="placeholder">
              <Activity size={64} style={{ opacity: 0.4 }} />
              <p>
                {analysisMode === 'upload' && uploadedVideoName
                  ? `Latest upload: ${uploadedVideoName}`
                  : 'Camera offline. Start a live session or upload a video for analysis.'}
              </p>
              {uploadError && <p className="upload-error">{uploadError}</p>}
            </div>
          )}

          {isLive && stats.alerts?.length > 0 && (
            <div className="alerts-container">
              {stats.alerts.map((alert, index) => (
                <div key={index} className="alert-toast">
                  {alert.replace('_', ' ').toUpperCase()}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="stats-grid">
          <div className="glass-panel stat-card">
            <div style={{ width: '100%' }}>
              <div className="stat-title">Attention Score</div>
              <div className="stat-value" style={{ color: stats.attention < 40 ? 'var(--danger)' : 'var(--secondary)' }}>
                {stats.attention?.toFixed(1)}%
              </div>
              <div className="progress-container">
                <div
                  className="progress-bar"
                  style={{
                    width: `${stats.attention}%`,
                    background: stats.attention < 40 ? 'var(--danger)' : undefined,
                  }}
                />
              </div>
            </div>
          </div>

          <div className="glass-panel stat-card">
            <div style={{ width: '100%' }}>
              <div className="stat-title">Focus Time</div>
              <div className="stat-value">{stats.focus_pct}%</div>
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${stats.focus_pct}%` }} />
              </div>
            </div>
          </div>

          <div className={`glass-panel stat-card ${stats.alerts?.includes('microsleep') ? 'stat-alert' : ''}`}>
            <div style={{ width: '100%' }}>
              <div className="stat-title">Microsleeps</div>
              <div className="stat-value">{stats.microsleeps}</div>
            </div>
          </div>

          <div className="glass-panel stat-card">
            <div style={{ width: '100%' }}>
              <div className="stat-title">Blinks / Min</div>
              <div className="stat-value">{stats.blink_rate}</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Total: {stats.blinks}</div>
            </div>
          </div>

          <div className={`glass-panel stat-card ${stats.perclos > 15 ? 'stat-alert' : ''}`}>
            <div style={{ width: '100%' }}>
              <div className="stat-title">PERCLOS</div>
              <div className="stat-value">{stats.perclos}%</div>
            </div>
          </div>

          <div className="glass-panel stat-card">
            <div style={{ width: '100%' }}>
              <div className="stat-title">Eye Direction</div>
              <div className="stat-value" style={{ fontSize: '1.4rem', color: stats.eye_dir !== 'Center' ? 'var(--warning)' : 'var(--secondary)' }}>
                {stats.eye_dir}
              </div>
              <div className="direction-row">
                {['L', 'R', 'U', 'D'].map((key) => (
                  <span key={key} className="direction-chip">
                    {key}: <strong>{stats.eye_counts?.[key] ?? 0}</strong>
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="glass-panel stat-card">
            <div style={{ width: '100%' }}>
              <div className="stat-title">Head Direction</div>
              <div className="stat-value" style={{ fontSize: '1.4rem', color: stats.head_dir !== 'Center' ? 'var(--warning)' : 'var(--secondary)' }}>
                {stats.head_dir}
              </div>
              <div className="direction-row">
                {['L', 'R', 'U', 'D'].map((key) => (
                  <span key={key} className="direction-chip">
                    {key}: <strong>{stats.head_counts?.[key] ?? 0}</strong>
                  </span>
                ))}
              </div>
              <div className="pose-line">
                Y: {stats.pose?.yaw?.toFixed?.(1) ?? '0.0'}° | P: {stats.pose?.pitch?.toFixed?.(1) ?? '0.0'}°
              </div>
            </div>
          </div>

          <div className="glass-panel stat-card">
            <div style={{ width: '100%' }}>
              <div className="stat-title">Upload Summary</div>
              <div className="upload-summary">
                <span>Video: {stats.source_video || uploadedVideoName || 'None'}</span>
                <span>Frames: {stats.processed_frames || 0}</span>
                <span>Duration: {stats.duration_s || 0}s</span>
              </div>
            </div>
          </div>

          {analysisMode === 'upload' && eventTimeline.length > 0 && (
            <div className="glass-panel movement-timeline">
              <div className="stat-title">Movement Timeline</div>
              <div className="timeline-meta">Current: {formatSeconds(currentUploadTime)}</div>
              <div className="timeline-list">
                {eventTimeline.map((evt, idx) => (
                  <button
                    key={`${evt.kind}-${evt.t}-${idx}`}
                    className={`timeline-item ${Math.abs((Number(evt.t) || 0) - currentUploadTime) < 0.6 ? 'active' : ''}`}
                    onClick={() => jumpToEvent(evt.t)}
                  >
                    <span className="timeline-time">{formatSeconds(evt.t)}</span>
                    <span className="timeline-kind">{evt.kind}</span>
                    <span className="timeline-detail">{evt.detail || '-'}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {analysisMode === 'live' && (
            <div className="glass-panel movement-timeline">
              <div className="stat-title">Live Movement Timeline</div>
              <div className="timeline-meta">Live Time: {formatSeconds(liveSessionTime)}</div>
              <div className="timeline-list">
                {[...liveTimeline].reverse().map((evt, idx) => (
                  <div key={`${evt.kind}-${evt.t}-${idx}`} className="timeline-item active">
                    <span className="timeline-time">{formatSeconds(evt.t)}</span>
                    <span className="timeline-kind">{evt.kind}</span>
                    <span className="timeline-detail">{evt.detail || '-'}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="glass-panel movement-timeline">
            <div className="stat-title">Saved Results</div>
            <div className="timeline-meta">{resultsLoading ? 'Loading records...' : `${savedResults.length} saved`}</div>
            <div className="timeline-list">
              {savedResults.map((result) => (
                <button
                  key={result.id}
                  className={`timeline-item ${selectedResultId === result.id ? 'active' : ''}`}
                  onClick={() => openSavedResult(result.id)}
                >
                  <span className="timeline-time">{new Date(result.created_at).toLocaleDateString()}</span>
                  <span className="timeline-kind">score {result.attention_final ?? '-'}</span>
                  <span className="timeline-detail">{result.video_file}</span>
                </button>
              ))}
              {!resultsLoading && savedResults.length === 0 && (
                <div className="timeline-detail">No saved results yet.</div>
              )}
            </div>
          </div>
        </div>
      </main>

      {showReport && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>{analysisMode === 'upload' ? 'Video Analysis Report' : 'Session Report'}</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
              {analysisMode === 'upload' ? 'Summary generated from your uploaded video' : 'Your cognitive load analysis'}
            </p>
            {analysisMode === 'live' && (
              <p style={{ color: 'var(--text-muted)', marginBottom: '1.2rem' }}>
                Reloading automatically for a fresh session with attention reset to 100%.
              </p>
            )}
            {analysisMode === 'upload' && (
              <p style={{ color: 'var(--text-muted)', marginBottom: '1.2rem' }}>
                Refreshing automatically in 5 seconds so you can start a new upload quickly.
              </p>
            )}
            <div className="report-grid">
              <div className="report-item">
                <span className="label">Final Attention</span>
                <span className="val" style={{ color: stats.attention < 50 ? 'var(--danger)' : 'var(--success)' }}>
                  {stats.attention?.toFixed(1)}%
                </span>
              </div>
              <div className="report-item">
                <span className="label">Total Focus</span>
                <span className="val">{stats.focus_pct}%</span>
              </div>
              <div className="report-item">
                <span className="label">Microsleeps</span>
                <span className="val" style={{ color: stats.microsleeps > 0 ? 'var(--danger)' : '#fff' }}>
                  {stats.microsleeps}
                </span>
              </div>
              <div className="report-item">
                <span className="label">Total Blinks</span>
                <span className="val">{stats.blinks}</span>
              </div>
              <div className="report-item">
                <span className="label">Duration</span>
                <span className="val">{stats.duration_s || 0}s</span>
              </div>
              <div className="report-item">
                <span className="label">Frames Analysed</span>
                <span className="val">{stats.processed_frames || 0}</span>
              </div>
            </div>
            <button className="btn btn-primary modal-btn" onClick={() => setShowReport(false)}>
              Close Report
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
