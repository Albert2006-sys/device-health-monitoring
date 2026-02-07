import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import mqtt from 'mqtt';
import {
  Wifi,
  WifiOff,
  Radio,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Unplug,
} from 'lucide-react';

/* ────────────────────────────────────────────────
   Types
   ──────────────────────────────────────────────── */

interface DeviceMessage {
  status: 'HEALTHY' | 'WARNING' | 'ANOMALY';
  fault: string;
  confidence: number;
  error: number;
  /** Added on receipt */
  timestamp?: string;
}

type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

/* ────────────────────────────────────────────────
   Status config
   ──────────────────────────────────────────────── */

const STATUS_CONFIG: Record<
  string,
  { label: string; color: string; bg: string; border: string; icon: React.ReactNode }
> = {
  HEALTHY: {
    label: 'Healthy',
    color: 'text-primary-green',
    bg: 'bg-primary-green/10',
    border: 'border-primary-green',
    icon: <CheckCircle size={28} className="text-primary-green" />,
  },
  WARNING: {
    label: 'Warning',
    color: 'text-yellow-400',
    bg: 'bg-yellow-400/10',
    border: 'border-yellow-500',
    icon: <AlertTriangle size={28} className="text-yellow-400" />,
  },
  ANOMALY: {
    label: 'Anomaly',
    color: 'text-primary-red',
    bg: 'bg-primary-red/10',
    border: 'border-primary-red',
    icon: <AlertTriangle size={28} className="text-primary-red" />,
  },
};

const DEFAULT_BROKER = 'wss://broker.hivemq.com:8884/mqtt';
const DEFAULT_TOPIC = 'car/engine/diagnostics';
const MAX_HISTORY = 50;

/* ────────────────────────────────────────────────
   Component
   ──────────────────────────────────────────────── */

export const LiveDeviceMonitor: React.FC = () => {
  /* Connection settings */
  const [brokerUrl, setBrokerUrl] = useState(DEFAULT_BROKER);
  const [topic, setTopic] = useState(DEFAULT_TOPIC);
  const [connState, setConnState] = useState<ConnectionState>('disconnected');
  const [connError, setConnError] = useState<string | null>(null);

  /* Live data */
  const [latest, setLatest] = useState<DeviceMessage | null>(null);
  const [history, setHistory] = useState<DeviceMessage[]>([]);
  const [pulseKey, setPulseKey] = useState(0); // triggers pulse animation

  const clientRef = useRef<mqtt.MqttClient | null>(null);

  /* ─── Connect ─── */
  const handleConnect = useCallback(() => {
    if (clientRef.current) return;
    setConnState('connecting');
    setConnError(null);

    try {
      const client = mqtt.connect(brokerUrl, {
        reconnectPeriod: 5000,
        connectTimeout: 10000,
      });

      client.on('connect', () => {
        setConnState('connected');
        setConnError(null);
        client.subscribe(topic, { qos: 0 }, (err) => {
          if (err) setConnError(`Subscribe error: ${err.message}`);
        });
      });

      client.on('error', (err) => {
        setConnState('error');
        setConnError(err.message);
      });

      client.on('close', () => {
        setConnState('disconnected');
      });

      client.on('message', (_t, payload) => {
        try {
          const raw = JSON.parse(payload.toString());
          // Validate expected shape
          if (
            typeof raw.status !== 'string' ||
            typeof raw.confidence !== 'number' ||
            typeof raw.error !== 'number'
          ) {
            return; // silently discard malformed messages
          }
          const msg: DeviceMessage = {
            status: raw.status,
            fault: raw.fault ?? 'None',
            confidence: Math.max(0, Math.min(1, raw.confidence)),
            error: raw.error,
            timestamp: new Date().toLocaleTimeString(),
          };
          setLatest(msg);
          setHistory((prev) => [msg, ...prev].slice(0, MAX_HISTORY));
          setPulseKey((k) => k + 1);
        } catch {
          // Malformed JSON: ignore
        }
      });

      clientRef.current = client;
    } catch (e: unknown) {
      setConnState('error');
      setConnError(e instanceof Error ? e.message : 'Connection failed');
    }
  }, [brokerUrl, topic]);

  /* ─── Disconnect ─── */
  const handleDisconnect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.end(true);
      clientRef.current = null;
    }
    setConnState('disconnected');
  }, []);

  /* Cleanup on unmount */
  useEffect(() => {
    return () => {
      if (clientRef.current) {
        clientRef.current.end(true);
        clientRef.current = null;
      }
    };
  }, []);

  const isConnected = connState === 'connected';
  const statusCfg = latest ? STATUS_CONFIG[latest.status] ?? STATUS_CONFIG.WARNING : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto"
    >
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-4">
          <Radio className="w-4 h-4 text-primary-blue" />
          <span className="text-sm text-gray-300">Real-Time Telemetry</span>
        </div>
        <h2 className="text-3xl md:text-4xl font-heading font-bold mb-3">
          <span className="text-gradient-blue">Live Device Monitoring</span>
        </h2>
        <p className="text-gray-400 max-w-xl mx-auto">
          Connect to a real ESP32 device streaming engine diagnostics over MQTT.
          Data updates in real-time with no simulations or predictions.
        </p>
      </div>

      {/* ─── Connection Panel ─── */}
      <div className="glass-card rounded-xl border border-white/10 p-5 mb-6">
        <div className="flex items-center gap-2 mb-4">
          {isConnected ? (
            <Wifi size={18} className="text-primary-green" />
          ) : (
            <WifiOff size={18} className="text-gray-500" />
          )}
          <h3 className="font-semibold text-white text-sm">MQTT Connection</h3>
          <span
            className={`ml-auto text-xs font-mono px-2.5 py-1 rounded-full ${
              connState === 'connected'
                ? 'bg-primary-green/20 text-primary-green'
                : connState === 'connecting'
                  ? 'bg-yellow-500/20 text-yellow-400'
                  : connState === 'error'
                    ? 'bg-primary-red/20 text-primary-red'
                    : 'bg-gray-700 text-gray-400'
            }`}
          >
            {connState.toUpperCase()}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Broker URL (WebSocket)</label>
            <input
              type="text"
              value={brokerUrl}
              onChange={(e) => setBrokerUrl(e.target.value)}
              disabled={isConnected}
              className="w-full bg-dark-1 border border-white/10 rounded-lg px-3 py-2
                         text-sm text-white font-mono placeholder-gray-600
                         disabled:opacity-50 focus:outline-none focus:border-primary-blue"
              placeholder="wss://broker.hivemq.com:8884/mqtt"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Topic</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              disabled={isConnected}
              className="w-full bg-dark-1 border border-white/10 rounded-lg px-3 py-2
                         text-sm text-white font-mono placeholder-gray-600
                         disabled:opacity-50 focus:outline-none focus:border-primary-blue"
              placeholder="car/engine/diagnostics"
            />
          </div>
          <div className="flex items-end">
            {isConnected ? (
              <button
                onClick={handleDisconnect}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                           font-semibold text-sm bg-primary-red/20 text-primary-red
                           border border-primary-red/40 hover:bg-primary-red/30 transition-all"
              >
                <Unplug size={16} />
                Disconnect
              </button>
            ) : (
              <button
                onClick={handleConnect}
                disabled={connState === 'connecting'}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg
                           font-semibold text-sm bg-primary-green/20 text-primary-green
                           border border-primary-green/40 hover:bg-primary-green/30
                           disabled:opacity-50 transition-all"
              >
                <Wifi size={16} />
                {connState === 'connecting' ? 'Connecting...' : 'Connect'}
              </button>
            )}
          </div>
        </div>
        {connError && (
          <p className="text-xs text-primary-red mt-1">Error: {connError}</p>
        )}
      </div>

      {/* ─── Live Status Display ─── */}
      <AnimatePresence mode="wait">
        {!isConnected && !latest && (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass-card rounded-xl border border-white/10 p-12 text-center"
          >
            <WifiOff size={48} className="text-gray-600 mx-auto mb-4" />
            <p className="text-gray-500 text-lg font-semibold">Waiting for device...</p>
            <p className="text-gray-600 text-sm mt-1">
              Connect to an MQTT broker and subscribe to a topic to see live data.
            </p>
          </motion.div>
        )}

        {(isConnected || latest) && (
          <motion.div
            key="live"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            {/* Current status card */}
            {latest && statusCfg ? (
              <motion.div
                key={pulseKey}
                initial={{ scale: 1 }}
                animate={{ scale: [1, 1.005, 1] }}
                transition={{ duration: 0.3 }}
                className={`rounded-xl border-2 p-6 mb-6 ${statusCfg.bg} ${statusCfg.border}`}
              >
                <div className="flex items-center justify-between mb-5">
                  <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">
                      Device Status (Live)
                    </p>
                    <div className="flex items-center gap-3">
                      {statusCfg.icon}
                      <h3 className={`text-3xl font-heading font-bold ${statusCfg.color}`}>
                        {statusCfg.label}
                      </h3>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-500">Last update</p>
                    <p className="text-sm text-white font-mono">{latest.timestamp}</p>
                  </div>
                </div>

                {/* Metrics row */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <p className="text-xs text-gray-400 mb-1 uppercase">Status</p>
                    <p className={`text-xl font-bold ${statusCfg.color}`}>{latest.status}</p>
                  </div>
                  <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <p className="text-xs text-gray-400 mb-1 uppercase">Fault</p>
                    <p className="text-xl font-bold text-white">
                      {latest.fault === 'None' ? '---' : latest.fault.replace(/_/g, ' ')}
                    </p>
                  </div>
                  <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <div className="flex items-center gap-1 mb-1">
                      <Activity size={12} className="text-gray-400" />
                      <p className="text-xs text-gray-400 uppercase">Confidence</p>
                    </div>
                    <p className="text-xl font-bold font-mono text-white">
                      {(latest.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <p className="text-xs text-gray-400 mb-1 uppercase">Anomaly Score</p>
                    <p className="text-xl font-bold font-mono text-white">
                      {latest.error.toFixed(4)}
                    </p>
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="glass-card rounded-xl border border-white/10 p-12 text-center mb-6">
                <Radio size={36} className="text-primary-blue mx-auto mb-3 animate-pulse" />
                <p className="text-gray-400 font-semibold">Connected. Waiting for first message...</p>
                <p className="text-xs text-gray-600 mt-1">
                  Subscribed to <span className="font-mono text-primary-blue">{topic}</span>
                </p>
              </div>
            )}

            {/* Message history */}
            {history.length > 0 && (
              <div className="glass-card rounded-xl border border-white/10 p-5">
                <div className="flex items-center gap-2 mb-3">
                  <Clock size={14} className="text-gray-400" />
                  <p className="text-xs text-gray-400 font-semibold uppercase">
                    Recent Messages ({history.length})
                  </p>
                </div>
                <div className="max-h-64 overflow-y-auto space-y-1.5 pr-1">
                  {history.map((msg, i) => {
                    const cfg = STATUS_CONFIG[msg.status] ?? STATUS_CONFIG.WARNING;
                    return (
                      <div
                        key={`${msg.timestamp}-${i}`}
                        className="flex items-center gap-3 text-xs font-mono py-1.5 px-3
                                   rounded-lg bg-dark-1/60 border border-white/5"
                      >
                        <span className="text-gray-500 w-20 shrink-0">{msg.timestamp}</span>
                        <span className={`font-semibold w-20 shrink-0 ${cfg.color}`}>
                          {msg.status}
                        </span>
                        <span className="text-gray-400 w-32 shrink-0 truncate">
                          {msg.fault === 'None' ? '---' : msg.fault}
                        </span>
                        <span className="text-gray-300">
                          conf={msg.confidence.toFixed(2)} err={msg.error.toFixed(4)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer */}
      <p className="text-xs text-gray-600 mt-6 text-center italic">
        All data shown is received directly from the connected device. No
        simulations, predictions, or synthetic values are generated.
      </p>
    </motion.div>
  );
};

export default LiveDeviceMonitor;
