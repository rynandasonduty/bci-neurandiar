"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { LineChart, Line, ResponsiveContainer, YAxis, BarChart, Bar, XAxis, CartesianGrid, Tooltip } from "recharts"
import { Zap, AlertCircle, Settings, Battery, Wifi, TrendingDown } from "lucide-react"
import { WS_URL } from "@/lib/api"

// ─────────────────────────────────────────────────────────────────────────────
// TYPES & CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
type NodeQuality = "good" | "fair" | "poor"

interface NodeQualityMap {
  [key: string]: NodeQuality
}

// Default state saat belum ada data
const DEFAULT_NODE_QUALITIES: NodeQualityMap = {
  AF3: "poor", F7: "poor", F3: "poor", FC5: "poor", T7: "poor", P7: "poor",
  O1: "poor", O2: "poor", P8: "poor", T8: "poor", FC6: "poor", F4: "poor",
  F8: "poor", AF4: "poor",
}

const DEFAULT_MENTAL_STATES = { stress: 0, fatigue: 0, focus: 0, relaxation: 0 }

const DEFAULT_BANDPOWER = [
  { band: "Theta", power: 0, color: "#00A3FF" },
  { band: "Alpha", power: 0, color: "#10B981" },
  { band: "Beta", power: 0, color: "#FF903F" },
  { band: "Gamma", power: 0, color: "#8B5CF6" },
]

const ELECTRODE_POSITIONS: Record<string, { x: number; y: number }> = {
  AF3: { x: 0.38, y: 0.22 }, AF4: { x: 0.62, y: 0.22 },
  F7:  { x: 0.22, y: 0.33 }, F3:  { x: 0.38, y: 0.33 },
  F4:  { x: 0.62, y: 0.33 }, F8:  { x: 0.78, y: 0.33 },
  FC5: { x: 0.28, y: 0.44 }, FC6: { x: 0.72, y: 0.44 },
  T7:  { x: 0.14, y: 0.55 }, T8:  { x: 0.86, y: 0.55 },
  P7:  { x: 0.22, y: 0.67 }, P8:  { x: 0.78, y: 0.67 },
  O1:  { x: 0.38, y: 0.78 }, O2:  { x: 0.62, y: 0.78 },
}

const QUALITY_COLORS: Record<NodeQuality, string> = {
  good: "#10B981", fair: "#FF903F", poor: "#EF4444",
}
const QUALITY_BG: Record<NodeQuality, string> = {
  good: "rgba(16,185,129,0.18)", fair: "rgba(255,144,63,0.18)", poor: "rgba(239,68,68,0.18)",
}

// Kita pilih 8 channel paling representatif untuk dirender agar grafik tidak terlalu padat
const EEG_CHANNELS = ["AF3", "F3", "T7", "O1", "O2", "T8", "F4", "AF4"]
type EEGDataPoint = Record<string, number> & { t: number }

// ─────────────────────────────────────────────────────────────────────────────
// KOMPONEN VISUALISASI
// ─────────────────────────────────────────────────────────────────────────────
function HeadMap({ nodeQualities }: { nodeQualities: NodeQualityMap }) {
  const size = 260
  return (
    <div className="flex flex-col items-center gap-4">
      <svg width={size} height={size} viewBox="0 0 260 260">
        <ellipse cx="130" cy="133" rx="104" ry="115" fill="none" stroke="#E2EBF3" strokeWidth="2" />
        <path d="M 118,16 Q 130,5 142,16" fill="none" stroke="#E2EBF3" strokeWidth="2" strokeLinecap="round" />
        <path d="M 26,133 Q 18,133 18,142 Q 18,155 26,155" fill="none" stroke="#E2EBF3" strokeWidth="2" strokeLinecap="round" />
        <path d="M 234,133 Q 242,133 242,142 Q 242,155 234,155" fill="none" stroke="#E2EBF3" strokeWidth="2" strokeLinecap="round" />
        <line x1="130" y1="18" x2="130" y2="248" stroke="#E2EBF3" strokeWidth="1" strokeDasharray="4 4" />
        <line x1="26" y1="133" x2="234" y2="133" stroke="#E2EBF3" strokeWidth="1" strokeDasharray="4 4" />
        <circle cx="130" cy="133" r="3" fill="#CBD5E1" />

        {Object.entries(ELECTRODE_POSITIONS).map(([name, pos]) => {
          const cx = pos.x * 260
          const cy = pos.y * 260
          const quality = nodeQualities[name] ?? "poor"
          const color = QUALITY_COLORS[quality]
          const bg = QUALITY_BG[quality]

          return (
            <g key={name} className="transition-all duration-300">
              <circle cx={cx} cy={cy} r={14} fill={bg} />
              <circle cx={cx} cy={cy} r={9} fill={color} />
              <circle cx={cx} cy={cy} r={9} fill="none" stroke="white" strokeWidth="1.5" />
              <text x={cx} y={cy + 22} textAnchor="middle" fontSize="8" fontWeight="600" fill="#64748B" fontFamily="var(--font-inter)">
                {name}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="flex items-center gap-4">
        {(["good", "fair", "poor"] as NodeQuality[]).map((q) => (
          <div key={q} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: QUALITY_COLORS[q] }} />
            <span className="text-xs text-slate-500 capitalize" style={{ fontFamily: "var(--font-inter)" }}>{q}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function EEGOscilloscope({ data }: { data: EEGDataPoint[] }) {
  const channelColors = ["#00A3FF", "#10B981", "#FF903F", "#8B5CF6", "#EF4444", "#F59E0B", "#06B6D4", "#EC4899"]

  return (
    <div className="flex flex-col gap-0.5">
      {EEG_CHANNELS.map((ch, idx) => (
        <div key={ch} className="flex items-center gap-2 h-8">
          <span className="text-xs font-semibold w-8 flex-shrink-0 text-right" style={{ color: channelColors[idx], fontFamily: "var(--font-inter)" }}>
            {ch}
          </span>
          <div className="flex-1 h-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <YAxis domain={[-110, 110]} hide />
                <Line type="monotone" dataKey={ch} stroke={channelColors[idx]} strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ))}
    </div>
  )
}

function CircularGauge({ label, value, color }: { label: string, value: number, color: string }) {
  const radius = 30
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference - (value / 100) * circumference

  return (
    <div className="flex flex-col items-center gap-2 py-3">
      <div className="relative w-20 h-20 flex items-center justify-center">
        <svg width="80" height="80" viewBox="0 0 80 80" className="-rotate-90">
          <circle cx="40" cy="40" r={radius} fill="none" stroke="#F1F5F9" strokeWidth="6" />
          <circle cx="40" cy="40" r={radius} fill="none" stroke={color} strokeWidth="6" strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={dashOffset} className="transition-all duration-300" />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="text-sm font-bold" style={{ color, fontFamily: "var(--font-poppins)" }}>{value}</span>
          <span className="text-xs text-slate-400" style={{ fontFamily: "var(--font-inter)" }}>/100</span>
        </div>
      </div>
      <span className="text-xs font-semibold text-[#196484] text-center" style={{ fontFamily: "var(--font-poppins)" }}>{label}</span>
    </div>
  )
}

function FrequencyBandpower({ data }: { data: any[] }) {
  return (
    <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
      <CardHeader className="pb-2.5">
        <CardTitle className="text-sm font-semibold text-foreground" style={{ fontFamily: "var(--font-poppins)" }}>Frequency Bandpower</CardTitle>
        <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>Current power distribution across EEG bands</p>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={data} barSize={32}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis dataKey="band" tick={{ fontSize: 10, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ borderRadius: "8px", fontSize: "11px" }} formatter={(v: number) => [`${v} µV²`, "Power"]} />
            <Bar dataKey="power" fill="#00A3FF" radius={[4, 4, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

// (Sisa komponen AIEngineControls & HardwareTelemetry dibiarkan statis untuk UI Demo)
function AIEngineControls() {
  const [model, setModel] = useState("EEGNet-v1.2")
  const [threshold, setThreshold] = useState(75)
  const [isCalibrating, setIsCalibrating] = useState(false)
  const handleCalibrate = () => { setIsCalibrating(true); setTimeout(() => setIsCalibrating(false), 3000) }

  return (
    <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
      <CardHeader className="pb-2.5">
        <CardTitle className="text-sm font-semibold text-foreground flex items-center gap-2" style={{ fontFamily: "var(--font-poppins)" }}>
          <Settings className="w-4 h-4 text-[#00A3FF]" /> AI Engine Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-600 uppercase tracking-widest">AI Model</label>
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger className="h-9 text-xs rounded-lg border-border"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="EEGNet-v1.2">EEGNet-v1.2 (General)</SelectItem>
              <SelectItem value="EEGNet-v2.0">EEGNet-v2.0 (Personalized)</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-slate-600 uppercase tracking-widest">Confidence Threshold</label>
            <span className="text-xs font-semibold text-[#00A3FF]">{threshold}%</span>
          </div>
          <input type="range" min="0" max="100" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-full h-1.5 rounded-full accent-[#00A3FF]" />
        </div>
        <Button className="w-full gap-2 h-9 text-xs font-semibold rounded-lg bg-[#00A3FF] hover:bg-[#0089D9] text-white" onClick={handleCalibrate} disabled={isCalibrating}>
          <Zap className="w-3.5 h-3.5" /> {isCalibrating ? "Calibrating..." : "Calibrate Baseline"}
        </Button>
      </CardContent>
    </Card>
  )
}

function HardwareTelemetry() {
  return (
    <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
      <CardHeader className="pb-2.5">
        <CardTitle className="text-sm font-semibold text-foreground" style={{ fontFamily: "var(--font-poppins)" }}>Hardware Telemetry</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-center justify-between py-2 px-2.5 rounded-lg bg-slate-50 border border-border">
          <div className="flex items-center gap-2"><Battery className="w-4 h-4 text-orange-500" /><span className="text-xs font-semibold">Battery</span></div>
          <span className="text-xs font-extrabold text-orange-600">85%</span>
        </div>
        <div className="flex items-center justify-between py-2 px-2.5 rounded-lg bg-slate-50 border border-border">
          <div className="flex items-center gap-2"><Wifi className="w-4 h-4 text-emerald-500" /><span className="text-xs font-semibold">Signal</span></div>
          <span className="text-xs font-semibold text-emerald-600">2.4GHz Strong</span>
        </div>
        <div className="flex items-center justify-between py-2 px-2.5 rounded-lg bg-slate-50 border border-border">
          <div className="flex items-center gap-2"><TrendingDown className="w-4 h-4 text-blue-500" /><span className="text-xs font-semibold">Packet Loss</span></div>
          <span className="text-xs font-semibold text-blue-600">0.02%</span>
        </div>
      </CardContent>
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// HALAMAN UTAMA (PENYATU WEBSOCKET)
// ─────────────────────────────────────────────────────────────────────────────
export default function MonitorPage() {
  const [isConnected, setIsConnected] = useState(false)
  const [nodeQualities, setNodeQualities] = useState<NodeQualityMap>(DEFAULT_NODE_QUALITIES)
  const [mentalStates, setMentalStates] = useState(DEFAULT_MENTAL_STATES)
  const [bandpower, setBandpower] = useState(DEFAULT_BANDPOWER)
  
  // Inisialisasi 80 titik kosong agar grafik oskiloskop tidak error di awal
  const [eegData, setEegData] = useState<EEGDataPoint[]>(() => {
    return Array.from({ length: 80 }).map((_, i) => ({
      t: i, AF3: 0, F3: 0, T7: 0, O1: 0, O2: 0, T8: 0, F4: 0, AF4: 0
    }))
  })

  const ws = useRef<WebSocket | null>(null)
  const timeCounter = useRef(80) // Melanjutkan titik waktu ke-80

  useEffect(() => {
    ws.current = new WebSocket(`${WS_URL}/ws/telemetry`)

    ws.current.onopen = () => setIsConnected(true)
    ws.current.onclose = () => setIsConnected(false)

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      // Update Head Map
      if (data.cq) setNodeQualities(data.cq)
      
      // Update Mental State
      if (data.mental_state) {
        setMentalStates({
          stress: data.mental_state.Stress,
          fatigue: data.mental_state.Fatigue,
          focus: data.mental_state.Focus,
          relaxation: data.mental_state.Relaxation
        })
      }

      // Update Bar Chart Frekuensi
      if (data.bandpower) {
        setBandpower([
          { band: "Theta", power: data.bandpower.Theta, color: "#00A3FF" },
          { band: "Alpha", power: data.bandpower.Alpha, color: "#10B981" },
          { band: "Beta", power: data.bandpower.Beta, color: "#FF903F" },
          { band: "Gamma", power: data.bandpower.Gamma, color: "#8B5CF6" },
        ])
      }

      // Update Live EEG Chart (Menggeser array ke kiri)
      if (data.eeg) {
        timeCounter.current += 1
        setEegData((prev) => {
          const newPoint = { t: timeCounter.current, ...data.eeg }
          return [...prev.slice(1), newPoint] // Hapus indeks pertama, masukkan yang baru di akhir
        })
      }
    }

    return () => {
      if (ws.current) ws.current.close()
    }
  }, [])

  const qualityCounts = Object.values(nodeQualities).reduce(
    (acc, q) => { acc[q.toLowerCase() as NodeQuality]++; return acc },
    { good: 0, fair: 0, poor: 0 }
  )

  const mentalGauges = [
    { label: "Stress", key: "stress" as const, color: "#EF4444" },
    { label: "Fatigue", key: "fatigue" as const, color: "#FF903F" },
    { label: "Focus", key: "focus" as const, color: "#00A3FF" },
    { label: "Relaxation", key: "relaxation" as const, color: "#10B981" },
  ]

  return (
    <div className="p-5 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-[#196484] text-balance" style={{ fontFamily: "var(--font-poppins)" }}>
            Monitor &amp; Control
          </h1>
          <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
            Hardware status, signal quality, and mental state overview
          </p>
        </div>

        {/* Indikator Koneksi Stream Telemetri */}
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${isConnected ? 'bg-emerald-50 border-emerald-200' : 'bg-red-50 border-red-200'}`}>
          <span className="relative flex h-2.5 w-2.5">
            {isConnected && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />}
            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
          </span>
          <span className={`text-xs font-semibold ${isConnected ? 'text-emerald-700' : 'text-red-700'}`} style={{ fontFamily: "var(--font-inter)" }}>
            {isConnected ? 'Telemetry Stream Active' : 'Headset Disconnected'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 md:grid-cols-2 gap-4">
        {/* Contact Quality Map */}
        <Card className="rounded-2xl shadow-sm border-border bg-card col-span-1 row-span-2 md:col-span-1 md:row-span-1 transition-all duration-300">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold text-foreground">Contact Quality Map</CardTitle>
            <div className="flex gap-2 mt-1 flex-wrap">
              <Badge className="text-xs bg-emerald-100 text-emerald-700 border-emerald-200">{qualityCounts.good} Good</Badge>
              <Badge className="text-xs bg-orange-100 text-orange-700 border-orange-200">{qualityCounts.fair} Fair</Badge>
              <Badge className="text-xs bg-red-100 text-red-700 border-red-200">{qualityCounts.poor} Poor</Badge>
            </div>
          </CardHeader>
          <CardContent className="flex items-center justify-center py-3">
            <HeadMap nodeQualities={nodeQualities} />
          </CardContent>
        </Card>

        {/* EEG Oscilloscope */}
        <Card className="rounded-2xl shadow-sm border-border bg-card col-span-2 md:col-span-1 transition-all duration-300">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold text-foreground">Live EEG Oscilloscope</CardTitle>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">Raw microvolt amplitude · 8 key channels</p>
          </CardHeader>
          <CardContent>
            <EEGOscilloscope data={eegData} />
          </CardContent>
        </Card>

        {/* Mental State Gauges */}
        <Card className="rounded-2xl shadow-sm border-border bg-card col-span-2 md:col-span-1 transition-all duration-300">
          <CardHeader className="pb-2.5">
            <CardTitle className="text-sm font-semibold text-foreground">Mental State Dashboard</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 md:grid-cols-2 gap-1 divide-x divide-y md:divide-y divide-border">
              {mentalGauges.map((g) => (
                <CircularGauge key={g.key} label={g.label} value={mentalStates[g.key]} color={g.color} />
              ))}
            </div>
          </CardContent>
        </Card>

        <FrequencyBandpower data={bandpower} />
        <HardwareTelemetry />
        <AIEngineControls />
      </div>
    </div>
  )
}