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

// ─────────────────────────────────────────────────────────────────────────────
// PLACEHOLDER STATES — Replace with real WebSocket data
// ─────────────────────────────────────────────────────────────────────────────
type NodeQuality = "good" | "fair" | "poor"

interface NodeQualityMap {
  [key: string]: NodeQuality
}

const MOCK_NODE_QUALITIES: NodeQualityMap = {
  AF3: "good",
  F7: "good",
  F3: "good",
  FC5: "fair",
  T7: "good",
  P7: "good",
  O1: "fair",
  O2: "good",
  P8: "poor",
  T8: "fair",
  FC6: "good",
  F4: "good",
  F8: "poor",
  AF4: "good",
}

const MOCK_MENTAL_STATES = {
  stress: 42,
  fatigue: 28,
  focus: 76,
  relaxation: 61,
}

const MOCK_BANDPOWER_DATA = [
  { band: "Theta", power: 42, color: "#00A3FF" },
  { band: "Alpha", power: 58, color: "#10B981" },
  { band: "Beta", power: 71, color: "#FF903F" },
  { band: "Gamma", power: 35, color: "#8B5CF6" },
]

const MOCK_AI_STATE = {
  model: "EEGNet-v1.2 (General)",
  confidenceThreshold: 75,
  calibrationReady: true,
}
// ─────────────────────────────────────────────────────────────────────────────

// EEG electrode positions (normalized 0-1 within a unit circle head)
const ELECTRODE_POSITIONS: Record<string, { x: number; y: number }> = {
  AF3: { x: 0.38, y: 0.22 },
  AF4: { x: 0.62, y: 0.22 },
  F7:  { x: 0.22, y: 0.33 },
  F3:  { x: 0.38, y: 0.33 },
  F4:  { x: 0.62, y: 0.33 },
  F8:  { x: 0.78, y: 0.33 },
  FC5: { x: 0.28, y: 0.44 },
  FC6: { x: 0.72, y: 0.44 },
  T7:  { x: 0.14, y: 0.55 },
  T8:  { x: 0.86, y: 0.55 },
  P7:  { x: 0.22, y: 0.67 },
  P8:  { x: 0.78, y: 0.67 },
  O1:  { x: 0.38, y: 0.78 },
  O2:  { x: 0.62, y: 0.78 },
}

const QUALITY_COLORS: Record<NodeQuality, string> = {
  good: "#10B981",
  fair: "#FF903F",
  poor: "#EF4444",
}

const QUALITY_BG: Record<NodeQuality, string> = {
  good: "rgba(16,185,129,0.18)",
  fair: "rgba(255,144,63,0.18)",
  poor: "rgba(239,68,68,0.18)",
}

function HeadMap({ nodeQualities }: { nodeQualities: NodeQualityMap }) {
  const size = 260

  return (
    <div className="flex flex-col items-center gap-4">
      <svg
        width={size}
        height={size}
        viewBox="0 0 260 260"
        aria-label="EEG electrode contact quality map"
      >
        {/* Head outline */}
        <ellipse
          cx="130"
          cy="133"
          rx="104"
          ry="115"
          fill="none"
          stroke="#E2EBF3"
          strokeWidth="2"
        />
        {/* Nose */}
        <path
          d="M 118,16 Q 130,5 142,16"
          fill="none"
          stroke="#E2EBF3"
          strokeWidth="2"
          strokeLinecap="round"
        />
        {/* Left ear */}
        <path
          d="M 26,133 Q 18,133 18,142 Q 18,155 26,155"
          fill="none"
          stroke="#E2EBF3"
          strokeWidth="2"
          strokeLinecap="round"
        />
        {/* Right ear */}
        <path
          d="M 234,133 Q 242,133 242,142 Q 242,155 234,155"
          fill="none"
          stroke="#E2EBF3"
          strokeWidth="2"
          strokeLinecap="round"
        />
        {/* Crosshair lines */}
        <line x1="130" y1="18" x2="130" y2="248" stroke="#E2EBF3" strokeWidth="1" strokeDasharray="4 4" />
        <line x1="26" y1="133" x2="234" y2="133" stroke="#E2EBF3" strokeWidth="1" strokeDasharray="4 4" />
        {/* Circle for Cz */}
        <circle cx="130" cy="133" r="3" fill="#CBD5E1" />

        {/* Electrodes */}
        {Object.entries(ELECTRODE_POSITIONS).map(([name, pos]) => {
          const cx = pos.x * 260
          const cy = pos.y * 260
          const quality = nodeQualities[name] ?? "good"
          const color = QUALITY_COLORS[quality]
          const bg = QUALITY_BG[quality]

          return (
            <g key={name}>
              {/* Glow ring */}
              <circle cx={cx} cy={cy} r={14} fill={bg} />
              {/* Electrode dot */}
              <circle cx={cx} cy={cy} r={9} fill={color} />
              {/* Inner ring */}
              <circle cx={cx} cy={cy} r={9} fill="none" stroke="white" strokeWidth="1.5" />
              {/* Label */}
              <text
                x={cx}
                y={cy + 22}
                textAnchor="middle"
                fontSize="8"
                fontWeight="600"
                fill="#64748B"
                fontFamily="var(--font-inter)"
              >
                {name}
              </text>
            </g>
          )
        })}
      </svg>

      {/* Legend */}
      <div className="flex items-center gap-4">
        {(["good", "fair", "poor"] as NodeQuality[]).map((q) => (
          <div key={q} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: QUALITY_COLORS[q] }} />
            <span className="text-xs text-slate-500 capitalize" style={{ fontFamily: "var(--font-inter)" }}>
              {q}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// Generate a random EEG-like signal point
function generateEEGPoint(prev: number): number {
  const delta = (Math.random() - 0.5) * 35
  return Math.max(-100, Math.min(100, prev + delta))
}

const EEG_CHANNELS = ["AF3", "F3", "T7", "O1", "O2", "T8", "F4", "AF4"]

type EEGDataPoint = Record<string, number> & { t: number }

function EEGOscilloscope() {
  const [data, setData] = useState<EEGDataPoint[]>(() => {
    const initial: EEGDataPoint[] = []
    const vals: Record<string, number> = {}
    EEG_CHANNELS.forEach((ch) => (vals[ch] = 0))
    for (let i = 0; i < 80; i++) {
      const point: EEGDataPoint = { t: i }
      EEG_CHANNELS.forEach((ch) => {
        vals[ch] = generateEEGPoint(vals[ch])
        point[ch] = vals[ch]
      })
      initial.push(point)
    }
    return initial
  })

  const lastVals = useRef<Record<string, number>>({})

  useEffect(() => {
    const interval = setInterval(() => {
      setData((prev) => {
        const last = prev[prev.length - 1] ?? {}
        const newPoint: EEGDataPoint = { t: (last.t ?? 0) + 1 }
        EEG_CHANNELS.forEach((ch) => {
          const prevVal = lastVals.current[ch] ?? 0
          const next = generateEEGPoint(prevVal)
          newPoint[ch] = next
          lastVals.current[ch] = next
        })
        const updated = [...prev.slice(-79), newPoint]
        return updated
      })
    }, 50)
    return () => clearInterval(interval)
  }, [])

  const channelColors = [
    "#00A3FF", "#10B981", "#FF903F", "#8B5CF6",
    "#EF4444", "#F59E0B", "#06B6D4", "#EC4899"
  ]

  return (
    <div className="flex flex-col gap-0.5">
      {EEG_CHANNELS.map((ch, idx) => (
        <div key={ch} className="flex items-center gap-2 h-8">
          <span
            className="text-xs font-semibold w-8 flex-shrink-0 text-right"
            style={{ color: channelColors[idx], fontFamily: "var(--font-inter)" }}
          >
            {ch}
          </span>
          <div className="flex-1 h-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <YAxis domain={[-110, 110]} hide />
                <Line
                  type="monotone"
                  dataKey={ch}
                  stroke={channelColors[idx]}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ))}
    </div>
  )
}

function CircularGauge({
  label,
  value,
  color,
}: {
  label: string
  value: number
  color: string
}) {
  const radius = 30
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference - (value / 100) * circumference

  return (
    <div className="flex flex-col items-center gap-2 py-3">
      <div className="relative w-20 h-20 flex items-center justify-center">
        <svg width="80" height="80" viewBox="0 0 80 80" className="-rotate-90">
          <circle cx="40" cy="40" r={radius} fill="none" stroke="#F1F5F9" strokeWidth="6" />
          <circle
            cx="40"
            cy="40"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            className="transition-all duration-700"
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span
            className="text-sm font-bold"
            style={{ color, fontFamily: "var(--font-poppins)" }}
          >
            {value}
          </span>
          <span className="text-xs text-slate-400" style={{ fontFamily: "var(--font-inter)" }}>
            /100
          </span>
        </div>
      </div>
      <span
        className="text-xs font-semibold text-[#196484] text-center"
        style={{ fontFamily: "var(--font-poppins)" }}
      >
        {label}
      </span>
    </div>
  )
}

function FrequencyBandpower() {
  return (
    <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
      <CardHeader className="pb-2.5">
        <CardTitle
          className="text-sm font-semibold text-foreground"
          style={{ fontFamily: "var(--font-poppins)" }}
        >
          Frequency Bandpower
        </CardTitle>
        <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
          Current power distribution across EEG bands
        </p>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={MOCK_BANDPOWER_DATA} barSize={32}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
            <XAxis
              dataKey="band"
              tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "var(--font-inter)" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "var(--font-inter)" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#fff",
                border: "1px solid #E2EBF3",
                borderRadius: "8px",
                fontSize: "11px",
                fontFamily: "var(--font-inter)",
              }}
              formatter={(v: number) => [`${v} µV²`, "Power"]}
            />
            <Bar
              dataKey="power"
              fill="#00A3FF"
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

function AIEngineControls() {
  const [model, setModel] = useState("EEGNet-v1.2")
  const [threshold, setThreshold] = useState(75)
  const [isCalibrating, setIsCalibrating] = useState(false)

  const handleCalibrate = () => {
    setIsCalibrating(true)
    setTimeout(() => setIsCalibrating(false), 3000)
  }

  return (
    <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
      <CardHeader className="pb-2.5">
        <CardTitle
          className="text-sm font-semibold text-foreground flex items-center gap-2"
          style={{ fontFamily: "var(--font-poppins)" }}
        >
          <Settings className="w-4 h-4 text-[#00A3FF]" />
          AI Engine Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Model Selector */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-600 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>
            AI Model
          </label>
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger className="h-9 text-xs rounded-lg border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="EEGNet-v1.2">EEGNet-v1.2 (General)</SelectItem>
              <SelectItem value="EEGNet-v2.0">EEGNet-v2.0 (Personalized)</SelectItem>
              <SelectItem value="DeepConvNet">DeepConvNet (Experimental)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Confidence Threshold Slider */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-slate-600 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>
              Confidence Threshold
            </label>
            <span className="text-xs font-semibold text-[#00A3FF]" style={{ fontFamily: "var(--font-poppins)" }}>
              {threshold}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="w-full h-1.5 rounded-full accent-[#00A3FF]"
          />
          <p className="text-xs text-slate-400" style={{ fontFamily: "var(--font-inter)" }}>
            Ignore predictions below this confidence level
          </p>
        </div>

        {/* Calibrate Button */}
        <Button
          className="w-full gap-2 h-9 text-xs font-semibold rounded-lg transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg bg-[#00A3FF] hover:bg-[#0089D9] text-white"
          onClick={handleCalibrate}
          disabled={isCalibrating}
        >
          <Zap className="w-3.5 h-3.5" />
          {isCalibrating ? "Calibrating (60s)..." : "Calibrate Baseline"}
        </Button>
      </CardContent>
    </Card>
  )
}

function HardwareTelemetry() {
  return (
    <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
      <CardHeader className="pb-2.5">
        <CardTitle
          className="text-sm font-semibold text-foreground"
          style={{ fontFamily: "var(--font-poppins)" }}
        >
          Hardware Telemetry
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {/* Battery Level */}
        <div className="flex items-center justify-between py-2 px-2.5 rounded-lg bg-slate-50 border border-border">
          <div className="flex items-center gap-2">
            <Battery className="w-4 h-4 text-orange-500" />
            <span className="text-xs font-semibold text-slate-600" style={{ fontFamily: "var(--font-inter)" }}>
              Battery
            </span>
          </div>
          <span className="text-xs font-extrabold text-orange-600" style={{ fontFamily: "var(--font-poppins)" }}>
            85%
          </span>
        </div>

        {/* Wireless Signal */}
        <div className="flex items-center justify-between py-2 px-2.5 rounded-lg bg-slate-50 border border-border">
          <div className="flex items-center gap-2">
            <Wifi className="w-4 h-4 text-emerald-500" />
            <span className="text-xs font-semibold text-slate-600" style={{ fontFamily: "var(--font-inter)" }}>
              Signal
            </span>
          </div>
          <span className="text-xs font-semibold text-emerald-600" style={{ fontFamily: "var(--font-poppins)" }}>
            2.4GHz Strong
          </span>
        </div>

        {/* Packet Loss */}
        <div className="flex items-center justify-between py-2 px-2.5 rounded-lg bg-slate-50 border border-border">
          <div className="flex items-center gap-2">
            <TrendingDown className="w-4 h-4 text-blue-500" />
            <span className="text-xs font-semibold text-slate-600" style={{ fontFamily: "var(--font-inter)" }}>
              Packet Loss
            </span>
          </div>
          <span className="text-xs font-semibold text-blue-600" style={{ fontFamily: "var(--font-poppins)" }}>
            0.02%
          </span>
        </div>
      </CardContent>
    </Card>
  )
}

export default function MonitorPage() {
  // ─────────────────────────────────────────────────────────────────────────
  // PLACEHOLDER STATES — Replace with real WebSocket data
  // ─────────────────────────────────────────────────────────────────────────
  const [nodeQualities, setNodeQualities] = useState<NodeQualityMap>(MOCK_NODE_QUALITIES)
  const [mentalStates, setMentalStates] = useState(MOCK_MENTAL_STATES)
  // ─────────────────────────────────────────────────────────────────────────

  const qualityCounts = Object.values(nodeQualities).reduce(
    (acc, q) => { acc[q]++; return acc },
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
      {/* Page Header */}
      <div className="mb-6">
        <h1
          className="text-xl font-bold text-[#196484] text-balance"
          style={{ fontFamily: "var(--font-poppins)" }}
        >
          Monitor &amp; Control
        </h1>
        <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
          Hardware status, signal quality, and mental state overview
        </p>
      </div>

      {/* Bento Grid */}
      <div className="grid grid-cols-3 md:grid-cols-2 gap-4">

        {/* Contact Quality Map */}
        <Card className="rounded-2xl shadow-sm border-border bg-card col-span-1 row-span-2 md:col-span-1 md:row-span-1 transition-all duration-300">
          <CardHeader className="pb-2">
            <CardTitle
              className="text-sm font-semibold text-foreground"
              style={{ fontFamily: "var(--font-poppins)" }}
            >
              Contact Quality Map
            </CardTitle>
            <div className="flex gap-2 mt-1 flex-wrap">
              <Badge className="text-xs bg-emerald-100 text-emerald-700 border-emerald-200 border">
                {qualityCounts.good} Good
              </Badge>
              <Badge className="text-xs bg-orange-100 text-orange-700 border-orange-200 border">
                {qualityCounts.fair} Fair
              </Badge>
              <Badge className="text-xs bg-red-100 text-red-700 border-red-200 border">
                {qualityCounts.poor} Poor
              </Badge>
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
              <CardTitle
                className="text-sm font-semibold text-foreground"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                Live EEG Oscilloscope
              </CardTitle>
              <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-red-50 border border-red-200">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-red-500" />
                </span>
                <span className="text-xs font-semibold text-red-600" style={{ fontFamily: "var(--font-inter)" }}>
                  Recording
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
              Raw microvolt amplitude · 8 channels · 250 Hz
            </p>
          </CardHeader>
          <CardContent>
            <EEGOscilloscope />
          </CardContent>
        </Card>

        {/* Mental State Gauges */}
        <Card className="rounded-2xl shadow-sm border-border bg-card col-span-2 md:col-span-1 transition-all duration-300">
          <CardHeader className="pb-2.5">
            <CardTitle
              className="text-sm font-semibold text-foreground"
              style={{ fontFamily: "var(--font-poppins)" }}
            >
              Mental State Dashboard
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 md:grid-cols-2 gap-1 divide-x divide-y md:divide-y divide-border">
              {mentalGauges.map((g) => (
                <CircularGauge
                  key={g.key}
                  label={g.label}
                  value={mentalStates[g.key]}
                  color={g.color}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Frequency Bandpower */}
        <FrequencyBandpower />

        {/* Hardware Telemetry */}
        <HardwareTelemetry />

        {/* AI Engine Controls */}
        <Card className="rounded-2xl shadow-sm border-border bg-card col-span-2 md:col-span-1 transition-all duration-300">
          <CardHeader className="pb-2.5">
            <CardTitle
              className="text-sm font-semibold text-foreground"
              style={{ fontFamily: "var(--font-poppins)" }}
            >
              AI Engine Controls
            </CardTitle>
          </CardHeader>
          <CardContent>
            <AIEngineControls />
          </CardContent>
        </Card>

      </div>
    </div>
  )
}
