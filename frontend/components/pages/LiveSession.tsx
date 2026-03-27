"use client"

import { useState, useEffect } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Zap,
  Square,
  Volume2,
  CheckCircle2,
  Loader2,
  BrainCircuit,
  Waves,
  Layers,
  Sparkles,
  Radio,
  AlertCircle,
  Delete,
  Trash2,
  Filter,
  Mic,
  Settings2,
  Speaker,
  VolumeX,
} from "lucide-react"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

// ─────────────────────────────────────────────────────────────────────────────
// PLACEHOLDER STATE — swap these with real WebSocket data
// ─────────────────────────────────────────────────────────────────────────────
type IntentState = "idle" | "spike_detected" | "decoding" | "done"

const MOCK_RAW_DECODED_WORD = "MAKAN"
const MOCK_REFINED_SENTENCE = "Saya ingin makan sekarang."
const MOCK_AI_CONFIDENCE = 94.5
// ─────────────────────────────────────────────────────────────────────────────

const pipelineSteps = [
  { id: 0, label: "EEG Acquisition", icon: Waves },
  { id: 1, label: "Filtering & Extraction", icon: Filter },
  { id: 2, label: "EEGNet Decoding", icon: BrainCircuit },
  { id: 3, label: "Word Assembly", icon: Layers },
  { id: 4, label: "LLM Refining", icon: Sparkles },
]

function mapIntentStateToStep(state: IntentState): number {
  if (state === "idle") return -1
  if (state === "spike_detected") return 1
  if (state === "decoding") return 2
  if (state === "done") return 4
  return -1
}

function ConnectionStatusBadge() {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-50 border border-emerald-200">
      <span className="relative flex h-2.5 w-2.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
      </span>
      <span className="text-xs font-semibold text-emerald-700" style={{ fontFamily: "var(--font-inter)" }}>
        Connected to BCI Engine
      </span>
    </div>
  )
}

function PipelineStepper({ activeStep }: { activeStep: number }) {
  return (
    <div className="flex items-center gap-1.5 w-full">
      {pipelineSteps.map((step, idx) => {
        const Icon = step.icon
        const isActive = activeStep === idx
        const isDone = activeStep > idx
        const isLast = idx === pipelineSteps.length - 1

        return (
          <div key={step.id} className="flex items-center gap-1.5 flex-1">
            <div
              className={`flex flex-col items-center gap-1 flex-1 px-2.5 py-2.5 rounded-lg border transition-all duration-300 ease-in-out ${
                isActive
                  ? "border-[#00A3FF] bg-blue-50 shadow-sm"
                  : isDone
                  ? "border-emerald-200 bg-emerald-50"
                  : "border-border bg-white"
              }`}
            >
              <div
                className={`w-7 h-7 rounded-full flex items-center justify-center transition-all duration-300 ${
                  isActive
                    ? "bg-[#00A3FF] text-white"
                    : isDone
                    ? "bg-emerald-500 text-white"
                    : "bg-slate-100 text-slate-400"
                }`}
              >
                {isDone ? (
                  <CheckCircle2 className="w-3.5 h-3.5" />
                ) : isActive ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Icon className="w-3.5 h-3.5" />
                )}
              </div>
              <span
                className={`text-xs font-semibold text-center leading-tight transition-colors duration-300 ${
                  isActive
                    ? "text-[#00A3FF]"
                    : isDone
                    ? "text-emerald-600"
                    : "text-slate-400"
                }`}
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                {step.label}
              </span>
            </div>
            {!isLast && (
              <div
                className={`h-0.5 w-4 flex-shrink-0 rounded-full transition-all duration-300 ${
                  isDone || isActive ? "bg-[#00A3FF]" : "bg-slate-200"
                }`}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

function SessionMetadataControls() {
  const [subject, setSubject] = useState("subject-01")
  const [isTTSEnabled, setIsTTSEnabled] = useState(true)

  return (
    <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
      <CardHeader className="pb-3">
        <CardTitle
          className="text-sm font-semibold text-foreground"
          style={{ fontFamily: "var(--font-poppins)" }}
        >
          Session Metadata
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Subject Profile Selector */}
        <div className="space-y-1">
          <label className="text-xs font-semibold text-slate-600 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>
            Subject
          </label>
          <Select value={subject} onValueChange={setSubject}>
            <SelectTrigger className="h-9 text-xs rounded-lg border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="subject-01">Subject-01 (Andiar)</SelectItem>
              <SelectItem value="subject-02">Subject-02 (Anonim)</SelectItem>
              <SelectItem value="subject-03">Subject-03 (Test)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* TTS Feedback Toggle */}
        <div className="flex items-center justify-between py-2 px-2.5 rounded-lg border border-border bg-slate-50/50">
          <div className="flex items-center gap-2">
            {isTTSEnabled ? (
              <Speaker className="w-4 h-4 text-[#00A3FF]" />
            ) : (
              <VolumeX className="w-4 h-4 text-slate-400" />
            )}
            <span className="text-xs font-semibold text-slate-600" style={{ fontFamily: "var(--font-inter)" }}>
              Audio Feedback
            </span>
          </div>
          <Switch
            checked={isTTSEnabled}
            onCheckedChange={setIsTTSEnabled}
          />
        </div>

        {/* Status Indicator */}
        <div className="flex items-center gap-2 p-2 rounded-lg bg-emerald-50 border border-emerald-200">
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
          </span>
          <span className="text-xs font-semibold text-emerald-700" style={{ fontFamily: "var(--font-inter)" }}>
            Session Ready
          </span>
        </div>
      </CardContent>
    </Card>
  )
}

function TTSWaveAnimation() {
  return (
    <span className="flex items-end gap-0.5 h-4 ml-2" aria-label="Text-to-speech playing">
      {[1, 2, 3, 4, 3].map((h, i) => (
        <span
          key={i}
          className="w-0.5 rounded-full bg-[#00A3FF] animate-tts-wave"
          style={{
            height: `${h * 3}px`,
            animationDelay: `${i * 0.1}s`,
          }}
        />
      ))}
    </span>
  )
}

function ConfidenceRing({ value }: { value: number }) {
  const radius = 30
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference - (value / 100) * circumference

  return (
    <div className="relative flex items-center justify-center w-20 h-20">
      <svg width="80" height="80" viewBox="0 0 80 80" className="-rotate-90">
        <circle
          cx="40"
          cy="40"
          r={radius}
          fill="none"
          stroke="#E2EBF3"
          strokeWidth="6"
        />
        <circle
          cx="40"
          cy="40"
          r={radius}
          fill="none"
          stroke="#00A3FF"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          className="transition-all duration-700"
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span
          className="text-xl font-extrabold text-[#196484] leading-none"
          style={{ fontFamily: "var(--font-poppins)" }}
        >
          {Math.round(value)}%
        </span>
      </div>
    </div>
  )
}

function ResultDisplay({
  intentState,
  rawDecodedWord,
  refinedSentence,
  aiConfidence,
  onDeleteLastWord,
  onClearSentence,
}: {
  intentState: IntentState
  rawDecodedWord: string
  refinedSentence: string
  aiConfidence: number
  onDeleteLastWord: () => void
  onClearSentence: () => void
}) {
  const isVisible = intentState === "done"

  return (
    <Card className="rounded-2xl shadow-md border-border bg-card col-span-2 transition-all duration-300">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle
            className="text-sm font-semibold text-foreground"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            Inference Result
          </CardTitle>
          {isVisible && (
            <Badge className="bg-emerald-100 text-emerald-700 border-emerald-200 border text-xs">
              Output Ready
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {!isVisible ? (
          <div className="flex flex-col items-center justify-center py-8 gap-3">
            <div
              className="w-12 h-12 rounded-2xl flex items-center justify-center"
              style={{ backgroundColor: "#EFF6FF" }}
            >
              <BrainCircuit className="w-6 h-6 text-slate-300" />
            </div>
            <p className="text-slate-400 text-xs" style={{ fontFamily: "var(--font-inter)" }}>
              Results will appear after inference completes.
            </p>
          </div>
        ) : (
          <>
            {/* Result displays — horizontal layout */}
            <div className="grid grid-cols-3 gap-4 items-start">
              {/* Raw decoded word */}
              <div className="flex flex-col gap-1">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>
                  Raw Decoded
                </span>
                <span
                  className="text-4xl font-extrabold tracking-tight"
                  style={{
                    fontFamily: "var(--font-poppins)",
                    color: "#196484",
                  }}
                >
                  {rawDecodedWord}
                </span>
              </div>

              {/* LLM Refined sentence */}
              <div className="col-span-1 flex flex-col gap-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>
                  LLM Refined Output
                </span>
                <div className="px-3 py-2 rounded-lg border border-[#00A3FF]/30 bg-blue-50/60 transition-all duration-300 hover:border-[#00A3FF]/60">
                  <div className="flex items-center gap-1">
                    <p
                      className="text-sm font-semibold text-[#196484]"
                      style={{ fontFamily: "var(--font-poppins)" }}
                    >
                      {refinedSentence}
                    </p>
                    <div className="flex items-center gap-0.5 ml-1 flex-shrink-0">
                      <Volume2 className="w-3 h-3 text-[#00A3FF]" />
                      <TTSWaveAnimation />
                    </div>
                  </div>
                </div>
              </div>

              {/* Confidence */}
              <div className="flex flex-col items-center gap-1">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>
                  Confidence
                </span>
                <ConfidenceRing value={aiConfidence} />
              </div>
            </div>

            {/* Manual Correction Controls */}
            <div className="pt-3 border-t border-border flex gap-2">
              <Button
                size="sm"
                variant="outline"
                className="flex-1 gap-1.5 text-xs transition-all duration-200 hover:scale-[1.02]"
                onClick={onDeleteLastWord}
              >
                <Delete className="w-3.5 h-3.5" />
                Delete Last Word
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="flex-1 gap-1.5 text-xs text-red-600 border-red-200 hover:bg-red-600 hover:text-white transition-colors"
                onClick={onClearSentence}
              >
                <Trash2 className="w-3.5 h-3.5" />
                Clear All
              </Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}

export default function LiveSessionPage() {
  // ─────────────────────────────────────────────────────────────────────────
  // PLACEHOLDER STATES — Replace with real WebSocket data
  // ─────────────────────────────────────────────────────────────────────────
  const [intentState, setIntentState] = useState<IntentState>("idle")
  const [rawDecodedWord, setRawDecodedWord] = useState<string>(MOCK_RAW_DECODED_WORD)
  const [refinedSentence, setRefinedSentence] = useState<string>(MOCK_REFINED_SENTENCE)
  const [aiConfidence, setAiConfidence] = useState<number>(MOCK_AI_CONFIDENCE)
  const [isRunning, setIsRunning] = useState(false)
  // ─────────────────────────────────────────────────────────────────────────

  const activeStep = mapIntentStateToStep(intentState)

  // Simulated inference sequence — replace with WebSocket event handler
  const startInferenceSequence = () => {
    if (isRunning) return
    setIsRunning(true)
    setIntentState("spike_detected")

    const t1 = setTimeout(() => setIntentState("decoding"), 2500)
    const t2 = setTimeout(() => {
      setIntentState("done")
      setIsRunning(false)
    }, 5000)

    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }

  const emergencyStop = () => {
    setIntentState("idle")
    setIsRunning(false)
  }

  const handleDeleteLastWord = () => {
    const words = refinedSentence.trim().split(" ")
    if (words.length > 0) {
      words.pop()
      setRefinedSentence(words.join(" ") || "")
    }
  }

  const handleClearSentence = () => {
    setRefinedSentence("")
    setRawDecodedWord("")
    setIntentState("idle")
  }

  return (
    <div className="p-5 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1
            className="text-xl font-bold text-[#196484] text-balance"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            Live Session
          </h1>
          <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
            Real-time BCI inference dashboard
          </p>
        </div>
        <ConnectionStatusBadge />
      </div>

      {/* Action Controls - Full Width at Top */}
      <Card className="rounded-2xl shadow-sm border-border bg-card mb-4 transition-all duration-300">
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <Button
              className="flex-1 gap-2 h-12 text-sm font-semibold rounded-lg transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg"
              style={{
                background: isRunning
                  ? "#94A3B8"
                  : "linear-gradient(135deg, #00A3FF, #0080CC)",
                color: "#fff",
                border: "none",
              }}
              onClick={startInferenceSequence}
              disabled={isRunning}
              aria-label="Start inference sequence"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Start Inference Sequence
                </>
              )}
            </Button>
            <Button
              variant="outline"
              className="h-12 px-6 gap-2 text-sm font-semibold rounded-lg border-2 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg"
              style={{
                borderColor: "#FF903F",
                color: "#FF903F",
                backgroundColor: "transparent",
              }}
              onClick={emergencyStop}
              aria-label="Emergency stop"
            >
              <Square className="w-4 h-4" />
              Emergency Stop
            </Button>
            <div
              className="px-3 py-2 rounded-lg text-xs text-slate-500 flex items-center gap-2"
              style={{ backgroundColor: "#F8FAFC" }}
            >
              <Radio className="w-3 h-3 text-slate-400 flex-shrink-0" />
              <span style={{ fontFamily: "var(--font-inter)" }}>
                State: <strong className="capitalize text-[#196484]">{intentState.replace(/_/g, " ")}</strong>
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Bento Grid */}
      <div className="grid grid-cols-3 md:grid-cols-2 gap-4">
        {/* Session Metadata & Controls — col 1 */}
        <div className="col-span-1">
          <SessionMetadataControls />
        </div>

        {/* Pipeline Stepper — col 2-3 */}
        <div className="col-span-2 md:col-span-1">
          <Card className="rounded-2xl shadow-sm border-border bg-card h-full transition-all duration-300">
            <CardHeader className="pb-3">
              <CardTitle
                className="text-sm font-semibold text-foreground"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                AI Pipeline
              </CardTitle>
            </CardHeader>
            <CardContent>
              <PipelineStepper activeStep={activeStep} />
            </CardContent>
          </Card>
        </div>

        {/* Result Display — full width */}
        <ResultDisplay
          intentState={intentState}
          rawDecodedWord={rawDecodedWord}
          refinedSentence={refinedSentence}
          aiConfidence={aiConfidence}
          onDeleteLastWord={handleDeleteLastWord}
          onClearSentence={handleClearSentence}
        />
      </div>
    </div>
  )
}
