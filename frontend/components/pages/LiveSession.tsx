"use client"

import { useState, useEffect, useRef } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { API_URL, WS_URL } from "@/lib/api"
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
  Delete,
  Trash2,
  Filter,
  Speaker,
  VolumeX,
  Database,
} from "lucide-react"

// 1. Tipe Data untuk Status Intent
type IntentState = "idle" | "spike_detected" | "decoding" | "done"

// 2. Konfigurasi Langkah Pipeline
const pipelineSteps = [
  { id: 0, label: "EEG Acquisition", icon: Waves },
  { id: 1, label: "Filtering & Extraction", icon: Filter },
  { id: 2, label: "EEGNet Decoding", icon: BrainCircuit },
  { id: 3, label: "Word Assembly", icon: Layers },
  { id: 4, label: "LLM Refining", icon: Sparkles },
]

// 3. Komponen Visual Pipeline
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

// 4. Komponen Metadata (Sekarang tersambung ke state utama)
function SessionMetadataControls({
  subject,
  setSubject,
  isTTSEnabled,
  setIsTTSEnabled
}: {
  subject: string;
  setSubject: (val: string) => void;
  isTTSEnabled: boolean;
  setIsTTSEnabled: (val: boolean) => void;
}) {
  return (
    <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold text-foreground" style={{ fontFamily: "var(--font-poppins)" }}>
          Session Metadata
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1">
          <label className="text-xs font-semibold text-slate-600 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>
            Subject
          </label>
          <Select value={subject} onValueChange={setSubject}>
            <SelectTrigger className="h-9 text-xs rounded-lg border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Subject-01">Subject-01 (Andiar)</SelectItem>
              <SelectItem value="Subject-02">Subject-02 (Anonim)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center justify-between py-2 px-2.5 rounded-lg border border-border bg-slate-50/50">
          <div className="flex items-center gap-2">
            {isTTSEnabled ? <Speaker className="w-4 h-4 text-[#00A3FF]" /> : <VolumeX className="w-4 h-4 text-slate-400" />}
            <span className="text-xs font-semibold text-slate-600" style={{ fontFamily: "var(--font-inter)" }}>
              Audio Feedback
            </span>
          </div>
          <Switch checked={isTTSEnabled} onCheckedChange={setIsTTSEnabled} />
        </div>
      </CardContent>
    </Card>
  )
}

function TTSWaveAnimation() {
  return (
    <span className="flex items-end gap-0.5 h-4 ml-2" aria-label="Text-to-speech playing">
      {[1, 2, 3, 4, 3].map((h, i) => (
        <span key={i} className="w-0.5 rounded-full bg-[#00A3FF] animate-pulse" style={{ height: `${h * 3}px`, animationDelay: `${i * 0.1}s` }} />
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
        <circle cx="40" cy="40" r={radius} fill="none" stroke="#E2EBF3" strokeWidth="6" />
        <circle cx="40" cy="40" r={radius} fill="none" stroke="#00A3FF" strokeWidth="6" strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={dashOffset} className="transition-all duration-700" />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-xl font-extrabold text-[#196484] leading-none" style={{ fontFamily: "var(--font-poppins)" }}>
          {Math.round(value)}%
        </span>
      </div>
    </div>
  )
}

// 5. Komponen Result (Sekarang memiliki tombol Save to History)
function ResultDisplay({
  intentState,
  rawDecodedWord,
  refinedSentence,
  aiConfidence,
  isSaved,
  onDeleteLastWord,
  onClearSentence,
  onSaveLog
}: {
  intentState: IntentState
  rawDecodedWord: string
  refinedSentence: string
  aiConfidence: number
  isSaved: boolean
  onDeleteLastWord: () => void
  onClearSentence: () => void
  onSaveLog: () => void
}) {
  const isVisible = intentState === "done"

  return (
    <Card className="rounded-2xl shadow-md border-border bg-card col-span-2 transition-all duration-300">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-foreground" style={{ fontFamily: "var(--font-poppins)" }}>
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
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center" style={{ backgroundColor: "#EFF6FF" }}>
              <BrainCircuit className="w-6 h-6 text-slate-300" />
            </div>
            <p className="text-slate-400 text-xs" style={{ fontFamily: "var(--font-inter)" }}>
              Results will appear after inference completes.
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-4 items-start">
              <div className="flex flex-col gap-1">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>Raw Decoded</span>
                <span className="text-4xl font-extrabold tracking-tight text-[#196484]" style={{ fontFamily: "var(--font-poppins)" }}>{rawDecodedWord}</span>
              </div>

              <div className="col-span-1 flex flex-col gap-2">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>LLM Refined Output</span>
                <div className="px-3 py-2 rounded-lg border border-[#00A3FF]/30 bg-blue-50/60 transition-all duration-300 hover:border-[#00A3FF]/60">
                  <div className="flex items-center gap-1">
                    <p className="text-sm font-semibold text-[#196484]" style={{ fontFamily: "var(--font-poppins)" }}>{refinedSentence}</p>
                    <div className="flex items-center gap-0.5 ml-1 flex-shrink-0">
                      <Volume2 className="w-3 h-3 text-[#00A3FF]" />
                      <TTSWaveAnimation />
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex flex-col items-center gap-1">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest" style={{ fontFamily: "var(--font-inter)" }}>Confidence</span>
                <ConfidenceRing value={aiConfidence} />
              </div>
            </div>

            <div className="pt-3 border-t border-border flex gap-2">
              <Button size="sm" variant="outline" className="flex-1 gap-1.5 text-xs transition-all duration-200 hover:scale-[1.02]" onClick={onDeleteLastWord}>
                <Delete className="w-3.5 h-3.5" /> Delete Last Word
              </Button>
              <Button size="sm" variant="outline" className="flex-1 gap-1.5 text-xs text-red-600 border-red-200 hover:bg-red-600 hover:text-white transition-colors" onClick={onClearSentence}>
                <Trash2 className="w-3.5 h-3.5" /> Clear All
              </Button>
              <Button size="sm" variant="outline" disabled={isSaved} onClick={onSaveLog} className={`flex-1 gap-1.5 text-xs transition-colors ${isSaved ? 'bg-emerald-500 text-white hover:bg-emerald-600' : 'text-emerald-600 border-emerald-200 hover:bg-emerald-50'}`}>
                <Database className="w-3.5 h-3.5" /> {isSaved ? 'Saved' : 'Save to History'}
              </Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}

// ============================================================================
// MAIN PAGE COMPONENT
// ============================================================================
export default function LiveSessionPage() {
  const [isConnected, setIsConnected] = useState(false)
  const [statusMsg, setStatusMsg] = useState('Standby')
  const [activeStep, setActiveStep] = useState(0) 
  const [intentState, setIntentState] = useState<IntentState>("idle")
  
  const [rawDecodedWord, setRawDecodedWord] = useState<string>("---")
  const [refinedSentence, setRefinedSentence] = useState<string>("Awaiting brain signal...")
  const [aiConfidence, setAiConfidence] = useState<number>(0)
  const [isRunning, setIsRunning] = useState(false)
  
  const [activeSubject, setActiveSubject] = useState("Subject-01") 
  const [isTTSEnabled, setIsTTSEnabled] = useState(true)
  const [isSaved, setIsSaved] = useState(false)
  
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    ws.current = new WebSocket(`${WS_URL}/ws/inference`)

    ws.current.onopen = () => {
      setIsConnected(true)
      setStatusMsg('Connected to BCI Engine')
    }

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.status === 'processing') {
        setStatusMsg(data.message)
        setActiveStep(data.step) 
        setIntentState("decoding")
      } 
      else if (data.status === 'success') {
        setStatusMsg('Decoding Complete!')
        setActiveStep(5) 
        setIntentState("done")
        setIsRunning(false)
        setIsSaved(false) 
        
        setRawDecodedWord(data.decoded_word)
        setAiConfidence(data.confidence)
        setRefinedSentence(data.refined_sentence) 
      }
    }

    ws.current.onclose = () => {
      setIsConnected(false)
      setStatusMsg('Connection Lost')
      setActiveStep(0)
      setIntentState("idle")
      setIsRunning(false)
    }

    return () => {
      if (ws.current) ws.current.close()
    }
  }, [])

  const handleStartInference = () => {
    if (ws.current && isConnected) {
      setIsRunning(true)
      setActiveStep(0)
      setIntentState("spike_detected")
      setRawDecodedWord('...')
      setRefinedSentence('Analysing...')
      setAiConfidence(0)
      
      ws.current.send(`START_DECODE|${activeSubject}`)
    }
  }

  const emergencyStop = () => {
    if (ws.current && isConnected) {
        ws.current.send('EMERGENCY_STOP') 
    }
    setActiveStep(0)
    setIntentState("idle")
    setIsRunning(false)
    setStatusMsg('Emergency Stop Activated')
  }

  const handleDeleteLastWord = () => {
    const words = refinedSentence.trim().split(" ")
    if (words.length > 0) {
      words.pop()
      setRefinedSentence(words.join(" ") || "")
      setIsSaved(false)
    }
  }

  const handleClearSentence = () => {
    setRefinedSentence("")
    setRawDecodedWord("---")
    setActiveStep(0)
    setIntentState("idle")
  }

  const handleSaveToHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/api/logs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject: activeSubject,
          raw_word: rawDecodedWord,
          final_sentence: refinedSentence,
          confidence: aiConfidence
        })
      });
      if (response.ok) {
        setIsSaved(true);
        setStatusMsg("Saved to History Successfully!");
      }
    } catch (error) {
      console.error("Gagal menyimpan log:", error);
    }
  }

  return (
    <div className="p-5 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-[#196484]" style={{ fontFamily: "var(--font-poppins)" }}>
            Live Session
          </h1>
          <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
            Real-time BCI inference dashboard
          </p>
        </div>
        
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${isConnected ? 'bg-emerald-50 border-emerald-200' : 'bg-red-50 border-red-200'}`}>
          <span className="relative flex h-2.5 w-2.5">
            {isConnected && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />}
            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
          </span>
          <span className={`text-xs font-semibold ${isConnected ? 'text-emerald-700' : 'text-red-700'}`} style={{ fontFamily: "var(--font-inter)" }}>
            {isConnected ? 'Connected to BCI Engine' : 'Server Offline'}
          </span>
        </div>
      </div>

      <Card className="rounded-2xl shadow-sm border-border bg-card mb-4 transition-all duration-300">
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <Button
              className="flex-1 gap-2 h-12 text-sm font-semibold rounded-lg transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg disabled:opacity-70 disabled:hover:translate-y-0"
              style={{
                background: isRunning || !isConnected ? "#94A3B8" : "linear-gradient(135deg, #00A3FF, #0080CC)",
                color: "#fff", border: "none"
              }}
              onClick={handleStartInference}
              disabled={isRunning || !isConnected}
            >
              {isRunning ? <><Loader2 className="w-4 h-4 animate-spin" />{statusMsg}</> : <><Zap className="w-4 h-4" />Start Inference Sequence</>}
            </Button>
            <Button
              variant="outline"
              className="h-12 px-6 gap-2 text-sm font-semibold rounded-lg border-2 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg"
              style={{ borderColor: "#FF903F", color: "#FF903F", backgroundColor: "transparent" }}
              onClick={emergencyStop}
            >
              <Square className="w-4 h-4" /> Emergency Stop
            </Button>
            <div className="px-3 py-2 rounded-lg text-xs text-slate-500 flex items-center gap-2 bg-slate-50">
              <Radio className="w-3 h-3 text-slate-400 flex-shrink-0" />
              <span style={{ fontFamily: "var(--font-inter)" }}>
                State: <strong className="capitalize text-[#196484]">{intentState.replace(/_/g, " ")}</strong>
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-3 md:grid-cols-2 gap-4">
        <div className="col-span-1">
          <SessionMetadataControls 
            subject={activeSubject} 
            setSubject={setActiveSubject} 
            isTTSEnabled={isTTSEnabled} 
            setIsTTSEnabled={setIsTTSEnabled} 
          />
        </div>

        <div className="col-span-2 md:col-span-1">
          <Card className="rounded-2xl shadow-sm border-border bg-card h-full transition-all duration-300">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold text-foreground" style={{ fontFamily: "var(--font-poppins)" }}>
                AI Pipeline
              </CardTitle>
            </CardHeader>
            <CardContent>
              <PipelineStepper activeStep={activeStep} />
            </CardContent>
          </Card>
        </div>

        <ResultDisplay
          intentState={intentState}
          rawDecodedWord={rawDecodedWord}
          refinedSentence={refinedSentence}
          aiConfidence={aiConfidence}
          isSaved={isSaved}
          onDeleteLastWord={handleDeleteLastWord}
          onClearSentence={handleClearSentence}
          onSaveLog={handleSaveToHistory}
        />
      </div>
    </div>
  )
}