"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import {
  Download,
  FileJson,
  Database,
  TrendingUp,
  Grid3X3,
  Brain,
  Sparkles,
} from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart as RechartsLineChart,
  Line,
  Legend,
} from "recharts"

// ─────────────────────────────────────────────────────────────────────────────
// MOCK DATA
// ─────────────────────────────────────────────────────────────────────────────
const MOCK_INFERENCE_LOGS = [
  { id: 1, timestamp: "14:23:45", rawWord: "MAKAN", refined: "Saya ingin makan", confidence: 94.5, status: "success" },
  { id: 2, timestamp: "14:23:52", rawWord: "TIDUR", refined: "Saya ingin tidur sekarang", confidence: 91.2, status: "success" },
  { id: 3, timestamp: "14:24:01", rawWord: "MINUM", refined: "Saya ingin minum air", confidence: 87.8, status: "success" },
  { id: 4, timestamp: "14:24:08", rawWord: "GANTI", refined: "Ganti posisi tubuh", confidence: 72.3, status: "warning" },
  { id: 5, timestamp: "14:24:15", rawWord: "BANTU", refined: "Butuh bantuan", confidence: 65.1, status: "warning" },
]

const MOCK_DATASETS = [
  { id: 1, subject: "Subject-01", trials: 120, rejected: 8, cleanEpochs: 1456 },
  { id: 2, subject: "Subject-02", trials: 115, rejected: 12, cleanEpochs: 1324 },
  { id: 3, subject: "Subject-03", trials: 128, rejected: 5, cleanEpochs: 1598 },
]

const MOCK_RAW_LOGS = `Trial 1: MAKAN - Marker 1 [t=0ms, quality=good]
Trial 2: TIDUR - Marker 2 [t=2500ms, quality=good]
Trial 3: MINUM - Marker 3 [t=5100ms, quality=fair]
Trial 4: GANTI - Marker 4 [t=7800ms, quality=good]
Trial 5: BANTU - Marker 5 [t=10200ms, quality=fair]
...`

const MOCK_OPTUNA_TRIALS = [
  { trial: 1, dropout: 0.25, f1Filters: 8, valAcc: 0.812 },
  { trial: 2, dropout: 0.30, f1Filters: 16, valAcc: 0.834 },
  { trial: 3, dropout: 0.35, f1Filters: 32, valAcc: 0.851 },
  { trial: 4, dropout: 0.28, f1Filters: 24, valAcc: 0.828 },
  { trial: 5, dropout: 0.32, f1Filters: 40, valAcc: 0.867 },
]

const MOCK_MODEL_VERSIONS = [
  { version: "v1.0", f1: 0.82, acc: 0.85, loss: 0.42 },
  { version: "v1.1", f1: 0.84, acc: 0.87, loss: 0.38 },
  { version: "v2.0", f1: 0.89, acc: 0.91, loss: 0.28 },
  { version: "v2.1-best", f1: 0.91, acc: 0.93, loss: 0.22 },
]

const MOCK_TRAINING_CURVES = [
  { epoch: 1, loss: 0.95, acc: 0.45, valLoss: 0.92, valAcc: 0.48 },
  { epoch: 5, loss: 0.68, acc: 0.65, valLoss: 0.71, valAcc: 0.62 },
  { epoch: 10, loss: 0.45, acc: 0.78, valLoss: 0.52, valAcc: 0.74 },
  { epoch: 15, loss: 0.32, acc: 0.85, valLoss: 0.38, valAcc: 0.82 },
  { epoch: 20, loss: 0.22, acc: 0.91, valLoss: 0.28, valAcc: 0.88 },
]

const MOCK_LATENCY_DATA = [
  { percentile: "P50", latency: 145 },
  { percentile: "P75", latency: 187 },
  { percentile: "P90", latency: 234 },
  { percentile: "P95", latency: 278 },
  { percentile: "P99", latency: 345 },
]

// ─────────────────────────────────────────────────────────────────────────────

function ExportButton({ format, filename }: { format: "csv" | "json" | "npy"; filename: string }) {
  const handleExport = () => {
    let content = ""
    let type = "text/plain"

    if (format === "csv") {
      content = "timestamp,raw_word,refined_output,confidence,status\n"
      MOCK_INFERENCE_LOGS.forEach(log => {
        content += `"${log.timestamp}","${log.rawWord}","${log.refined}",${log.confidence},"${log.status}"\n`
      })
      type = "text/csv"
    } else if (format === "json") {
      content = JSON.stringify(MOCK_INFERENCE_LOGS, null, 2)
      type = "application/json"
    } else {
      content = "Mock .npy binary data"
      type = "application/octet-stream"
    }

    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `${filename}.${format}`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Button
      size="sm"
      variant="outline"
      className="gap-1.5 text-xs h-9 transition-all duration-200 hover:-translate-y-0.5"
      onClick={handleExport}
    >
      <Download className="w-3 h-3" />
      Download {format.toUpperCase()}
    </Button>
  )
}

export default function EvaluationPage() {
  return (
    <div className="p-5 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1
            className="text-xl font-bold text-[#196484] text-balance"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            Data Center &amp; Evaluation
          </h1>
          <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
            Live inference logs, datasets, models, and performance metrics
          </p>
        </div>
        <Button
          className="gap-1.5 text-xs h-9 bg-[#196484] hover:bg-[#164A6F] text-white transition-all duration-200 hover:-translate-y-0.5"
        >
          <Download className="w-3 h-3" />
          Export Report (PDF)
        </Button>
      </div>

      {/* Top Metrics Row - Always Visible */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        {/* Inference Speed */}
        <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
          <CardContent className="p-4 flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "#EFF6FF" }}
            >
              <TrendingUp className="w-5 h-5 text-[#00A3FF]" />
            </div>
            <div>
              <p className="text-xs text-slate-500 font-medium" style={{ fontFamily: "var(--font-inter)" }}>
                Inference Speed (Median)
              </p>
              <p className="text-xl font-extrabold text-[#196484]" style={{ fontFamily: "var(--font-poppins)" }}>
                45.2 ms
              </p>
            </div>
          </CardContent>
        </Card>

        {/* 95th Percentile Latency */}
        <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
          <CardContent className="p-4 flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "#FEF3C7" }}
            >
              <Sparkles className="w-5 h-5 text-[#FF903F]" />
            </div>
            <div>
              <p className="text-xs text-slate-500 font-medium" style={{ fontFamily: "var(--font-inter)" }}>
                95th Percentile Latency
              </p>
              <p className="text-xl font-extrabold text-[#196484]" style={{ fontFamily: "var(--font-poppins)" }}>
                61.8 ms
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Active Production Model */}
        <Card className="rounded-2xl shadow-sm border-border bg-card transition-all duration-300">
          <CardContent className="p-4 flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "#ECFDF5" }}
            >
              <Brain className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <p className="text-xs text-slate-500 font-medium" style={{ fontFamily: "var(--font-inter)" }}>
                Active Production Model
              </p>
              <p className="text-lg font-extrabold text-[#196484]" style={{ fontFamily: "var(--font-poppins)" }}>
                EEGNet-v2 (Subject-01)
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs Component */}
      <Tabs defaultValue="logs" className="w-full">
        <TabsList className="grid w-full grid-cols-4 h-10 p-1 rounded-lg bg-slate-100 border border-border">
          <TabsTrigger
            value="logs"
            className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:text-[#196484] transition-all duration-200 rounded-md"
          >
            Live Inference Logs
          </TabsTrigger>
          <TabsTrigger
            value="datasets"
            className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:text-[#196484] transition-all duration-200 rounded-md"
          >
            Datasets &amp; Logs
          </TabsTrigger>
          <TabsTrigger
            value="mlflow"
            className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:text-[#196484] transition-all duration-200 rounded-md"
          >
            MLflow &amp; Optuna
          </TabsTrigger>
          <TabsTrigger
            value="evaluation"
            className="text-xs font-semibold data-[state=active]:bg-white data-[state=active]:text-[#196484] transition-all duration-200 rounded-md"
          >
            Model Evaluation &amp; XAI
          </TabsTrigger>
        </TabsList>

        {/* TAB 1: LIVE INFERENCE LOGS */}
        <TabsContent value="logs" className="mt-4 space-y-4">
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle
                  className="text-sm font-semibold text-foreground"
                  style={{ fontFamily: "var(--font-poppins)" }}
                >
                  Real-Time Decoding History
                </CardTitle>
                <ExportButton format="csv" filename="inference_logs" />
              </div>
              <p className="text-xs text-slate-400 mt-1" style={{ fontFamily: "var(--font-inter)" }}>
                Live inference results with confidence scores and session metadata
              </p>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table className="text-xs">
                  <TableHeader>
                    <TableRow className="border-border hover:bg-transparent">
                      <TableHead className="h-8 text-slate-600 font-semibold">Timestamp</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold">Raw Word</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold">Refined Output</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-right">Confidence</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {MOCK_INFERENCE_LOGS.map((log) => (
                      <TableRow key={log.id} className="border-border hover:bg-slate-50">
                        <TableCell className="py-2 text-slate-700" style={{ fontFamily: "var(--font-inter)" }}>
                          {log.timestamp}
                        </TableCell>
                        <TableCell className="py-2 text-slate-700 font-semibold">{log.rawWord}</TableCell>
                        <TableCell className="py-2 text-slate-700">{log.refined}</TableCell>
                        <TableCell className="py-2 text-right">
                          <div className="flex items-center justify-end gap-1">
                            <div className="h-1.5 w-12 bg-slate-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-[#00A3FF]"
                                style={{ width: `${log.confidence}%` }}
                              />
                            </div>
                            <span className="font-semibold text-[#196484] min-w-fit">{log.confidence}%</span>
                          </div>
                        </TableCell>
                        <TableCell className="py-2 text-center">
                          <Badge
                            className={`text-xs font-semibold ${
                              log.status === "success"
                                ? "bg-emerald-100 text-emerald-700 border-emerald-200 border"
                                : "bg-orange-100 text-orange-700 border-orange-200 border"
                            }`}
                          >
                            {log.status}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 2: DATASETS & OFFLINE LOGS */}
        <TabsContent value="datasets" className="mt-4 space-y-4">
          {/* Dataset Metadata */}
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5">
              <CardTitle
                className="text-sm font-semibold text-foreground"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                Dataset Metadata
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
                Summary of collected EEG datasets and data quality metrics
              </p>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table className="text-xs">
                  <TableHeader>
                    <TableRow className="border-border">
                      <TableHead className="h-8 text-slate-600 font-semibold">Subject ID</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Trials</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Artifacts Rejected</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Clean Epochs</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {MOCK_DATASETS.map((ds) => (
                      <TableRow key={ds.id} className="border-border hover:bg-slate-50">
                        <TableCell className="py-2 text-slate-700 font-semibold">{ds.subject}</TableCell>
                        <TableCell className="py-2 text-center text-slate-700">{ds.trials}</TableCell>
                        <TableCell className="py-2 text-center">
                          <Badge className="bg-red-100 text-red-700 border-red-200 border text-xs">
                            {ds.rejected}
                          </Badge>
                        </TableCell>
                        <TableCell className="py-2 text-center text-slate-700 font-semibold">{ds.cleanEpochs}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          {/* Raw Experiment Logs */}
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5">
              <CardTitle
                className="text-sm font-semibold text-foreground"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                Raw Experiment Logs
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
                Terminal-style log output for debugging
              </p>
            </CardHeader>
            <CardContent>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto max-h-40 overflow-y-auto">
                <pre>{MOCK_RAW_LOGS}</pre>
              </div>
              <div className="mt-3 flex gap-2">
                <ExportButton format="json" filename="experiment_logs" />
                <Button
                  size="sm"
                  variant="outline"
                  className="gap-1.5 text-xs h-9 transition-all duration-200 hover:-translate-y-0.5"
                >
                  <FileJson className="w-3 h-3" />
                  Download X_features.npy
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="gap-1.5 text-xs h-9 transition-all duration-200 hover:-translate-y-0.5"
                >
                  <FileJson className="w-3 h-3" />
                  Download y_labels.npy
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 3: MLFLOW & OPTUNA REGISTRY */}
        <TabsContent value="mlflow" className="mt-4 space-y-4">
          {/* Model Versions Table */}
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5">
              <CardTitle
                className="text-sm font-semibold text-foreground flex items-center gap-2"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                <TrendingUp className="w-4 h-4 text-[#00A3FF]" />
                Model Registry (MLflow)
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
                Trained model versions and performance
              </p>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table className="text-xs">
                  <TableHeader>
                    <TableRow className="border-border">
                      <TableHead className="h-8 text-slate-600 font-semibold">Version</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">F1 Score</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Accuracy</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Loss</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Action</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {MOCK_MODEL_VERSIONS.map((m, i) => (
                      <TableRow key={i} className="border-border hover:bg-slate-50">
                        <TableCell className="py-2">
                          <Badge
                            className={`text-xs font-semibold ${
                              m.version.includes("best")
                                ? "bg-emerald-100 text-emerald-700 border-emerald-200 border"
                                : "bg-slate-100 text-slate-700 border-slate-200 border"
                            }`}
                          >
                            {m.version}
                          </Badge>
                        </TableCell>
                        <TableCell className="py-2 text-center text-slate-700 font-semibold">{(m.f1 * 100).toFixed(1)}%</TableCell>
                        <TableCell className="py-2 text-center text-slate-700 font-semibold">{(m.acc * 100).toFixed(1)}%</TableCell>
                        <TableCell className="py-2 text-center text-slate-700">{m.loss.toFixed(3)}</TableCell>
                        <TableCell className="py-2 text-center">
                          <Button size="sm" variant="ghost" className="h-6 text-xs">
                            Download
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>

          {/* Optuna Hyperparameter Tuning */}
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5">
              <CardTitle
                className="text-sm font-semibold text-foreground flex items-center gap-2"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                <Sparkles className="w-4 h-4 text-[#FF903F]" />
                Optuna Hyperparameter Tuning
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
                Automated hyperparameter optimization trials
              </p>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <Table className="text-xs">
                  <TableHeader>
                    <TableRow className="border-border">
                      <TableHead className="h-8 text-slate-600 font-semibold">Trial #</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Dropout</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">F1 Filters</TableHead>
                      <TableHead className="h-8 text-slate-600 font-semibold text-center">Val Accuracy</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {MOCK_OPTUNA_TRIALS.map((t) => (
                      <TableRow key={t.trial} className="border-border hover:bg-slate-50">
                        <TableCell className="py-2 text-slate-700 font-semibold">Trial {t.trial}</TableCell>
                        <TableCell className="py-2 text-center text-slate-700">{(t.dropout * 100).toFixed(0)}%</TableCell>
                        <TableCell className="py-2 text-center text-slate-700">{t.f1Filters}</TableCell>
                        <TableCell className="py-2 text-center">
                          <span className="font-semibold text-[#00A3FF]">{(t.valAcc * 100).toFixed(1)}%</span>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 4: MODEL EVALUATION & XAI */}
        <TabsContent value="evaluation" className="mt-4 space-y-4">
          {/* Training Curves */}
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5">
              <CardTitle
                className="text-sm font-semibold text-foreground"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                Training Curves
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
                Loss and accuracy progression during model training
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <RechartsLineChart data={MOCK_TRAINING_CURVES}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: "#94A3B8" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94A3B8" }} />
                  <Tooltip contentStyle={{ background: "#fff", border: "1px solid #E2EBF3", borderRadius: "8px", fontSize: "11px" }} />
                  <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "var(--font-inter)" }} />
                  <Line type="monotone" dataKey="loss" stroke="#FF903F" dot={false} isAnimationActive={false} strokeWidth={1.5} />
                  <Line type="monotone" dataKey="acc" stroke="#00A3FF" dot={false} isAnimationActive={false} strokeWidth={1.5} />
                </RechartsLineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Latency Percentiles */}
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5">
              <CardTitle
                className="text-sm font-semibold text-foreground"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                Inference Latency Percentiles
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
                End-to-end latency distribution across inference batches
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={MOCK_LATENCY_DATA} barSize={40}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
                  <XAxis dataKey="percentile" tick={{ fontSize: 10, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "#fff", border: "1px solid #E2EBF3", borderRadius: "8px", fontSize: "11px" }} formatter={(v: number) => [`${v}ms`, "Latency"]} />
                  <Bar dataKey="latency" fill="#00A3FF" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Confusion Matrix Placeholder */}
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5">
              <CardTitle
                className="text-sm font-semibold text-foreground flex items-center gap-2"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                <Grid3X3 className="w-4 h-4 text-[#10B981]" />
                Confusion Matrix (19x19 Intent Classes)
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
                Classification accuracy per intent class
              </p>
            </CardHeader>
            <CardContent>
              <div className="h-48 flex items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50">
                <div className="text-center">
                  <Grid3X3 className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                  <p className="text-xs text-slate-400">19×19 confusion matrix heatmap</p>
                  <p className="text-xs text-slate-400">showing per-class classification rates</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* SHAP / DeepLIFT Explainability */}
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5">
              <CardTitle
                className="text-sm font-semibold text-foreground flex items-center gap-2"
                style={{ fontFamily: "var(--font-poppins)" }}
              >
                <Brain className="w-4 h-4 text-[#8B5CF6]" />
                SHAP / DeepLIFT Explainability
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
                Feature importance heatmap showing which EEG channels and time windows the model uses
              </p>
            </CardHeader>
            <CardContent>
              <div className="h-48 flex items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50">
                <div className="text-center">
                  <Brain className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                  <p className="text-xs text-slate-400">Feature Importance Heatmap</p>
                  <p className="text-xs text-slate-400">(Channels × Time) - Motor Cortex Focus</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
