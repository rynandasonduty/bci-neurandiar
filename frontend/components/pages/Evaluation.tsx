"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, TrendingUp, Brain, Sparkles } from "lucide-react"
import { API_URL } from "@/lib/api"
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart as RechartsLineChart, Line, Legend,
} from "recharts"

// Komponen Tombol Export (Menerima Data Dinamis)
function ExportButton({ format, filename, data }: { format: "csv" | "json" | "npy"; filename: string; data: any[] }) {
  const handleExport = () => {
    let content = ""
    let type = "text/plain"

    if (format === "csv") {
      content = "timestamp,raw_word,refined_output,confidence\n"
      data.forEach((log: any) => {
        content += `"${log.timestamp}","${log.rawWord}","${log.refined}",${log.confidence}\n`
      })
      type = "text/csv"
    } else if (format === "json") {
      content = JSON.stringify(data, null, 2)
      type = "application/json"
    } else {
      content = "Binary data placeholder for .npy export"
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
    <Button size="sm" variant="outline" className="gap-1.5 text-xs h-9 transition-all duration-200 hover:-translate-y-0.5" onClick={handleExport}>
      <Download className="w-3 h-3" /> Download {format.toUpperCase()}
    </Button>
  )
}

export default function EvaluationPage() {
  // --- STATE PENAMPUNG DATA DINAMIS ---
  const [inferenceLogs, setInferenceLogs] = useState<any[]>([])
  const [overviewMetrics, setOverviewMetrics] = useState({ median_latency: 0, p95_latency: 0, active_model: "Loading..." })
  const [modelVersions, setModelVersions] = useState<any[]>([])
  
  // State Baru untuk menggantikan Mock Data
  const [datasetMeta, setDatasetMeta] = useState<any[]>([])
  const [rawLogsPreview, setRawLogsPreview] = useState<string>("Loading logs...")
  const [optunaTrials, setOptunaTrials] = useState<any[]>([])
  const [trainingCurves, setTrainingCurves] = useState<any[]>([])

  // --- FETCH DATA DARI BACKEND FASTAPI ---
  useEffect(() => {
    // 1. Fetch live inference log table
    fetch(`${API_URL}/api/logs`)
      .then(res => res.json())
      .then(data => {
        if (data.status === "success") {
          const formattedLogs = data.data.map((log: any, index: number) => ({
            id: index,
            timestamp: log["timestamp"],
            rawWord: log["raw_word"],
            refined: log["final_sentence"],
            confidence: parseFloat(log["confidence"]),
          }))
          setInferenceLogs(formattedLogs)
        }
      }).catch(err => console.error("[Evaluation] Logs fetch error:", err))

    // 2. Fetch all metrics, registry, and evaluation data
    fetch(`${API_URL}/api/metrics`)
      .then(res => res.json())
      .then(data => {
        if (data.status === "success") {
          setOverviewMetrics(data.overview)
          setModelVersions(data.mlflow_registry)
          setDatasetMeta(data.dataset_meta)
          setRawLogsPreview(data.raw_logs_preview)
          setOptunaTrials(data.optuna_trials)
          setTrainingCurves(data.training_curves)
        }
      }).catch(err => console.error("[Evaluation] Metrics fetch error:", err))
  }, [])

  return (
    <div className="p-5 max-w-7xl mx-auto">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-[#196484]" style={{ fontFamily: "var(--font-poppins)" }}>
            Data Center &amp; Evaluation
          </h1>
          <p className="text-xs text-slate-400 mt-0.5" style={{ fontFamily: "var(--font-inter)" }}>
            Live inference logs, datasets, models, and performance metrics
          </p>
        </div>
        <Button className="gap-1.5 text-xs h-9 bg-[#196484] hover:bg-[#164A6F] text-white">
          <Download className="w-3 h-3" /> Export Report (PDF)
        </Button>
      </div>

      {/* METRIC CARDS (DINAMIS) */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <Card className="rounded-2xl shadow-sm border-border bg-card">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-blue-50">
              <TrendingUp className="w-5 h-5 text-[#00A3FF]" />
            </div>
            <div>
              <p className="text-xs text-slate-500 font-medium">Inference Speed (Median)</p>
              <p className="text-xl font-extrabold text-[#196484]">{overviewMetrics.median_latency} ms</p>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-2xl shadow-sm border-border bg-card">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-orange-50">
              <Sparkles className="w-5 h-5 text-[#FF903F]" />
            </div>
            <div>
              <p className="text-xs text-slate-500 font-medium">95th Percentile Latency</p>
              <p className="text-xl font-extrabold text-[#196484]">{overviewMetrics.p95_latency} ms</p>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-2xl shadow-sm border-border bg-card">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-emerald-50">
              <Brain className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <p className="text-xs text-slate-500 font-medium">Active Production Model</p>
              <p className="text-lg font-extrabold text-[#196484] truncate w-40">{overviewMetrics.active_model}</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="logs" className="w-full">
        <TabsList className="grid w-full grid-cols-4 h-10 p-1 rounded-lg bg-slate-100 border border-border">
          <TabsTrigger value="logs" className="text-xs font-semibold rounded-md">Live Inference Logs</TabsTrigger>
          <TabsTrigger value="datasets" className="text-xs font-semibold rounded-md">Datasets &amp; Logs</TabsTrigger>
          <TabsTrigger value="mlflow" className="text-xs font-semibold rounded-md">MLflow &amp; Optuna</TabsTrigger>
          <TabsTrigger value="evaluation" className="text-xs font-semibold rounded-md">Model Evaluation &amp; XAI</TabsTrigger>
        </TabsList>

        {/* TAB 1: LOGS */}
        <TabsContent value="logs" className="mt-4 space-y-4">
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-3 flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-semibold">Real-Time Decoding History</CardTitle>
              <ExportButton format="csv" filename="inference_logs" data={inferenceLogs} />
            </CardHeader>
            <CardContent>
              <Table className="text-xs">
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead><TableHead>Raw Word</TableHead>
                    <TableHead>Refined Output</TableHead><TableHead className="text-right">Confidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {inferenceLogs.length === 0 ? (
                    <TableRow><TableCell colSpan={4} className="text-center py-4 text-slate-500">Belum ada riwayat.</TableCell></TableRow>
                  ) : (
                    inferenceLogs.map((log) => (
                      <TableRow key={log.id}>
                        <TableCell className="py-2">{log.timestamp}</TableCell>
                        <TableCell className="py-2 font-semibold">{log.rawWord}</TableCell>
                        <TableCell className="py-2">{log.refined}</TableCell>
                        <TableCell className="py-2 text-right">
                          <span className="font-semibold text-[#196484]">{log.confidence.toFixed(1)}%</span>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 2: DATASETS */}
        <TabsContent value="datasets" className="mt-4 space-y-4">
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5 flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-semibold">Dataset Metadata</CardTitle>
              <div className="flex gap-2">
                <ExportButton format="npy" filename="X_features" data={[]} />
                <ExportButton format="npy" filename="y_labels" data={[]} />
              </div>
            </CardHeader>
            <CardContent>
              <Table className="text-xs">
                <TableHeader>
                  <TableRow>
                    <TableHead>Subject ID</TableHead><TableHead className="text-center">Trials</TableHead>
                    <TableHead className="text-center">Artifacts Rejected</TableHead><TableHead className="text-center">Clean Epochs</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {datasetMeta.map((ds, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-semibold">{ds.subject}</TableCell><TableCell className="text-center">{ds.trials}</TableCell>
                      <TableCell className="text-center"><Badge variant="destructive">{ds.rejected}</Badge></TableCell><TableCell className="text-center">{ds.cleanEpochs}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5"><CardTitle className="text-sm font-semibold">Raw Experiment Logs</CardTitle></CardHeader>
            <CardContent>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 font-mono text-xs text-slate-300 overflow-y-auto max-h-40">
                <pre>{rawLogsPreview}</pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 3: MLFLOW */}
        <TabsContent value="mlflow" className="mt-4 space-y-4">
          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5"><CardTitle className="text-sm font-semibold">Model Registry (MLflow)</CardTitle></CardHeader>
            <CardContent>
              <Table className="text-xs">
                <TableHeader>
                  <TableRow>
                    <TableHead>Version</TableHead><TableHead className="text-center">Status</TableHead>
                    <TableHead className="text-center">F1 Score</TableHead><TableHead className="text-center">Loss</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {modelVersions.map((m, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-semibold text-[#196484]">{m.version}</TableCell>
                      <TableCell className="text-center"><Badge variant="outline">{m.status}</Badge></TableCell>
                      <TableCell className="text-center">{(m.f1_score * 100).toFixed(1)}%</TableCell><TableCell className="text-center">{m.loss.toFixed(3)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5"><CardTitle className="text-sm font-semibold">Optuna Hyperparameter Tuning</CardTitle></CardHeader>
            <CardContent>
              <Table className="text-xs">
                <TableHeader>
                  <TableRow><TableHead>Trial</TableHead><TableHead>Dropout</TableHead><TableHead>Val Acc</TableHead></TableRow>
                </TableHeader>
                <TableBody>
                  {optunaTrials.map((t, i) => (
                    <TableRow key={i}><TableCell>Trial {t.trial}</TableCell><TableCell>{t.dropout}</TableCell><TableCell>{t.valAcc}</TableCell></TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        {/* TAB 4: EVALUATION */}
        <TabsContent value="evaluation" className="mt-4 space-y-4">
           <Card className="rounded-2xl shadow-sm border-border bg-card">
            <CardHeader className="pb-2.5"><CardTitle className="text-sm font-semibold">Training Curves</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <RechartsLineChart data={trainingCurves}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="loss" stroke="#FF903F" strokeWidth={1.5} name="Training Loss" />
                  <Line type="monotone" dataKey="acc" stroke="#00A3FF" strokeWidth={1.5} name="Training Acc" />
                </RechartsLineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}