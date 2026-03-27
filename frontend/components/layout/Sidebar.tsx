"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  Brain,
  Activity,
  Database,
  Cpu,
  Wifi,
} from "lucide-react"
import { cn } from "@/lib/utils"

const navItems = [
  {
    label: "Live Session",
    href: "/",
    icon: Brain,
    description: "Real-time inference",
  },
  {
    label: "Monitor & Control",
    href: "/monitor",
    icon: Activity,
    description: "Hardware & signals",
  },
  {
    label: "Data & Evaluation",
    href: "/evaluation",
    icon: Database,
    description: "Logs & metrics",
  },
]

export default function Sidebar() {
  const pathname = usePathname()
  const [ping, setPing] = useState(12)

  // Simulate ping updates
  useEffect(() => {
    const interval = setInterval(() => {
      setPing(Math.floor(Math.random() * 30) + 8)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <aside
      className="fixed left-0 top-0 h-screen w-64 flex flex-col z-40"
      style={{ backgroundColor: "var(--bci-dark-sidebar)" }}
    >
      {/* Logo */}
      <div className="px-5 py-6 border-b border-white/10">
        <div className="flex items-center gap-2">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "linear-gradient(135deg, #00A3FF, #10B981)" }}
          >
            <Cpu className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1
              className="text-sm font-bold leading-tight"
              style={{
                fontFamily: "var(--font-poppins)",
                background: "linear-gradient(90deg, #00A3FF, #10B981)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              NEURANDIAR
            </h1>
            <p className="text-xs text-slate-400 tracking-widest font-medium leading-none">
              BCI
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-5 flex flex-col gap-0.5">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest px-2 mb-2">
          Navigation
        </p>
        {navItems.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href)
          const Icon = item.icon

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2.5 px-2.5 py-2.5 rounded-lg transition-all duration-200 group"  ,
                isActive
                  ? "text-white"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
              )}
              style={
                isActive
                  ? { backgroundColor: "rgba(0,163,255,0.15)", borderLeft: "3px solid #00A3FF" }
                  : { borderLeft: "3px solid transparent" }
              }
            >
              <Icon
                className={cn(
                  "w-4 h-4 flex-shrink-0 transition-colors",
                  isActive ? "text-[#00A3FF]" : "text-slate-500 group-hover:text-slate-300"
                )}
              />
              <div>
                <p
                  className={cn(
                    "text-xs font-semibold leading-tight",
                    isActive ? "text-white" : ""
                  )}
                  style={{ fontFamily: "var(--font-poppins)" }}
                >
                  {item.label}
                </p>
                <p className="text-xs text-slate-500 leading-none">{item.description}</p>
              </div>
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-white/10">
        <div className="flex items-center justify-between gap-2 px-2">
          <div className="flex items-center gap-1.5">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
            </span>
            <span className="text-xs font-semibold text-slate-300" style={{ fontFamily: "var(--font-inter)" }}>
              Connected
            </span>
          </div>
          <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-slate-800/50 border border-slate-700">
            <Wifi className="w-3 h-3 text-slate-400" />
            <span className="text-xs font-semibold text-slate-300" style={{ fontFamily: "var(--font-inter)" }}>
              {ping}ms
            </span>
          </div>
        </div>
        <p className="text-xs text-slate-600 text-center mt-3">
          © 2026 BCI Research
        </p>
      </div>
    </aside>
  )
}
