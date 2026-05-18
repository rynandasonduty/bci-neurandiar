/**
 * Central API URL configuration.
 * All components must import these constants rather than hardcoding
 * http://127.0.0.1:8000, so that deploying to a different host only
 * requires changing NEXT_PUBLIC_API_URL in the .env.local file.
 */
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"
export const WS_URL  = process.env.NEXT_PUBLIC_WS_URL  ?? "ws://127.0.0.1:8000"
