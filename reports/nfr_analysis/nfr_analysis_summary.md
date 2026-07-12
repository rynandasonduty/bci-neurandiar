# NeurAndiAr NFR Analysis Report

**Generated:** 2026-07-13 03:32:48

All figures below are measured, not estimated: frontend from a real Lighthouse run against the production (`next build && next start`) server, backend from 25 real logged requests against the live `/ws/inference` endpoint (champion model P3_SVM, subject S3).

## Frontend (Lighthouse, production build)

| Score | Value |
|---|---|
| Performance | 73.0/100 |
| Accessibility | 88.0/100 |
| Best Practices | 93.0/100 |
| Seo | 100/100 |

| Lab metric | Value |
|---|---|
| LCP_ms | 4995.5 |
| CLS | 0 |
| FCP_ms | 1098.9 |
| TTI_ms | 4995.5 |
| TBT_ms | 284 |
| speed_index_ms | 4269.1 |
| TTFB_ms | 14.3 |
| max_potential_FID_ms | 239 |

> FID is a real-user (field) metric and cannot be measured by Lighthouse; 'max_potential_FID_ms' is a lab proxy Lighthouse computes, not true FID/INP. No CI/Lighthouse pipeline previously existed in this repo -- this is a fresh, one-off measurement.

## Backend inference latency (live production, P3_SVM champion, subject S3)

n = 25 real logged requests from `backend/logs/latency_history.csv`.

| Statistic | Total latency (ms) |
|---|---|
| mean | 1064.97 |
| median | 1058.07 |
| std_dev | 30.27 |
| min | 1021.2 |
| max | 1140.69 |
| p25 | 1045.18 |
| p50 | 1058.07 |
| p75 | 1097.25 |
| p95 | 1104.68 |
| p99 | 1132.05 |

### Per-stage breakdown (mean, % of total)

| Stage | Mean (ms) | % of total |
|---|---|---|
| Signal read + filter | 0.318 | 0.03% |
| SVM inference (slot 1, incl. Barlow feature extraction) | 535.47 | 50.28% |
| SVM inference (slot 2, incl. Barlow feature extraction) | 528.632 | 49.64% |
| Word assembly | 0.195 | 0.02% |
| Rule-based sentence refinement | 0.0 | 0.0% |

> The live API has no REST /inference endpoint, no client-selectable model, and no backend TTS step (TTS is client-side via the browser Web Speech API, fired after the response arrives). These are real per-request timings logged by the deployed system, not synthetic requests.

## Supplementary: offline P6-cascade comparison (research only, not deployed)

P6 is not wired into the live API; this is kept only as supplementary context from `backend/src/experiments_p4_p7/fair_comparison/results/p6_latency_measurement.json`.

- P3 single call (fair, incl. feature extraction): 652.48 ms
- P6 non-SA case: 1233.45 ms
- P6 SA-branch case: 1805.90 ms
