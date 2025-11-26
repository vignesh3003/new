import { useMemo, useState } from 'react'
import { MathJax, MathJaxContext } from 'better-react-mathjax'
import { analyzeEcg } from './lib/api'
import type { AnalysisResponse, FormulaPayload, PlotPayload, StepPayload } from './types'

const mathJaxConfig = {
  loader: { load: ['input/tex', 'output/chtml'] },
  tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
}

const FALLBACK_FORMULAS: FormulaPayload[] = [
  {
    title: 'Discrete Fourier Transform (DFT)',
    latex: String.raw`X(k) = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi kn / N}`,
  },
  {
    title: 'Inverse DFT',
    latex: String.raw`x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X(k)\, e^{j 2\pi kn / N}`,
  },
  {
    title: 'Heartbeat Alignment',
    latex: String.raw`\tilde{x}_k[n] = x\big(r_k + n - n_0\big),\; -n_0 \le n < n_1`,
  },
  {
    title: 'PCA Variance Maximization',
    latex: String.raw`\text{Var}(\mathbf{z}) = \lambda = \max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \mathbf{S}\, \mathbf{w}`,
  },
  {
    title: 'PCA Projection',
    latex: String.raw`\mathbf{z} = \mathbf{X}_c \mathbf{W}, \quad \mathbf{X}_c = \mathbf{X} - \mathbf{1}\mu^\top`,
  },
]

const FALLBACK_STEPS: StepPayload[] = [
  { title: 'Load ECG signal', details: 'Upload a single-column CSV that contains voltage samples.' },
  { title: 'R-peak alignment', details: 'Center every heartbeat around its detected R-peak.' },
  { title: 'Adaptive Fourier projection', details: 'Transform each beat to the frequency domain.' },
  { title: 'PCA', details: 'Project the spectral feature matrix into 2 principal components.' },
]

const FALLBACK_METHOD_SUMMARY =
  'Adaptive Heartbeat-Aligned Fourier PCA synchronizes each heartbeat by its R-peak, converts the aligned beats to spectral fingerprints, and then runs PCA to produce compact, noise-robust embeddings.'

function App() {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fileLabel, setFileLabel] = useState('No file selected')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const activeFormulas = analysis?.formulas ?? FALLBACK_FORMULAS
  const activeSteps = analysis?.steps ?? FALLBACK_STEPS
  const methodSummary = analysis?.method_summary ?? FALLBACK_METHOD_SUMMARY

  const dominantFrequency = useMemo(() => {
    if (!analysis) return null
    const { fft_magnitudes, fft_frequencies } = analysis
    if (!fft_magnitudes.length) return null
    let maxIdx = 0
    for (let i = 1; i < fft_magnitudes.length; i += 1) {
      if (fft_magnitudes[i] > fft_magnitudes[maxIdx]) {
        maxIdx = i
      }
    }
    return {
      freq: Number(fft_frequencies[maxIdx].toFixed(2)),
      magnitude: Number(fft_magnitudes[maxIdx].toFixed(4)),
    }
  }, [analysis])

  const varianceSummary = useMemo(() => {
    if (!analysis) return null
    return analysis.explained_variance.map((ratio) => `${(ratio * 100).toFixed(1)}%`).join(' / ')
  }, [analysis])

  function handleFileChange(fileList: FileList | null) {
    if (!fileList || !fileList.length) return
    const file = fileList[0]
    setFileLabel(file.name)
    setSelectedFile(file)
    setError(null)
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      setError('Select a CSV file before processing.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const result = await analyzeEcg(selectedFile)
      setAnalysis(result)
    } catch (err) {
      console.error(err)
      setError(
        err instanceof Error ? err.message : 'Unable to analyze the ECG file. Please try again.',
      )
    } finally {
      setLoading(false)
    }
  }

  const heartbeatCount = analysis?.heartbeat_count ?? null

  return (
    <MathJaxContext version={3} config={mathJaxConfig}>
      <div className="min-h-screen bg-slate-950 text-slate-100">
        <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10 lg:py-14">
          <header className="rounded-3xl border border-white/10 bg-gradient-to-r from-ocean-500/10 to-slate-800/50 p-8 shadow-2xl shadow-slate-900/30">
            <p className="text-sm uppercase tracking-[0.2em] text-ocean-200">Fourier + PCA</p>
            <h1 className="mt-2 text-3xl font-semibold text-white lg:text-4xl">
              ECG Signal Analysis Dashboard
            </h1>
            <p className="mt-3 max-w-3xl text-base text-slate-300">
              Upload a single-lead ECG CSV. The backend automatically detects R-peaks, aligns each
              heartbeat, converts the segments to spectral fingerprints, and projects them with PCA
              so you can inspect rhythm signatures in both the time and frequency domains.
            </p>
          </header>

          <MethodHighlight summary={methodSummary} />

          <section className="grid gap-6 lg:grid-cols-[1.6fr,1fr]">
            <div className="rounded-3xl border border-white/10 bg-slate-900/50 p-8">
              <h2 className="text-xl font-semibold text-white">1. Upload ECG CSV</h2>
              <p className="mt-2 text-sm text-slate-300">
                CSV must contain a single column of voltage samples (any sampling rate). We
                auto-normalize and validate the data server-side.
              </p>
              <label
                htmlFor="ecg-file"
                className="mt-6 flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/70 px-6 py-14 text-center transition hover:border-ocean-400 hover:bg-slate-900"
              >
                <span className="text-base font-medium text-white">{fileLabel}</span>
                <span className="mt-2 text-xs text-slate-400">
                  Drop a .csv file or click to browse
                </span>
                <input
                  id="ecg-file"
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(event) => handleFileChange(event.target.files)}
                />
              </label>
              <button
                type="button"
                disabled={loading}
                className="mt-6 inline-flex items-center justify-center rounded-2xl border border-ocean-400 bg-ocean-500 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-ocean-500/30 transition hover:bg-ocean-400 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={handleAnalyze}
              >
                {loading ? 'Analyzing…' : 'Process Signal'}
              </button>
              {error && <p className="mt-4 text-sm text-rose-300">{error}</p>}
            </div>

            <FormulaPanel formulas={activeFormulas} />
          </section>

          <section className="grid gap-6 lg:grid-cols-3">
            <StatCard
              title="Dominant frequency"
              subtitle="Peak magnitude in the FFT"
              value={dominantFrequency ? `${dominantFrequency.freq} Hz` : '—'}
              detail={dominantFrequency ? `|X(f)| = ${dominantFrequency.magnitude}` : 'Run analysis'}
            />
            <StatCard
              title="PCA variance"
              subtitle="Explained variance ratio (PC1 / PC2)"
              value={varianceSummary ?? '—'}
              detail={analysis ? 'Whitened PCA on spectral matrix' : 'Awaiting analysis'}
            />
            <StatCard
              title="Heartbeats aligned"
              subtitle="Adaptive heartbeat segments"
              value={heartbeatCount !== null ? heartbeatCount.toString() : '—'}
              detail={
                heartbeatCount !== null ? 'R-peak centered beats ready for FFT' : 'Upload an ECG CSV'
              }
            />
          </section>

          {analysis && <PlotGallery plots={analysis.plots} />}
          <StepTimeline steps={activeSteps} completed={Boolean(analysis)} />
        </div>
      </div>
    </MathJaxContext>
  )
}

type StatProps = {
  title: string
  subtitle: string
  value: string
  detail: string
}

function StatCard({ title, subtitle, value, detail }: StatProps) {
  return (
    <div className="rounded-3xl border border-white/5 bg-slate-900/60 p-6">
      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">{subtitle}</p>
      <p className="mt-2 text-lg font-semibold text-white">{title}</p>
      <p className="mt-4 text-3xl font-semibold text-ocean-200">{value}</p>
      <p className="mt-3 text-sm text-slate-400">{detail}</p>
    </div>
  )
}

function PlotGallery({ plots }: { plots: PlotPayload[] }) {
  return (
    <section className="rounded-3xl border border-white/10 bg-slate-900/40 p-6">
      <h2 className="text-xl font-semibold text-white">Signal views</h2>
      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        {plots.map((plot) => (
          <figure
            key={plot.title}
            className="rounded-2xl border border-white/5 bg-slate-950/50 p-3 shadow-inner shadow-black/20"
          >
            <img
              src={plot.image_base64}
              alt={plot.title}
              className="h-56 w-full rounded-xl object-cover"
            />
            <figcaption className="mt-3 text-sm font-medium text-white">{plot.title}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}

function FormulaPanel({ formulas }: { formulas: FormulaPayload[] }) {
  return (
    <aside className="rounded-3xl border border-white/10 bg-slate-900/70 p-6">
      <h2 className="text-xl font-semibold text-white">Governing equations</h2>
      <p className="mt-2 text-sm text-slate-300">
        Rendered via MathJax so you can recall exactly what the backend computes.
      </p>
      <div className="mt-4 space-y-5">
        {formulas.map((formula) => (
          <div key={formula.title} className="rounded-2xl border border-white/5 bg-slate-950/50 p-4">
            <p className="text-sm font-semibold text-ocean-100">{formula.title}</p>
            <MathJax dynamic className="mt-2 text-base text-white">
              {`\\(${formula.latex}\\)`}
            </MathJax>
          </div>
        ))}
      </div>
    </aside>
  )
}

function StepTimeline({ steps, completed }: { steps: StepPayload[]; completed: boolean }) {
  return (
    <section className="rounded-3xl border border-white/10 bg-slate-900/40 p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-white">Pipeline trace</h2>
        <span className="text-xs uppercase tracking-[0.3em] text-slate-400">
          {completed ? 'Updated' : 'Ready'}
        </span>
      </div>
      <div className="mt-6 space-y-4">
        {steps.map((step, index) => (
          <div key={step.title} className="flex gap-4">
            <div className="flex flex-col items-center">
              <span className="flex h-10 w-10 items-center justify-center rounded-full bg-ocean-500/30 text-sm font-semibold text-ocean-100">
                {index + 1}
              </span>
              {index < steps.length - 1 && <span className="h-full w-px bg-slate-700" />}
            </div>
            <div>
              <p className="text-sm font-semibold text-white">{step.title}</p>
              <p className="text-sm text-slate-300">{step.details}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

function MethodHighlight({ summary }: { summary: string }) {
  return (
    <section className="rounded-3xl border border-white/10 bg-slate-900/40 p-6">
      <p className="text-xs uppercase tracking-[0.35em] text-ocean-200">Adaptive heartbeat-aligned Fourier PCA</p>
      <p className="mt-3 text-lg font-semibold text-white">What this pipeline does</p>
      <p className="mt-2 text-sm text-slate-300">{summary}</p>
    </section>
  )
}

export default App
