export interface PlotPayload {
  title: string
  image_base64: string
}

export interface FormulaPayload {
  title: string
  latex: string
}

export interface StepPayload {
  title: string
  details: string
}

export interface AnalysisResponse {
  fft_frequencies: number[]
  fft_magnitudes: number[]
  pca_components: number[][]
  explained_variance: number[]
  plots: PlotPayload[]
  formulas: FormulaPayload[]
  steps: StepPayload[]
  heartbeat_count: number
  r_peak_indices: number[]
  method_summary: string
}

