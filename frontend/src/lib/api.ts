import axios from 'axios'
import type { AnalysisResponse } from '../types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api',
  timeout: 60_000,
})

export async function analyzeEcg(file: File): Promise<AnalysisResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const { data } = await api.post<AnalysisResponse>('/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

