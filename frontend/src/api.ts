import {
  BatchPredictionResult,
  HealthStatus,
  ModelEvaluation,
  ModelInfo,
  PredictionResult,
} from './types'

export class ApiError extends Error {
  code: string
  status: number
  details: Array<{ field?: string; message: string }>

  constructor(
    status: number,
    code: string,
    message: string,
    details: Array<{ field?: string; message: string }> = []
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  const contentType = response.headers.get('content-type') || ''

  if (!response.ok) {
    let code = 'unknown_error'
    let message = `Request failed with status ${response.status}`
    let details: Array<{ field?: string; message: string }> = []

    if (contentType.includes('application/json')) {
      try {
        const errorBody = await response.json()
        if (errorBody?.error) {
          code = errorBody.error.code || code
          message = errorBody.error.message || message
          details = errorBody.error.details || []
        }
      } catch {
      }
    }
    throw new ApiError(response.status, code, message, details)
  }

  return response.json() as Promise<T>
}

export async function fetchHealth(): Promise<HealthStatus> {
  return request<HealthStatus>('/api/health')
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>('/api/model')
}

export async function fetchModelEvaluation(): Promise<ModelEvaluation> {
  return request<ModelEvaluation>('/api/model/evaluation')
}

export async function predictSingle(
  customer: Record<string, unknown>
): Promise<PredictionResult> {
  return request<PredictionResult>('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(customer),
  })
}

export async function predictBatch(
  file: File
): Promise<BatchPredictionResult> {
  const formData = new FormData()
  formData.append('file', file)
  return request<BatchPredictionResult>('/api/predict/batch', {
    method: 'POST',
    body: formData,
  })
}

export async function downloadScoredBatchCsv(file: File): Promise<Blob> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch('/api/predict/batch?format=csv', {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new ApiError(response.status, 'download_failed', 'Failed to generate scored CSV.')
  }
  return response.blob()
}

export async function downloadBatchTemplate(): Promise<Blob> {
  const response = await fetch('/api/predict/batch/template')
  if (!response.ok) {
    throw new ApiError(response.status, 'template_failed', 'Failed to fetch batch template.')
  }
  return response.blob()
}
