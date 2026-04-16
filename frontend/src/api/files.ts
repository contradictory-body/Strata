/**
 * files.ts — 文件上传 API
 * 对应后端 POST /api/files/upload（Round 6 实现）。
 */
import client from './client'
import type { FileUploadResponse } from '../types'

/** 支持的文件类型 */
export const ACCEPTED_TYPES = '.pdf,.docx,.doc,.jpg,.jpeg,.png,.webp,.gif'

/** 最大文件大小：20 MB */
export const MAX_FILE_SIZE = 20 * 1024 * 1024

/** 上传文件，返回解析结果（文本 or base64 图片） */
export async function uploadFile(file: File): Promise<FileUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await client.post<FileUploadResponse>('/api/files/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60_000,
  })
  return res.data
}

export function isImageFile(file: File): boolean {
  return file.type.startsWith('image/')
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024)        return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
