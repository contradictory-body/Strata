/**
 * FilePreview.tsx — 文件附件预览卡片
 * 显示在 MessageInput 上方：图片显示缩略图，文档显示图标+文件名+大小。
 */
import type { FileUploadItem } from '../types'
import { formatFileSize } from '../api/files'

interface FilePreviewProps {
  item:     FileUploadItem
  onRemove: () => void
}

export default function FilePreview({ item, onRemove }: FilePreviewProps) {
  return (
    <div className="relative inline-flex items-center gap-2.5
                    bg-canvas-sidebar dark:bg-night-panel
                    border border-canvas-border dark:border-night-border
                    rounded-xl px-3 py-2 max-w-xs animate-fade-in group">

      {item.kind === 'image' ? (
        <img
          src={item.preview}
          alt="preview"
          className="w-9 h-9 rounded-lg object-cover flex-shrink-0
                     border border-canvas-border dark:border-night-border"
        />
      ) : (
        <div className="w-9 h-9 rounded-lg flex items-center justify-center
                        bg-accent/10 dark:bg-night-accent/15 flex-shrink-0">
          <DocIcon />
        </div>
      )}

      <div className="min-w-0">
        <p className="text-xs font-medium text-ink dark:text-night-text
                      truncate max-w-[160px] leading-tight">
          {item.file.name}
        </p>
        <p className="text-[11px] text-ink-faint dark:text-night-faint mt-0.5">
          {formatFileSize(item.file.size)}
        </p>
      </div>

      {/* 移除按钮 */}
      <button
        type="button"
        onClick={onRemove}
        aria-label="移除附件"
        style={{ width: 18, height: 18 }}
        className="absolute -top-1.5 -right-1.5 rounded-full
                   flex items-center justify-center
                   bg-ink dark:bg-night-text
                   text-canvas dark:text-night text-[10px] font-bold
                   opacity-0 group-hover:opacity-100
                   hover:bg-accent dark:hover:bg-night-accent transition"
      >
        ×
      </button>
    </div>
  )
}

function DocIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="w-5 h-5 text-accent dark:text-night-accent">
      <path d="M4 4a2 2 0 012-2h5l5 5v9a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"
            stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M11 2v5h5M7 10h6M7 13h4"
            stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  )
}
