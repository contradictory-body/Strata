/**
 * ToolProgress.tsx — 工具执行进度指示器
 * ★ Round 5：深色模式 + pulse-dot 动画 + 中文工具描述
 */

const TOOL_LABELS: Record<string, { label: string; emoji: string }> = {
  search_job:               { emoji: '🔍', label: '正在搜索岗位信息'   },
  analyze_jd:               { emoji: '📋', label: '正在分析 JD 要求'   },
  company_research:         { emoji: '🏢', label: '正在调研目标公司'   },
  generate_resume:          { emoji: '📄', label: '正在生成简历内容'   },
  mock_interview:           { emoji: '🎤', label: '正在准备面试题目'   },
  evaluate_answer:          { emoji: '✅', label: '正在评估回答质量'   },
  cover_letter_gen:         { emoji: '✉️',  label: '正在撰写求职信'     },
  skill_gap_analysis:       { emoji: '📊', label: '正在分析技能差距'   },
  interview_review:         { emoji: '🔍', label: '正在复盘面试表现'   },
  career_path_planner:      { emoji: '🗺️',  label: '正在规划职业路径'   },
  resume_keyword_optimizer: { emoji: '🔑', label: '正在优化简历关键词' },
  full_preparation_skill:   { emoji: '⚡', label: '综合面试准备中'     },
  application_package_skill:{ emoji: '📦', label: '正在生成申请材料'   },
  post_interview_skill:     { emoji: '📝', label: '正在进行面试复盘'   },
}

export default function ToolProgress({ toolName }: { toolName: string }) {
  const info  = TOOL_LABELS[toolName]
  const emoji = info?.emoji ?? '⚙️'
  const label = info?.label ?? `${toolName} 执行中`

  return (
    <div className="flex gap-3 px-4 animate-fade-in">
      {/* 与 AssistantAvatar 对齐 */}
      <div className="w-7 h-7 rounded-full flex-shrink-0 mt-0.5 flex items-center justify-center
                      bg-accent/10 dark:bg-night-accent/15
                      border border-accent/20 dark:border-night-accent/30">
        <span className="text-accent dark:text-night-accent text-xs font-bold font-serif">S</span>
      </div>

      <div className="flex items-center gap-3 pt-1">
        {/* 三点 pulse 动画 */}
        <span className="flex gap-1">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="inline-block w-1.5 h-1.5 rounded-full
                         bg-accent/50 dark:bg-night-accent/50"
              style={{ animation: `pulse-dot 1.4s ${i * 0.2}s ease-in-out infinite` }}
            />
          ))}
        </span>
        <span className="text-sm text-ink-muted dark:text-night-muted">
          {emoji} {label}
        </span>
      </div>
    </div>
  )
}
