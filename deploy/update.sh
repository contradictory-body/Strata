#!/usr/bin/env bash
# ============================================================
# update.sh — Strata 更新脚本
# 在代码有修改时运行，自动拉取最新代码并重新部署。
#
# 使用方式：sudo bash deploy/update.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="/home/strata/venv"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET}   $*"; }

echo -e "\n${BOLD}Strata 更新部署${RESET}\n"

# ── 拉取最新代码 ─────────────────────────────────────────────
info "拉取最新代码..."
sudo -u strata git -C "$PROJECT_DIR" pull --ff-only
success "代码已更新"

# ── 更新 Python 依赖 ─────────────────────────────────────────
info "更新 Python 依赖..."
sudo -u strata "$VENV_DIR/bin/pip" install --quiet -r "$PROJECT_DIR/requirements.txt"
sudo -u strata "$VENV_DIR/bin/pip" install --quiet -r "$PROJECT_DIR/backend/requirements.txt"
success "Python 依赖已更新"

# ── 重新构建前端 ─────────────────────────────────────────────
info "重新构建前端..."
FRONTEND_DIR="$PROJECT_DIR/frontend"
sudo -u strata bash -c "cd '$FRONTEND_DIR' && npm install --silent && npm run build"
success "前端已重新构建"

# ── 重启后端服务 ─────────────────────────────────────────────
info "重启后端服务..."
systemctl restart strata-backend
sleep 2

if systemctl is-active --quiet strata-backend; then
    success "后端服务已重启"
else
    echo "服务启动失败，查看日志："
    journalctl -u strata-backend -n 20 --no-pager
    exit 1
fi

# ── 重载 nginx（静态文件已更新）─────────────────────────────
systemctl reload nginx
success "nginx 已重载"

echo -e "\n${GREEN}${BOLD}✓ 更新完成${RESET}\n"
