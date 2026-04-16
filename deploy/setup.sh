#!/usr/bin/env bash
# ============================================================
# setup.sh — Strata 一键部署脚本
# 目标系统：Ubuntu 22.04 / 24.04 LTS
# 硬件：Intel i5-14600KF + 32GB DDR5 + RTX 3090 Ti
#
# 使用方式：
#   1. 以普通用户（非 root）登录服务器
#   2. 克隆项目到 /home/$USER/Strata_v6
#   3. chmod +x deploy/setup.sh && sudo deploy/setup.sh
#
# 脚本功能：
#   - 安装系统依赖（nginx, postgresql, redis, certbot, node, python3）
#   - 创建 strata 系统用户
#   - 创建 Python 虚拟环境，安装所有依赖
#   - 初始化 PostgreSQL 数据库和用户
#   - 构建前端（npm build）
#   - 部署 nginx 配置
#   - 安装 systemd 服务
#   - 申请 Let's Encrypt SSL 证书
#   - 启动所有服务
# ============================================================

set -euo pipefail  # 任一步骤失败立即退出
IFS=$'\n\t'

# ── 颜色输出 ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

# ── 必须以 root 运行 ─────────────────────────────────────────
[[ $EUID -eq 0 ]] || error "请用 sudo 运行此脚本: sudo $0"

# ── 读取配置 ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"   # Strata_v6/

info "项目目录: $PROJECT_DIR"

# 交互式配置（首次部署时填写）
echo -e "\n${BOLD}请输入部署配置：${RESET}"

read -rp "  域名（如 strata.example.com，留空跳过 SSL）: " DOMAIN
read -rp "  PostgreSQL 密码（strata 用户）: "               DB_PASSWORD
read -rp "  JWT 密钥（直接回车自动生成）: "                 JWT_SECRET
read -rp "  LLM API Key: "                                  LLM_API_KEY
read -rp "  LLM Base URL [https://dashscope.aliyuncs.com/compatible-mode/v1]: " LLM_BASE_URL
LLM_BASE_URL="${LLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
read -rp "  LLM 模型名 [qwen-plus]: "                      LLM_MODEL
LLM_MODEL="${LLM_MODEL:-qwen-plus}"
read -rp "  Tavily API Key（可留空）: "                     TAVILY_API_KEY

# 自动生成 JWT 密钥
if [[ -z "$JWT_SECRET" ]]; then
    JWT_SECRET=$(openssl rand -hex 32)
    info "JWT 密钥已自动生成"
fi

# ── 安装系统依赖 ─────────────────────────────────────────────
step "1/9  安装系统依赖"

apt-get update -qq
apt-get install -y -qq \
    curl wget git build-essential \
    python3 python3-pip python3-venv python3-dev \
    postgresql postgresql-contrib \
    redis-server \
    nginx \
    certbot python3-certbot-nginx \
    nodejs npm \
    libpq-dev \
    2>/dev/null

success "系统依赖安装完成"

# Node.js 版本检查（需要 18+）
NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
if [[ "$NODE_VER" -lt 18 ]]; then
    info "升级 Node.js 到 LTS 版本..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
    apt-get install -y nodejs
fi
success "Node.js $(node -v) 就绪"

# ── 创建系统用户 ─────────────────────────────────────────────
step "2/9  创建 strata 系统用户"

if ! id -u strata &>/dev/null; then
    useradd -r -m -s /bin/bash -d /home/strata strata
    success "用户 strata 已创建"
else
    success "用户 strata 已存在，跳过"
fi

# 把项目目录的所有权交给 strata
chown -R strata:strata "$PROJECT_DIR"
success "项目目录权限已设置"

# ── Python 虚拟环境 ──────────────────────────────────────────
step "3/9  创建 Python 虚拟环境并安装依赖"

VENV_DIR="/home/strata/venv"

if [[ ! -d "$VENV_DIR" ]]; then
    sudo -u strata python3 -m venv "$VENV_DIR"
    success "虚拟环境已创建: $VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# 升级 pip
sudo -u strata "$PIP" install --quiet --upgrade pip setuptools wheel

# 安装项目依赖（顺序：先 agent，再 backend）
info "安装 agent 依赖..."
sudo -u strata "$PIP" install --quiet -r "$PROJECT_DIR/requirements.txt"

info "安装 backend 依赖..."
sudo -u strata "$PIP" install --quiet -r "$PROJECT_DIR/backend/requirements.txt"

success "Python 依赖安装完成"

# ── PostgreSQL 初始化 ────────────────────────────────────────
step "4/9  初始化 PostgreSQL"

systemctl enable postgresql --quiet
systemctl start  postgresql

# 创建数据库用户和数据库
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='strata'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER strata WITH PASSWORD '$DB_PASSWORD';"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='strata_db'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE strata_db OWNER strata;"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE strata_db TO strata;" -q

success "PostgreSQL 数据库初始化完成"

# ── Redis 配置 ───────────────────────────────────────────────
step "5/9  配置 Redis"

systemctl enable redis-server --quiet
systemctl start  redis-server

# 确认 Redis 可连接
redis-cli ping | grep -q PONG && success "Redis 连接正常" || \
    error "Redis 启动失败，请检查 systemctl status redis-server"

# ── 写入环境变量 ─────────────────────────────────────────────
step "6/9  生成 backend/.env"

ENV_FILE="$PROJECT_DIR/backend/.env"

cat > "$ENV_FILE" << EOF
# ── 由 setup.sh 自动生成 ── $(date '+%Y-%m-%d %H:%M:%S') ──
DATABASE_URL=postgresql+asyncpg://strata:${DB_PASSWORD}@localhost:5432/strata_db
REDIS_URL=redis://localhost:6379/0

JWT_SECRET_KEY=${JWT_SECRET}
JWT_ALGORITHM=HS256
JWT_EXPIRE_DAYS=7

CORS_ORIGINS=https://${DOMAIN:-localhost}

LLM_API_KEY=${LLM_API_KEY}
LLM_BASE_URL=${LLM_BASE_URL}
LLM_MODEL=${LLM_MODEL}
TAVILY_API_KEY=${TAVILY_API_KEY:-}

DATA_ROOT=${PROJECT_DIR}/data

APP_HOST=127.0.0.1
APP_PORT=8000
APP_DEBUG=false
EOF

chown strata:strata "$ENV_FILE"
chmod 600 "$ENV_FILE"   # 只有 strata 用户可读（包含密钥）

success "环境变量已写入 $ENV_FILE"

# 创建数据目录
mkdir -p "$PROJECT_DIR/data"
chown strata:strata "$PROJECT_DIR/data"
success "数据目录已创建: $PROJECT_DIR/data"

# ── 构建前端 ─────────────────────────────────────────────────
step "7/9  构建前端"

FRONTEND_DIR="$PROJECT_DIR/frontend"

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    info "安装 npm 依赖..."
    sudo -u strata bash -c "cd '$FRONTEND_DIR' && npm install --silent"
fi

# 写入前端 .env（生产环境不需要 VITE_* 变量，nginx 代理处理）
cat > "$FRONTEND_DIR/.env" << EOF
VITE_API_BASE_URL=
VITE_WS_BASE_URL=
EOF

info "构建 React 应用..."
sudo -u strata bash -c "cd '$FRONTEND_DIR' && npm run build"

success "前端构建完成: $FRONTEND_DIR/dist"

# ── nginx 配置 ───────────────────────────────────────────────
step "8/9  配置 nginx"

NGINX_CONF="/etc/nginx/sites-available/strata"
NGINX_ENABLED="/etc/nginx/sites-enabled/strata"

# 替换模板中的域名占位符
if [[ -n "$DOMAIN" ]]; then
    sed "s/YOUR_DOMAIN/$DOMAIN/g" "$SCRIPT_DIR/nginx.conf" > "$NGINX_CONF"
    success "nginx 配置已写入 $NGINX_CONF"

    # 申请 Let's Encrypt 证书
    if [[ ! -d "/etc/letsencrypt/live/$DOMAIN" ]]; then
        info "申请 SSL 证书（域名: $DOMAIN）..."
        info "注意：请确保域名 A 记录已指向本机 IP，否则验证会失败"

        # 先用 HTTP 验证（certbot 会临时启动）
        certbot certonly \
            --nginx \
            --non-interactive \
            --agree-tos \
            --email "admin@${DOMAIN}" \
            --domains "$DOMAIN" \
            --redirect \
            2>&1 | tail -5

        success "SSL 证书已申请: /etc/letsencrypt/live/$DOMAIN"
    else
        success "SSL 证书已存在，跳过申请"
    fi

    # 配置自动续签（certbot 安装时已自动添加 systemd timer，这里仅确认）
    systemctl is-enabled certbot.timer &>/dev/null && \
        success "SSL 证书自动续签已启用" || \
        warn "请手动启用 certbot.timer: systemctl enable certbot.timer"
else
    # 无域名：生成仅用于测试的 HTTP 配置（去掉 SSL 部分）
    warn "未提供域名，使用 HTTP-only 配置（仅供本地测试）"
    cat > "$NGINX_CONF" << 'NGINX_HTTP'
upstream strata_backend { server 127.0.0.1:8000; keepalive 32; }
server {
    listen 80;
    server_name _;
    client_max_body_size 25M;
    gzip on; gzip_types text/plain text/css application/json application/javascript;

    location /ws/ {
        proxy_pass http://strata_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 90s;
    }
    location /api/ {
        proxy_pass http://strata_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
    location / {
        root FRONTEND_DIST;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
}
NGINX_HTTP
    # 替换前端 dist 路径
    sed -i "s|FRONTEND_DIST|$FRONTEND_DIR/dist|g" "$NGINX_CONF"
fi

# 启用站点
[[ -L "$NGINX_ENABLED" ]] || ln -s "$NGINX_CONF" "$NGINX_ENABLED"

# 禁用默认站点（避免端口冲突）
[[ -L "/etc/nginx/sites-enabled/default" ]] && \
    rm -f /etc/nginx/sites-enabled/default

# 测试 nginx 配置
nginx -t && success "nginx 配置语法检查通过" || error "nginx 配置有误，请检查 $NGINX_CONF"

systemctl enable nginx --quiet
systemctl reload nginx

success "nginx 已启动/重载"

# ── systemd 服务 ─────────────────────────────────────────────
step "9/9  安装并启动 systemd 服务"

SERVICE_FILE="/etc/systemd/system/strata-backend.service"

# 替换模板中的路径占位符（服务文件中硬编码了 /home/strata）
sed "s|/home/strata/Strata_v6|$PROJECT_DIR|g; \
     s|/home/strata/venv|$VENV_DIR|g; \
     s|User=strata|User=strata|g" \
    "$SCRIPT_DIR/strata-backend.service" > "$SERVICE_FILE"

systemctl daemon-reload
systemctl enable strata-backend
systemctl restart strata-backend

# 等待服务启动
sleep 3

if systemctl is-active --quiet strata-backend; then
    success "strata-backend 服务已启动"
else
    error "服务启动失败，请查看日志: journalctl -u strata-backend -n 50"
fi

# ── 数据库建表 ───────────────────────────────────────────────
info "初始化数据库表结构..."
PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/agent/reme_light_job_agent_v2" \
    sudo -u strata "$PYTHON" -c "
import asyncio, sys
sys.path.insert(0, '$PROJECT_DIR')
from backend.database import init_db
asyncio.run(init_db())
print('数据库表初始化完成')
"
success "数据库表结构已创建"

# ── 完成 ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║     Strata 部署完成！                   ║${RESET}"
echo -e "${BOLD}${GREEN}╚════════════════════════════════════════╝${RESET}"
echo ""

if [[ -n "$DOMAIN" ]]; then
    echo -e "  访问地址：${CYAN}https://$DOMAIN${RESET}"
else
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    echo -e "  访问地址：${CYAN}http://$LOCAL_IP${RESET}（无域名模式）"
fi

echo ""
echo -e "  常用命令："
echo -e "    ${BOLD}查看后端日志${RESET}   sudo journalctl -u strata-backend -f"
echo -e "    ${BOLD}重启后端${RESET}       sudo systemctl restart strata-backend"
echo -e "    ${BOLD}重载 nginx${RESET}     sudo systemctl reload nginx"
echo -e "    ${BOLD}服务状态${RESET}       sudo systemctl status strata-backend"
echo ""
