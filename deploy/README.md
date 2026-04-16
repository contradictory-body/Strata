# Strata 部署说明

## 前置条件

- Ubuntu 22.04 / 24.04 LTS（推荐 24.04）
- 公网 IP（如需 HTTPS）
- 域名已解析到服务器 IP（申请 SSL 证书需要）
- 以普通用户登录（脚本内部会用 sudo）

## 一键部署

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/Strata_v6.git
cd Strata_v6

# 2. 赋予脚本执行权限
chmod +x deploy/setup.sh deploy/update.sh deploy/ssl-renew.sh

# 3. 运行部署脚本（按照交互提示填写配置）
sudo bash deploy/setup.sh
```

脚本会依次完成：系统依赖安装 → 用户创建 → Python 环境 → PostgreSQL 初始化 → Redis 启动 → 环境变量配置 → 前端构建 → nginx 配置 → SSL 证书申请 → systemd 服务安装。

## 部署后验证

```bash
# 查看服务状态
sudo systemctl status strata-backend
sudo systemctl status nginx

# 查看实时日志
sudo journalctl -u strata-backend -f

# 测试健康检查接口
curl https://your-domain.com/api/health
```

## 目录结构（部署后）

```
/home/strata/
├── Strata_v6/          # 项目根目录
│   ├── backend/        # FastAPI 后端
│   │   └── .env        # 环境变量（含密钥，权限 600）
│   ├── frontend/
│   │   └── dist/       # Vite 构建产物（nginx 托管）
│   ├── data/           # 用户数据（记忆库、画像）
│   │   └── {user_id}/
│   │       └── agent/
│   │           └── PROFILE.md
│   └── deploy/
└── venv/               # Python 虚拟环境
```

## 常用命令

| 操作 | 命令 |
|------|------|
| 查看后端日志 | `sudo journalctl -u strata-backend -f` |
| 重启后端 | `sudo systemctl restart strata-backend` |
| 重载 nginx | `sudo systemctl reload nginx` |
| 更新部署 | `sudo bash deploy/update.sh` |
| 手动续签 SSL | `sudo bash deploy/ssl-renew.sh` |
| 检查端口占用 | `sudo ss -tlnp \| grep -E '80\|443\|8000'` |

## 端口说明

| 端口 | 用途 | 对外暴露 |
|------|------|---------|
| 80 | HTTP（重定向到 443） | ✅ |
| 443 | HTTPS（nginx） | ✅ |
| 8000 | uvicorn（后端） | ❌ 仅 127.0.0.1 |
| 5432 | PostgreSQL | ❌ 仅 localhost |
| 6379 | Redis | ❌ 仅 localhost |

## 故障排查

**后端启动失败**
```bash
sudo journalctl -u strata-backend -n 50
# 常见原因：.env 未配置、PostgreSQL/Redis 未启动、Python 依赖缺失
```

**nginx 502 Bad Gateway**
```bash
sudo systemctl status strata-backend   # 确认后端在运行
sudo nginx -t                          # 检查配置语法
```

**WebSocket 连接失败**
- 检查 nginx.conf 中 `/ws/` location 的 `Upgrade` 和 `Connection` 头
- 确认防火墙放行了 443 端口：`sudo ufw allow 443`

**SSL 证书申请失败**
- 确认域名 A 记录已生效：`dig your-domain.com`
- 确认 80 端口可访问：`curl http://your-domain.com`
