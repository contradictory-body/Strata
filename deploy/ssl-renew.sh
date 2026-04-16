#!/usr/bin/env bash
# ============================================================
# ssl-renew.sh — SSL 证书手动续签 + nginx 重载
#
# certbot 安装后会自动注册 systemd timer 每天检查续签，
# 此脚本用于手动强制续签或在 cron 中调用。
#
# 使用方式：
#   sudo bash deploy/ssl-renew.sh
#
# 添加到 cron（每天凌晨 2 点检查）：
#   0 2 * * * /home/strata/Strata_v6/deploy/ssl-renew.sh >> /var/log/strata-ssl-renew.log 2>&1
# ============================================================

set -euo pipefail

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始 SSL 证书续签检查..."

# 续签（证书有效期 > 30 天时 certbot 会自动跳过）
certbot renew --quiet --nginx

# 重载 nginx 让新证书生效
systemctl reload nginx

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SSL 续签检查完成"
