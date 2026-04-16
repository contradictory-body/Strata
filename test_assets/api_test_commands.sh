#!/usr/bin/env bash
# ============================================================
# Strata v7 API 测试命令集
# 在项目根目录执行，后端需已运行在 localhost:8000
# 需要先设置：export APP_DEBUG=true（开启 /docs）
#
# 使用方式：
#   1. 逐条复制到终端执行
#   2. 或者 bash test_assets/api_test_commands.sh
# ============================================================

BASE="http://localhost:8000"

echo "========================================"
echo " Strata v7 API 自动化测试"
echo "========================================"

# ── 工具函数 ──────────────────────────────────────────────
check() {
  local name="$1"
  local code="$2"
  local expect="$3"
  if echo "$code" | grep -q "$expect"; then
    echo "  ✓  $name"
  else
    echo "  ✗  $name  (返回: $code)"
  fi
}

# ── T1：健康检查 ───────────────────────────────────────────
echo ""
echo "【1】服务健康检查"
HEALTH=$(curl -s "$BASE/api/health")
check "服务在线" "$HEALTH" '"status":"ok"'
check "版本号" "$HEALTH" '"version":"2.0.0"'
echo "    响应: $HEALTH"

# ── T2：用户注册 ───────────────────────────────────────────
echo ""
echo "【2】用户注册"
REG=$(curl -s -X POST "$BASE/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@strata.dev","password":"Test1234!"}')
check "注册成功" "$REG" '"access_token"'
check "返回用户名" "$REG" '"username":"testuser"'
echo "    响应: $REG" | head -c 200

# 提取 Token
TOKEN=$(echo "$REG" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('access_token',''))" 2>/dev/null)
echo "    Token: ${TOKEN:0:40}..."

# ── T3：用户登录 ───────────────────────────────────────────
echo ""
echo "【3】用户名登录"
LOGIN=$(curl -s -X POST "$BASE/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"Test1234!"}')
check "登录成功" "$LOGIN" '"access_token"'
TOKEN=$(echo "$LOGIN" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('access_token',''))" 2>/dev/null)

echo ""
echo "【3b】邮箱登录"
LOGIN2=$(curl -s -X POST "$BASE/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"test@strata.dev","password":"Test1234!"}')
check "邮箱登录成功" "$LOGIN2" '"access_token"'

echo ""
echo "【3c】错误密码（期望 401）"
BADLOGIN=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"wrongpassword"}')
check "错误密码返回 401" "$BADLOGIN" "401"

# ── T4：获取当前用户 ───────────────────────────────────────
echo ""
echo "【4】获取当前用户 /me"
ME=$(curl -s "$BASE/api/auth/me" -H "Authorization: Bearer $TOKEN")
check "用户名正确" "$ME" '"username":"testuser"'
check "邮箱正确" "$ME" '"email":"test@strata.dev"'
check "账号启用" "$ME" '"is_active":true'

# 无 Token 访问（期望 403/401）
NTOKEN=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/api/auth/me")
check "无 Token 返回 401" "$NTOKEN" "401"

# ── T5：会话管理 ───────────────────────────────────────────
echo ""
echo "【5】创建会话"
SESSION=$(curl -s -X POST "$BASE/api/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}')
check "会话创建成功" "$SESSION" '"id"'
SESSION_ID=$(echo "$SESSION" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null)
echo "    Session ID: $SESSION_ID"

echo ""
echo "【5b】创建带标题的会话"
SESSION2=$(curl -s -X POST "$BASE/api/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Python面试准备"}')
check "带标题会话创建成功" "$SESSION2" '"title":"Python面试准备"'
SESSION_ID2=$(echo "$SESSION2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null)

echo ""
echo "【5c】获取会话列表"
SESSIONS=$(curl -s "$BASE/api/sessions" -H "Authorization: Bearer $TOKEN")
check "会话列表返回" "$SESSIONS" '"id"'
SESSION_COUNT=$(echo "$SESSIONS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
echo "    当前会话数: $SESSION_COUNT"

echo ""
echo "【5d】获取会话历史消息（初始为空）"
MSGS=$(curl -s "$BASE/api/sessions/$SESSION_ID/messages" \
  -H "Authorization: Bearer $TOKEN")
check "历史消息接口正常" "$MSGS" "[]"

echo ""
echo "【5e】访问不存在的会话（期望 404）"
NOTFOUND=$(curl -s -o /dev/null -w "%{http_code}" \
  "$BASE/api/sessions/00000000-0000-0000-0000-000000000000/messages" \
  -H "Authorization: Bearer $TOKEN")
check "不存在会话返回 404" "$NOTFOUND" "404"

# ── T6：画像 API ───────────────────────────────────────────
echo ""
echo "【6】获取初始画像"
PROFILE=$(curl -s "$BASE/api/profile" -H "Authorization: Bearer $TOKEN")
check "画像接口正常" "$PROFILE" '"raw"'

echo ""
echo "【6b】更新画像"
UPDATE=$(curl -s -X PUT "$BASE/api/profile" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "updates": {
      "目标岗位": "Python 后端工程师（AI 方向）",
      "目标城市": "北京",
      "技术栈": "Python、FastAPI、PostgreSQL、Redis、基础 LLM 应用",
      "薪资预期": "25K-35K"
    }
  }')
check "画像更新成功" "$UPDATE" '"updated_fields"'
echo "    更新字段: $(echo $UPDATE | python3 -c 'import sys,json; print(json.load(sys.stdin).get("updated_fields",[]))' 2>/dev/null)"

echo ""
echo "【6c】验证画像已保存"
PROFILE2=$(curl -s "$BASE/api/profile" -H "Authorization: Bearer $TOKEN")
check "目标岗位已保存" "$PROFILE2" 'Python 后端工程师'
check "目标城市已保存" "$PROFILE2" '"北京"'

echo ""
echo "【6d】获取画像摘要"
SUMMARY=$(curl -s "$BASE/api/profile/summary" -H "Authorization: Bearer $TOKEN")
check "摘要接口正常" "$SUMMARY" '"summary"'

# ── T7：文件上传 ───────────────────────────────────────────
echo ""
echo "【7】文件上传 - 无效格式（期望 400）"
BADFILE=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "$BASE/api/files/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/etc/hostname;type=text/plain;filename=test.exe")
check "非法格式返回 400" "$BADFILE" "400"

echo ""
echo "【7b】文件上传 - 文本文件（模拟 JD）"
# 创建临时测试文件
cat > /tmp/test_jd.txt << 'TXT'
职位：Python 后端工程师
要求：3年Python经验，熟悉FastAPI、PostgreSQL、Redis
薪资：25K-35K，北京
TXT

UPLOAD=$(curl -s -X POST "$BASE/api/files/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/tmp/test_jd.txt;type=text/plain;filename=jd.txt")
echo "    上传结果: $UPLOAD" | head -c 300

# ── T8：删除会话 ───────────────────────────────────────────
echo ""
echo "【8】删除会话 $SESSION_ID2"
DEL=$(curl -s -o /dev/null -w "%{http_code}" \
  -X DELETE "$BASE/api/sessions/$SESSION_ID2" \
  -H "Authorization: Bearer $TOKEN")
check "会话删除成功（204）" "$DEL" "204"

echo ""
echo "【8b】验证已删除（期望 404）"
DELETED=$(curl -s -o /dev/null -w "%{http_code}" \
  "$BASE/api/sessions/$SESSION_ID2/messages" \
  -H "Authorization: Bearer $TOKEN")
check "删除后访问返回 404" "$DELETED" "404"

# ── T9：注销 ──────────────────────────────────────────────
echo ""
echo "【9】登出"
LOGOUT=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "$BASE/api/auth/logout" \
  -H "Authorization: Bearer $TOKEN")
check "登出成功（204）" "$LOGOUT" "204"

# ── 汇总 ──────────────────────────────────────────────────
echo ""
echo "========================================"
echo " API 测试完成"
echo " SESSION_ID=$SESSION_ID （用于 WebSocket 测试）"
echo " TOKEN=$TOKEN"
echo "========================================"
