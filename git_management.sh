#!/usr/bin/env bash
# 用法：./git-push.sh "你的提交说明"
# 若不填写提交说明，将自动使用时间戳

set -e  # 脚本出错即退出

# 配置 Git 用户信息
git config user.name "yuxuehui"
git config user.email "1170574199@qq.com"

# 确保使用正确的 SSH 密钥
eval "$(ssh-agent -s)" > /dev/null
ssh-add ~/.ssh/id_rsa_yxh > /dev/null 2>&1

# 当前分支
branch=$(git symbolic-ref --short HEAD)

############################################
# 1️⃣ 仅将需要的文件加入暂存区
############################################
# -u：更新已跟踪文件的修改/删除
git add -u

# .      ：递归查找当前目录
# ':!output' 和 ':!output/**' ：排除 output 及其子文件
git add . ':!output' ':!output/**'

############################################
# 2️⃣ 提交
############################################
msg=${1:-"auto commit $(date '+%Y-%m-%d %H:%M:%S')"}
git commit -m "$msg"

############################################
# 3️⃣ 推送
############################################
git push origin "$branch"

# 清理 SSH 代理
ssh-agent -k > /dev/null 2>&1 || true