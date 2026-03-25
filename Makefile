.PHONY: install run eval experiment clean help

## 安装依赖
install:
	pip install -r requirements.txt

## 启动求职助手
run:
	cd agent/reme_light_job_agent_v2 && python job_agent.py

eval:
	cd agent/reme_light_job_agent_v2 && python eval.py

## 完整消融实验（50 组数据集）
experiment:
	cd experiments && python ablation_study.py

## 生成数据集
dataset:
	cd experiments && python dataset/generate_dataset.py

## 清理运行时产生的本地数据（保留代码）
clean:
	rm -rf .job_agent_v2 .job_agent_v2_eval .reme .ad_ltm_db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo ""
	@echo "Strata — 可用命令："
	@echo "  make install     安装依赖"
	@echo "  make run         启动求职助手"
	@echo "  make eval        快速四维评估（小样本）"
	@echo "  make experiment  完整消融实验（50 组）"
	@echo "  make dataset     生成评估数据集"
	@echo "  make clean       清理运行数据"
	@echo ""
