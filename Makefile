# kaggle-aas-private-eval-proxy — package + run targets.
#
# Three ways to run the replay + cell scoring:
#   make run-openrouter         local replay over OpenRouter (COST: $, no GPU, needs the key)
#   make run-kaggle-openrouter  push the notebook to Kaggle on the OpenRouter backend + poll
#   make run-kaggle-gpu         push the notebook to Kaggle on the GGUF GPU backend + poll
#
# PUBLIC=1 on the Kaggle run targets also executes the SDK public baseline (evaluate_redteam) in
# the notebook, for an apples-to-apples comparison against the public leaderboard. Default off.
# (Locally, score the public env directly with `make run-openrouter ENV=public`.)

PY ?= python3
NB_SRC := notebooks/kaggle-aas-private-eval-proxy.py
NB_OUT := notebooks/kaggle-aas-private-eval-proxy.ipynb
NB_TMP := notebooks/.configured.py
CANDIDATES ?= candidates.sample.jsonl
TARGETS ?= gpt_oss,gemma
ENV ?= private
REPS ?= 3
CONCURRENCY ?= 8
PUBLIC ?= 0
KERNEL ?= maj0rt0m/private-eval-proxy
SECRET_DATASET ?= maj0rt0m/aas-env
PROXY_REF ?=

# PUBLIC=1 -> Python `True` baked into the notebook's RUN_PUBLIC; anything else -> `False`.
PUBPY := $(if $(filter 1,$(PUBLIC)),True,False)

# PROXY_REF=<branch|tag|sha> pins the notebook's git install to that ref (appends @REF to the
# git+https PROXY_SOURCE). Used to test in-flight fixes from a debug branch before they hit main.
# Empty (default) -> install tracks the repo's default branch, as committed.
PROXY_SED := $(if $(PROXY_REF),-e 's#\(PROXY_SOURCE = "git+[^"@]*\)"#\1@$(PROXY_REF)"#',)

# Load secrets from env.json into the shell BEFORE any command that needs them. Python targets
# pick up env.json via `_sdk` at import, but shell commands (the kaggle CLI) do not — so recipes
# that shell out must prefix `$(LOADENV)`. Fails loudly if env.json is absent (no silent skip).
LOADENV = test -f env.json || { echo "ERROR: env.json missing — copy env.example to env.json"; exit 1; }; eval "$$($(PY) -c 'import json,shlex; print(chr(10).join("export %s=%s" % (k, shlex.quote(str(v))) for k,v in json.load(open("env.json")).items()))')";

.PHONY: help sdk install test notebook run-openrouter run-kaggle-openrouter run-kaggle-gpu push-secret package clean

help:
	@echo "Targets:"
	@echo "  make sdk                  - check the competition SDK is locatable (fresh-install gate)"
	@echo "  make install              - pip install -e . (and dev/kaggle extras)"
	@echo "  make test                 - offline smoke + scoring tests (deterministic agent, mock judge)"
	@echo "  make notebook             - render $(NB_OUT) from the jupytext source"
	@echo "  make run-openrouter       - local replay over OpenRouter, ENV=private|public (COST view)"
	@echo "  make run-kaggle-openrouter- push notebook to Kaggle on the OpenRouter backend + poll"
	@echo "  make run-kaggle-gpu       - push notebook to Kaggle on the GGUF GPU backend + poll"
	@echo "  make push-secret          - (re)create the private OPENROUTER_API_KEY dataset for the kernel"
	@echo "  make package              - notebook + tests, ready to publish"
	@echo ""
	@echo "Args: ENV=private|public  TARGETS=gpt_oss,gemma  REPS=3  PUBLIC=0|1 (Kaggle targets)"
	@echo "  e.g. make run-openrouter ENV=private TARGETS=gpt_oss REPS=3"
	@echo "       make run-kaggle-gpu PUBLIC=1"

# Fresh-install gate: the package must import against a pristine SDK with zero patching.
sdk:
	@AICOMP_SDK_DIR=$(AICOMP_SDK_DIR) $(PY) -c "from private_eval_proxy import _sdk; import aicomp_sdk; print('SDK OK:', aicomp_sdk.__file__)"

install:
	$(PY) -m pip install -e ".[dev]"

# Mock judge keeps this offline; llm-guard 1st-line scanners still run on CPU.
test:
	PYTHONPATH=. PRIVATE_GUARD_JUDGE_BACKEND=mock $(PY) -m pytest -q tests/

# Render the notebook straight from the jupytext source (no config baking).
notebook:
	jupytext --to notebook $(NB_SRC) -o $(NB_OUT)
	@$(GATE)

# Token-literal gate, reused by every render path. Fails loudly if a key leaked into the notebook.
GATE = if grep -nE "sk-or-[A-Za-z0-9]{8}|sk_live_[A-Za-z0-9]{6}|(OPENROUTER_API_KEY|API_KEY)[\"' ]*=[ ]*[\"'][A-Za-z0-9_-]{12}" $(NB_OUT); then \
		echo "ERROR: possible token literal in notebook"; exit 1; \
	else echo "notebook rendered; no token literal"; fi

# OpenRouter = COST. Local pure replay against ONE env (ENV=private|public). Needs internet + key.
# Cell scoring (score_v2) runs inline; budget allocation is a downstream/repo concern, not here.
run-openrouter:
	@mkdir -p artifacts
	@$(LOADENV) PYTHONPATH=. $(PY) -m private_eval_proxy.runner --backend openrouter --env $(ENV) --targets $(TARGETS) --reps $(REPS) --concurrency $(CONCURRENCY) --candidates $(CANDIDATES) --out artifacts/results-$(ENV).json

# Both Kaggle targets bake BACKEND + RUN_PUBLIC into the config cell (Kaggle can't inherit local
# env vars), render, push headlessly, then poll the kernel to a terminal status.
run-kaggle-openrouter:
	@$(MAKE) _kaggle-push BACKEND=openrouter

run-kaggle-gpu:
	@$(MAKE) _kaggle-push BACKEND=kaggle_gguf

# internal: bake config -> render -> token-gate -> push -> poll.
_kaggle-push:
	@sed -e 's/^BACKEND = .*/BACKEND = "$(BACKEND)"  # baked by make/' \
	     -e 's/^RUN_PUBLIC = .*/RUN_PUBLIC = $(PUBPY)  # baked by make/' \
	     $(PROXY_SED) \
	     $(NB_SRC) > $(NB_TMP)
	jupytext --to notebook $(NB_TMP) -o $(NB_OUT)
	@rm -f $(NB_TMP)
	@$(GATE)
	@$(LOADENV) cd notebooks && kaggle kernels push
	@echo "polling $(KERNEL) (Ctrl-C to detach; the run continues on Kaggle)..."
	@$(LOADENV) while true; do \
		s=$$(kaggle kernels status $(KERNEL) 2>/dev/null | tr -d '\r'); \
		echo "  $$s"; \
		case "$$s" in *complete*|*error*|*cancelAcknowledged*) break;; esac; \
		sleep 20; \
	done

# (Re)create the PRIVATE secret dataset that delivers OPENROUTER_API_KEY to the kernel headlessly.
# Builds a scratch env.json trimmed to ONLY OPENROUTER_API_KEY (never the Kaggle token / others)
# in a temp dir, then uploads. The dataset slug is referenced by kernel-metadata dataset_sources.
push-secret:
	@$(LOADENV) d=$$(mktemp -d); $(PY) -c "import json,os; json.dump({'OPENROUTER_API_KEY':os.environ['OPENROUTER_API_KEY']}, open('$$d/env.json','w'))"; \
	printf '{\n  "title": "aas-env",\n  "id": "%s",\n  "licenses": [{"name": "CC0-1.0"}]\n}\n' "$(SECRET_DATASET)" > $$d/dataset-metadata.json; \
	(kaggle datasets create -p $$d --dir-mode skip || kaggle datasets version -p $$d -m "update" --dir-mode skip); \
	rm -rf $$d

package: notebook test
	@echo "package ready: $(NB_OUT) rendered, tests green."

clean:
	rm -rf artifacts __pycache__ private_eval_proxy/__pycache__ tests/__pycache__ *.egg-info $(NB_TMP)
