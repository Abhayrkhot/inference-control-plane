# Run `make help` for the list.
PY ?= python3.11
BIN := .venv/bin
CLUSTER := icp
# Every kubectl/helm call names the kind context, never whatever context is current.
KUBECTL := kubectl --context kind-$(CLUSTER)
HPC_HOST ?= login.explorer.northeastern.edu
HPC_KEY ?= $(HOME)/.ssh/icp_hpc

.PHONY: help setup test lint api web fake-vllm proxy prom-forward cluster-up images monitoring deploy-api hpc-secret cluster-down

help: ## Show this list
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | awk -F":.*## " "{printf \"  %-14s %s\\n\", \$$1, \$$2}"

setup: ## Create .venv, install the Python packages and the web dependencies
	$(PY) -m venv .venv
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ./api -e ./cli -e ./fake-vllm -r requirements-dev.txt
	npm --prefix web install
	@test -f .env || cp .env.example .env

test: ## Run the Python tests
	$(BIN)/pytest

lint: ## Lint Python, type-check and build the web UI
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	npm --prefix web run build

api: ## Run the API on :8080 against your current kube context (reads .env)
	$(BIN)/uvicorn icp_api.main:app --reload --port 8080 --reload-dir api/src

web: ## Run the web UI on :5173 (proxies /api to :8080)
	npm --prefix web run dev

fake-vllm: ## Run the fake vLLM server on :8002
	$(BIN)/vllm serve facebook/opt-125m --served-model-name demo --port 8002

proxy: ## kubectl proxy on :8001, so a locally running API can reach Services in kind
	$(KUBECTL) proxy --port 8001

prom-forward: ## Forward Prometheus to localhost:9090
	$(KUBECTL) -n monitoring port-forward svc/kps-prometheus 9090

cluster-up: ## Create the kind cluster with namespaces, RBAC and the model cache
	kind create cluster --config deploy/kind/cluster.yaml
	@echo "note: kind switched kubectl's current context to kind-$(CLUSTER)"
	$(KUBECTL) apply -f deploy/k8s/namespaces.yaml
	$(KUBECTL) apply -f deploy/k8s/rbac.yaml -f deploy/k8s/model-cache.yaml

images: ## Build the fake vLLM, gateway and API images and load them into kind
	docker build -t icp-fake-vllm:dev fake-vllm
	docker build -t icp-gateway:dev gateway
	docker build -t icp-api:dev api
	kind load docker-image --name $(CLUSTER) icp-fake-vllm:dev icp-gateway:dev icp-api:dev

monitoring: ## Install Prometheus (kube-prometheus-stack) and the scrape configs
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
	helm upgrade --install --kube-context kind-$(CLUSTER) kps prometheus-community/kube-prometheus-stack -n monitoring --create-namespace -f deploy/prometheus/values.yaml --wait
	$(KUBECTL) apply -f deploy/prometheus/monitors.yaml

deploy-api: ## Run the API inside the cluster instead of on your machine
	$(KUBECTL) apply -f deploy/k8s/api.yaml

hpc-secret: ## Store the HPC SSH key, host key and user as a secret (HPC_USER=<you>)
	@test -n "$(HPC_USER)" || (echo "usage: make hpc-secret HPC_USER=<cluster username>"; exit 1)
	ssh-keygen -F $(HPC_HOST) | grep -v "^#" > .hpc_known_hosts
	for ns in icp-system icp-models; do \
	  $(KUBECTL) -n $$ns create secret generic icp-hpc-ssh --from-file=id_ed25519=$(HPC_KEY) \
	    --from-file=known_hosts=.hpc_known_hosts --from-literal=user=$(HPC_USER) \
	    --dry-run=client -o yaml | $(KUBECTL) apply -f - ; \
	done
	rm -f .hpc_known_hosts

cluster-down: ## Delete the kind cluster
	kind delete cluster --name $(CLUSTER)
